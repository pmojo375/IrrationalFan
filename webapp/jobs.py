"""Small, persistent single-worker queue independent of HTTP request lifetimes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

log = logging.getLogger(__name__)
TERMINAL = ("complete", "failed")


def now():
    return datetime.now(timezone.utc).isoformat()


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, path, limit=32, queue_limit=8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The engine and its observation cache are deliberately single-process.
        self._lease = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lease.close()
            raise RuntimeError("Discovery store is already in use. Run Uvicorn with one worker.")
        self.limit, self.queue_limit = limit, queue_limit
        self._lock = threading.RLock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discovery")
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_key TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                metadata TEXT NOT NULL, result TEXT, error TEXT, error_status INTEGER
            )''')
            db.execute("CREATE INDEX IF NOT EXISTS jobs_key ON jobs(request_key)")
            db.execute("UPDATE jobs SET status='failed', finished_at=?, error=?, error_status=503 "
                       "WHERE status IN ('queued','running')",
                       (now(), "Search interrupted by a server restart. Retry this discovery."))
            self._prune(db)

    @contextmanager
    def _db(self):
        with self._lock:
            db = sqlite3.connect(self.path, timeout=10)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def _prune(self, db):
        db.execute("DELETE FROM jobs WHERE id IN (SELECT id FROM jobs "
                   "WHERE status IN ('complete','failed') ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                   (self.limit,))

    def submit(self, request_key, metadata, work):
        with self._lock:
            if self._closed:
                raise RuntimeError("Discovery queue is shutting down")
            with self._db() as db:
                row = db.execute("SELECT * FROM jobs WHERE request_key=? AND status != 'failed' "
                                 "ORDER BY created_at DESC LIMIT 1", (request_key,)).fetchone()
                if row:
                    return self._view(row), True
                active = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
                if active >= self.queue_limit:
                    raise QueueFull()
                job_id = uuid4().hex
                db.execute("INSERT INTO jobs(id,request_key,status,created_at,metadata) VALUES (?,?,?,?,?)",
                           (job_id, request_key, "queued", now(), json.dumps(metadata)))
            self._executor.submit(self._run, job_id, work)
            return self.get(job_id), False

    def _run(self, job_id, work):
        with self._db() as db:
            db.execute("UPDATE jobs SET status='running',started_at=? WHERE id=?", (now(), job_id))
        try:
            result = json.dumps(work(), allow_nan=False)
        except Exception as exc:
            log.exception("Discovery job %s failed", job_id)
            status = getattr(exc, "status_code", 500)
            message = getattr(exc, "detail", None)
            if status == 500:
                message = "Discovery failed unexpectedly. Please retry; details are in the service logs."
            with self._db() as db:
                db.execute("UPDATE jobs SET status='failed',finished_at=?,error=?,error_status=? WHERE id=?",
                           (now(), str(message), status, job_id))
                self._prune(db)
        else:
            with self._db() as db:
                db.execute("UPDATE jobs SET status='complete',finished_at=?,result=? WHERE id=?",
                           (now(), result, job_id))
                self._prune(db)

    def _view(self, row, include_result=True):
        job = {key: row[key] for key in ("id", "status", "created_at", "started_at", "finished_at", "error", "error_status")}
        job.update(json.loads(row["metadata"]))
        if include_result and row["result"]:
            job["result"] = json.loads(row["result"])
        if row["status"] == "queued":
            with self._db() as db:
                job["queue_position"] = db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running') AND created_at<=?",
                    (row["created_at"],)).fetchone()[0]
        return job

    def get(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._view(row) if row else None

    def list(self):
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [self._view(row, include_result=False) for row in rows]

    def clear_completed(self):
        with self._db() as db:
            count = db.execute("DELETE FROM jobs WHERE status IN ('complete','failed')").rowcount
            return count

    def close(self):
        with self._lock:
            self._closed = True
        # In-flight work owns the queue, even after an HTTP disconnect.
        self._executor.shutdown(wait=True)
        self._lease.close()
