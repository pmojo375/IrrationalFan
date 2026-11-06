"""Regressions for disconnected searches, duplicate retries and durable history."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from webapp import app as module
from webapp.jobs import JobStore


@pytest.fixture
def engine(monkeypatch):
    game = {"game_id": "123", "home_id": 1, "away_id": 2,
            "home_team": "Michigan State", "away_team": "Minnesota", "date": "2026-11-07"}
    monkeypatch.setattr(module, "_read_games", lambda now: [game])
    state = SimpleNamespace(started=threading.Event(), release=threading.Event(), calls=0)

    def discover(**kwargs):
        state.calls += 1
        state.started.set()
        assert state.release.wait(5), "test failed to release the engine"
        return SimpleNamespace(team=kwargs["team"], opponent=kwargs["opponent"],
                               report="Saved report", discoveries=[])

    monkeypatch.setattr(module, "discover", discover)
    yield state
    state.release.set()


def wait_done(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("complete", "failed"):
            return job
        time.sleep(.01)
    pytest.fail("Job did not finish")


def test_disconnect_retry_and_queue_are_independent_of_http(engine):
    client = TestClient(module.app)
    first = client.post("/api/jobs", json={"game_id": "123"})
    assert first.status_code == 202
    job_id = first.json()["id"]
    assert engine.started.wait(2)
    # A new connection retries the same request with an explicit team/default sliders.
    client.close()
    client = TestClient(module.app)
    retry = client.post("/api/jobs", json={"game_id": "123", "team_id": 1, "weirdness": 80})
    assert retry.status_code == 202
    assert retry.json()["id"] == job_id
    other = client.post("/api/jobs", json={"game_id": "123", "team_id": 2})
    assert other.status_code == 202
    assert other.json()["status"] == "queued"
    assert other.json()["queue_position"] == 2
    assert engine.calls == 1
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/health").status_code == 200
    # Clearing history does not wait for or delete active jobs.
    assert client.delete("/api/cache").json() == {"cleared": 0}
    engine.release.set()
    assert wait_done(client, job_id)["result"]["team"] == "Michigan State"
    assert wait_done(client, other.json()["id"])["result"]["team"] == "Minnesota"
    assert engine.calls == 2
    cached = client.post("/api/jobs", json={"game_id": "123"})
    assert cached.status_code == 200
    assert cached.json()["id"] == job_id
    assert engine.calls == 2
    assert "result" not in client.get("/api/jobs").json()["jobs"][0]


def test_queue_full_still_allows_following_existing_job(engine):
    module._JOBS.queue_limit = 2
    client = TestClient(module.app)
    a = client.post("/api/jobs", json={"game_id": "123"}).json()
    assert engine.started.wait(2)
    client.post("/api/jobs", json={"game_id": "123", "team_id": 2})
    full = client.post("/api/jobs", json={"game_id": "123", "max_conditions": 5})
    assert full.status_code == 429
    assert full.headers["Retry-After"] == "10"
    assert client.post("/api/jobs", json={"game_id": "123"}).json()["id"] == a["id"]
    assert client.post("/api/jobs", json={"game_id": "123", "feature_groups": []}).status_code == 422


def test_simultaneous_duplicate_posts_only_run_once(engine):
    def post(_):
        return TestClient(module.app).post("/api/jobs", json={"game_id": "123"}).json()["id"]
    with ThreadPoolExecutor(max_workers=5) as pool:
        ids = list(pool.map(post, range(5)))
    assert len(set(ids)) == 1
    assert engine.started.wait(2)
    assert engine.calls == 1


def test_failed_job_can_retry_and_does_not_stop_queue(engine, monkeypatch):
    def fail(**kwargs):
        raise ValueError("No qualifying history for this matchup")
    monkeypatch.setattr(module, "discover", fail)
    client = TestClient(module.app)
    job = client.post("/api/jobs", json={"game_id": "123"}).json()
    failed = wait_done(client, job["id"])
    assert failed["error_status"] == 422
    assert "No qualifying history" in failed["error"]
    again = client.post("/api/jobs", json={"game_id": "123"}).json()
    assert again["id"] != job["id"]
    wait_done(client, again["id"])
    assert client.get("/api/jobs/unknown").status_code == 404


def test_unexpected_error_does_not_expose_internal_details(engine, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("private internal details")
    monkeypatch.setattr(module, "discover", fail)
    client = TestClient(module.app)
    job = client.post("/api/jobs", json={"game_id": "123"}).json()
    failed = wait_done(client, job["id"])
    assert failed["error_status"] == 500
    assert "private internal" not in failed["error"]


def test_completed_results_survive_reopening_store(engine, monkeypatch):
    client = TestClient(module.app)
    engine.release.set()
    job = client.post("/api/jobs", json={"game_id": "123"}).json()
    wait_done(client, job["id"])
    path = module._JOBS.path
    module._JOBS.close()
    reopened = JobStore(path)
    monkeypatch.setattr(module, "_JOBS", reopened)
    try:
        assert client.get(f'/api/jobs/{job["id"]}').json()["result"]["report"] == "Saved report"
        assert client.post("/api/jobs", json={"game_id": "123"}).json()["id"] == job["id"]
        assert engine.calls == 1
        assert client.delete("/api/cache").json() == {"cleared": 1}
        assert client.get(f'/api/jobs/{job["id"]}').status_code == 404
    finally:
        reopened.close()


def test_restart_marks_unfinished_jobs_retryable(tmp_path):
    path = tmp_path / "restart.sqlite3"
    store = JobStore(path)
    store.close()
    with sqlite3.connect(path) as db:
        for state in ("queued", "running"):
            db.execute("INSERT INTO jobs(id,request_key,status,created_at,metadata) VALUES(?,?,?,?,?)",
                       (state, state, state, datetime.now(timezone.utc).isoformat(), json.dumps({"team": "MSU"})))
    reopened = JobStore(path)
    try:
        for job in reopened.list():
            assert job["status"] == "failed"
            assert "server restart" in job["error"]
            assert job["error_status"] == 503
    finally:
        reopened.close()


def test_store_rejects_second_process_owner_and_bounds_history(tmp_path):
    store = JobStore(tmp_path / "bounded.sqlite3", limit=2)
    try:
        with pytest.raises(RuntimeError, match="one worker"):
            JobStore(store.path)
        for n in range(4):
            job, _ = store.submit(str(n), {}, lambda: {"report": "ok"})
            deadline = time.monotonic() + 3
            while store.get(job["id"])["status"] not in ("complete", "failed"):
                assert time.monotonic() < deadline
                time.sleep(.01)
        assert len(store.list()) == 2
    finally:
        store.close()


def test_changed_dataset_does_not_reuse_saved_result(engine, monkeypatch, tmp_path):
    from dataclasses import replace
    data = tmp_path / "processed.parquet"
    data.write_bytes(b"first snapshot")
    original = module._effective_config
    monkeypatch.setattr(module, "_effective_config", lambda request: replace(original(request), data_path=data))
    client = TestClient(module.app)
    engine.release.set()
    first = client.post("/api/jobs", json={"game_id": "123"}).json()
    wait_done(client, first["id"])
    data.write_bytes(b"updated, larger snapshot")
    second = client.post("/api/jobs", json={"game_id": "123"}).json()
    wait_done(client, second["id"])
    assert first["id"] != second["id"]
    assert engine.calls == 2


def test_observations_reload_after_dataset_replacement(monkeypatch, tmp_path):
    from discovery.sports.college_football import matchup
    data = tmp_path / "processed.parquet"
    data.write_text("first")
    monkeypatch.setattr(matchup.pd, "read_parquet", lambda path: Path(path).read_text())
    monkeypatch.setattr(matchup, "enrich_team_games", lambda raw: raw)
    try:
        assert matchup.load_observations(data) == "first"
        data.write_text("second snapshot")
        assert matchup.load_observations(data) == "second snapshot"
    finally:
        matchup.clear_data_cache()
