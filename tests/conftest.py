import pytest


@pytest.fixture(autouse=True)
def isolated_web_jobs(monkeypatch, tmp_path):
    from webapp import app as module
    from webapp.jobs import JobStore
    store = JobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(module, "_JOBS", store)
    yield
    store.close()
