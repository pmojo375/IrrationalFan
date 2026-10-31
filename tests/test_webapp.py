"""API smoke tests using isolated schedule/engine fixtures."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pandas as pd

from webapp import app as module


def test_schedule_and_discovery(monkeypatch, tmp_path):
    schedule = tmp_path / "cfb_schedules_2026.parquet"
    pd.DataFrame([{
        "game_id": 123, "season": 2026, "week": 10,
        "start_date": "2026-11-07T17:00:00Z",
        "home_id": 1, "away_id": 2,
        "home_team": "Michigan State", "away_team": "Minnesota",
        "status": "STATUS_SCHEDULED", "home_score": None, "away_score": None,
        "neutral_site": False,
    }]).to_parquet(schedule)
    monkeypatch.setattr(module, "_schedule_files", lambda now: [schedule])
    monkeypatch.setattr(module, "discover", lambda **kwargs: SimpleNamespace(
        team=kwargs["team"], opponent=kwargs["opponent"],
        report="A ridiculous but true statistic", discoveries=[1]))
    client = TestClient(module.app)
    response = client.get("/api/games?mode=week")
    assert response.status_code == 200
    assert response.json()["games"][0]["game_id"] == "123"
    report = client.post("/api/discover", json={"game_id": "123", "team_id": 2})
    assert report.status_code == 200
    assert report.json()["team"] == "Minnesota"
    assert report.json()["count"] == 1
    assert client.post("/api/discover", json={"game_id": "123", "team_id": 99}).status_code == 422
    assert client.post("/api/discover", json={"game_id": "not-real"}).status_code == 404


def test_home_page():
    response = TestClient(module.app).get("/")
    assert response.status_code == 200
    assert "Irrational Fan" in response.text
