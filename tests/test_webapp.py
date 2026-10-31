"""API smoke tests using isolated schedule/engine fixtures."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import pandas as pd

from webapp import app as module
from discovery.engine.models import Discovery, Condition, Operator, MatchingGame


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
        report="A ridiculous but true statistic", discoveries=[Discovery(
            scope="team", conditions=(Condition("week", Operator.EQ, 10),),
            wins=1, losses=0, n_games=1, win_pct=1.0, streak=1,
            ranking_score=10.0, description="One historical win",
            matching_games=[MatchingGame(2025, "2025-11-01", "Minnesota", "Iowa", 21, 14, True)],
        )]))
    module._REPORT_CACHE.clear()
    client = TestClient(module.app)
    response = client.get("/api/games?mode=week")
    assert response.status_code == 200
    assert response.json()["games"][0]["game_id"] == "123"
    report = client.post("/api/discover", json={"game_id": "123", "team_id": 2})
    assert report.status_code == 200
    assert report.json()["team"] == "Minnesota"
    assert report.json()["count"] == 1
    assert report.json()["discoveries"][0]["scope"] == "team"
    assert report.json()["discoveries"][0]["matching_games"][0]["opponent"] == "Iowa"
    monkeypatch.setattr(module, "discover", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("cached matchup was recomputed")))
    assert client.post("/api/discover", json={"game_id": "123", "team_id": 2}).status_code == 200
    assert client.post("/api/discover", json={"game_id": "123", "team_id": 99}).status_code == 422
    assert client.post("/api/discover", json={"game_id": "not-real"}).status_code == 404


def test_home_page():
    response = TestClient(module.app).get("/")
    assert response.status_code == 200
    assert "Irrational Fan" in response.text


def test_busy_discovery_returns_429(monkeypatch):
    import asyncio
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setattr(module, "_read_games", lambda now: [{
        "game_id": "busy-test", "home_id": 1, "away_id": 2,
        "home_team": "A", "away_team": "B", "date": "2026-11-07",
    }])
    module._REPORT_CACHE.clear()

    async def run():
        await module._DISCOVERY_LOCK.acquire()
        try:
            async with AsyncClient(transport=ASGITransport(app=module.app), base_url="http://test") as client:
                response = await client.post("/api/discover", json={"game_id": "busy-test"})
                assert response.status_code == 429
        finally:
            module._DISCOVERY_LOCK.release()

    asyncio.run(run())
