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


def test_discovery_modes_change_search_objective():
    entertaining = module._discovery_config("entertaining")
    balanced = module._discovery_config("balanced")
    evidence = module._discovery_config("evidence")
    assert entertaining.ranking.rarity > balanced.ranking.rarity > evidence.ranking.rarity
    assert entertaining.ranking.sample_size < balanced.ranking.sample_size < evidence.ranking.sample_size
    assert entertaining.search.beam_width < balanced.search.beam_width < evidence.search.beam_width
    assert entertaining.search.min_success_rate < balanced.search.min_success_rate < evidence.search.min_success_rate


def test_alternate_styles_have_distinct_search_parameters():
    perfect = module._discovery_config("perfect")
    rivalry = module._discovery_config("rivalry")
    chaos = module._discovery_config("chaos")
    assert perfect.ranking.perfect_record_bonus > module._discovery_config("entertaining").ranking.perfect_record_bonus
    assert rivalry.search.scopes[0] == "h2h"
    assert "all" not in rivalry.search.scopes
    assert chaos.search.max_atomic_candidates > module._discovery_config("entertaining").search.max_atomic_candidates
    assert chaos.ranking.n_conditions > perfect.ranking.n_conditions


def test_alternate_styles_accepted_by_api(monkeypatch):
    monkeypatch.setattr(module, "_read_games", lambda now: [{
        "game_id": "styles", "home_id": 1, "away_id": 2,
        "home_team": "A", "away_team": "B", "date": "2026-11-07",
    }])
    seen = []
    def fake_discover(**kwargs):
        seen.append(kwargs["config"])
        return SimpleNamespace(team="A", opponent="B", report="Original report",
            discoveries=[
                Discovery(scope="team", conditions=(), wins=5, losses=0, n_games=5,
                          win_pct=1, streak=5, ranking_score=20, description="Perfect",
                          matching_games=[]),
                Discovery(scope="h2h", conditions=(), wins=3, losses=1, n_games=4,
                          win_pct=.75, streak=2, ranking_score=15, description="Rivalry",
                          matching_games=[]),
            ])
    monkeypatch.setattr(module, "discover", fake_discover)
    client = TestClient(module.app)
    for mode in ("perfect", "rivalry", "chaos"):
        response = client.post("/api/discover", json={"game_id": "styles", "discovery_mode": mode})
        assert response.status_code == 200
        assert response.json()["settings"]["discovery_mode"] == mode
    perfect = client.post("/api/discover", json={"game_id": "styles", "discovery_mode": "perfect"}).json()
    assert perfect["count"] == 1
    assert perfect["discoveries"][0]["losses"] == 0
    rivalry = client.post("/api/discover", json={"game_id": "styles", "discovery_mode": "rivalry"}).json()
    assert rivalry["discoveries"][0]["scope"] == "h2h"
    assert len(seen) == 3
    assert client.post("/api/discover", json={"game_id": "styles", "discovery_mode": "nonsense"}).status_code == 422


def test_custom_parameters_change_backend_and_cache(monkeypatch):
    monkeypatch.setattr(module, "_read_games", lambda now: [{
        "game_id": "custom", "home_id": 1, "away_id": 2,
        "home_team": "A", "away_team": "B", "date": "2026-11-07",
    }])
    seen = []
    def fake_discover(**kwargs):
        seen.append(kwargs["config"])
        return SimpleNamespace(team="A", opponent="B", report="", discoveries=[])
    monkeypatch.setattr(module, "discover", fake_discover)
    client = TestClient(module.app)
    low = {"game_id": "custom", "discovery_mode": "custom",
           "evidence_strength": 0, "weirdness": 0}
    high = {**low, "evidence_strength": 100, "weirdness": 100,
            "min_sample_size": 10, "max_conditions": 6}
    a = client.post("/api/discover", json=low)
    b = client.post("/api/discover", json=high)
    assert a.status_code == b.status_code == 200
    assert len(seen) == 2
    assert seen[1].search.min_success_rate > seen[0].search.min_success_rate
    assert seen[1].search.beam_width > seen[0].search.beam_width
    assert seen[1].ranking.rarity > seen[0].ranking.rarity
    assert seen[1].ranking.sample_size > seen[0].ranking.sample_size
    assert seen[1].search.max_depth == 6
    assert seen[1].search.min_sample_size == 10
    assert b.json()["effective_config"]["search"]["beam_width"] == 100
    assert client.post("/api/discover", json=high).status_code == 200
    assert len(seen) == 2
    assert client.get("/api/cache").json()["count"] == 2
    for bad in ({**high, "weirdness": 101}, {**high, "evidence_strength": -1}):
        assert client.post("/api/discover", json=bad).status_code == 422
