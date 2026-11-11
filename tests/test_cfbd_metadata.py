import json

import pandas as pd

from discovery.sports.college_football.cfbd_metadata import enrich_metadata, load_metadata


def test_d1_only_metadata_and_unknowns(tmp_path):
    path = tmp_path / "teams.json"
    path.write_text(json.dumps([
        {"id": 1, "school": "A", "classification": "fbs", "mascot": "Bears",
         "color": "#112233", "location": {"capacity": 10000, "latitude": 42, "longitude": -86}},
        {"id": 2, "school": "B", "classification": "fcs", "mascot": "Bears",
         "color": "#112233", "location": {"capacity": 20000, "latitude": 41, "longitude": -85}},
        {"id": 3, "school": "C", "classification": "ii", "mascot": "Bears"},
    ]))
    meta = load_metadata(path)
    assert set(meta.team_id) == {1, 2}
    games = pd.DataFrame({"team_id": [1, 2, 3], "opponent_id": [2, 1, 1]})
    out = enrich_metadata(games, path)
    assert len(out) == 3
    assert bool(out.loc[0, "same_mascot"])
    assert bool(out.loc[0, "opponent_stadium_larger"])
    assert out.loc[0, "campus_distance_miles"] > 0
    assert pd.isna(out.loc[2, "mascot"])
    assert pd.isna(out.loc[2, "same_mascot"])


def test_missing_snapshot_is_optional(tmp_path):
    games = pd.DataFrame({"team_id": [1], "opponent_id": [2]})
    assert enrich_metadata(games, tmp_path / "missing.json").equals(games)


def test_upcoming_pair_uses_selected_opponent_not_previous_opponent(monkeypatch, tmp_path):
    from discovery.sports.college_football import matchup
    path = tmp_path / "teams.json"
    path.write_text(json.dumps([
        {"id": 1, "classification": "fbs", "mascot": "Spartans", "color": "#123456",
         "location": {"capacity": 70000, "latitude": 42, "longitude": -85}},
        {"id": 2, "classification": "fbs", "mascot": "Gophers", "color": "#654321",
         "location": {"capacity": 50000, "latitude": 44, "longitude": -93}},
        {"id": 3, "classification": "fbs", "mascot": "Spartans", "color": "#123456",
         "location": {"capacity": 90000, "latitude": 41, "longitude": -84}},
    ]))
    monkeypatch.setattr(matchup, "load_metadata", lambda: load_metadata(path))
    team = {"same_mascot": True, "same_primary_color": True, "opponent_stadium_larger": True}
    opponent = {}
    matchup._apply_current_pair_metadata(team, opponent, 1, 2)
    assert team["same_mascot"] == False
    assert team["same_primary_color"] == False
    assert team["opponent_stadium_larger"] == False
    assert opponent["opponent_stadium_larger"] == True
    assert team["campus_distance_miles"] == opponent["campus_distance_miles"]
