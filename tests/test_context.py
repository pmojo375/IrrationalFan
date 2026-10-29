"""Pregame context must be identical in processing and upcoming discovery."""
from datetime import datetime, timezone
from io import BytesIO
import json
import urllib.error

import pandas as pd
import pytest

import process
from discovery.engine.conditions import generate_atomic_conditions
from discovery.engine.models import Condition, Operator
from discovery.sports.college_football.context import (
    CONTEXT_FEATURES, add_pregame_context, upcoming_context, schedule_metadata,
)
from discovery.sports.college_football.features import feature_specs
from discovery.sports.college_football.formatting import condition_phrase


def history():
    games = []
    for gid, date, score, allowed, conference in [
        (1, "2026-09-05T16:00Z", 14, 35, False),
        (2, "2026-09-12T23:30Z", 31, 20, True),
        (3, "2026-09-26T16:00Z", 17, 10, True),
        (4, "2027-09-04T16:00Z", 7, 21, True),
    ]:
        for tid, oid, a, b in [(1, 2, score, allowed), (2, 1, allowed, score)]:
            games.append(dict(game_id=gid, season=2026 if gid < 4 else 2027,
                              team_id=tid, opponent_id=oid, game_date=date,
                              team_score=a, opponent_score=b, won=a > b,
                              conference_competition=conference, opp_conference="Big Ten"))
    return pd.DataFrame(games)


def test_shift_rest_conference_and_season_reset():
    out = add_pregame_context(history()).query("team_id == 1").set_index("game_id")
    assert pd.isna(out.loc[1, "previous_margin"])
    assert out.loc[2, "previous_margin"] == -21
    assert not out.loc[2, "previous_won"]
    assert out.loc[2, "rest_days"] == 6
    assert out.loc[3, "rest_days"] == 13
    assert out.loc[3, "after_bye"]
    assert out.loc[3, "conference_wins"] == 1
    assert out.loc[3, "conference_losses"] == 0
    assert out.loc[3, "opp_conference_losses"] == 1
    assert out.loc[3, "opp_previous_margin"] == -11
    assert pd.isna(out.loc[4, "previous_points_for"])
    assert pd.isna(out.loc[4, "rest_days"])
    assert pd.isna(out.loc[4, "after_bye"])
    assert out.loc[4, "conference_wins"] == 0


def test_unknown_conference_games_do_not_become_nonconference():
    raw = history()
    raw["conference_competition"] = raw["conference_competition"].astype("boolean")
    raw.loc[raw.game_id.eq(1), "conference_competition"] = pd.NA
    out = add_pregame_context(raw)
    assert out.loc[out.game_id.eq(2), "conference_wins"].isna().all()


def test_upcoming_context_matches_historical_calculation_and_ignores_future():
    raw = history()
    meta = {"conference_competition": True, "opp_conference": "Big Ten"}
    # Query game 3 using its real kickoff; game 3 and later outcomes are excluded.
    a, b = upcoming_context(raw, team_id=1, opponent_id=2, game_date="2026-09-26T16:00Z",
                            season=2026, metadata=meta, opponent_metadata=meta)
    expected = add_pregame_context(raw).query("game_id == 3 and team_id == 1").iloc[0]
    for name, *_ in CONTEXT_FEATURES:
        if name not in {"rank", "conference"}:
            assert a[name] == expected[name]
    assert b["previous_points_for"] == 20
    assert a["opp_previous_points_for"] == b["previous_points_for"]
    assert a["conference_wins"] == 1
    assert a["rest_days"] == 13


def test_schedule_metadata_uses_perspective_and_unknown_rank():
    row = {"home_rank": 7, "away_rank": None, "home_conference": "Big Ten Conference",
           "away_conference": "Mountain West", "conference_competition": None,
           "conference_game": False, "season": 2026, "week": 9}
    home, away = schedule_metadata(row, home=True), schedule_metadata(row, home=False)
    assert home["rank"] == 7 and away["opp_rank"] == 7
    assert pd.isna(away["rank"])
    assert home["conference"] == "Big Ten"
    assert away["opp_conference"] == "Big Ten"
    assert home["conference_competition"] is False


def test_feature_budget_does_not_drop_appended_context():
    current = {f.name: True if f.kind == "boolean" else "Big Ten" if f.kind == "categorical" else 7
               for f in feature_specs()}
    frame = pd.DataFrame([current] * 4)
    conditions = generate_atomic_conditions(frame, feature_specs(), current, max_candidates=120)
    names = {c.feature for c in conditions}
    for name in ["previous_margin", "rest_days", "rank", "opp_rank", "conference", "opp_conference"]:
        assert name in names


def test_new_feature_phrasing():
    assert condition_phrase(Condition("previous_won", Operator.EQ, False)) == "Not: won previous game"
    assert condition_phrase(Condition("opp_rank", Operator.LE, 10)) == "Opponent: Pregame poll rank <= 10"


def test_download_caches_raw_sources_and_refreshes_current_season(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 8 else now.year - 1
    for name in ["SCHEDULES_DIR", "SUMMARIES_DIR", "TEAMS_DIR"]:
        monkeypatch.setattr(process, name, tmp_path / name)
    calls = []
    def download(url, timeout):
        calls.append(url)
        season = int(url.rsplit('_', 1)[-1].split('.')[0])
        if 'cfb_schedules/' in url:
            data = {"season": [season], "game_id": [1]}
        elif 'cfb_team_summaries_weekly/' in url:
            data = {"season": [season], "team_id": [1], "through_week": [1]}
        else:
            data = {"season": [season], "team_id": [1], "conference_name": ["Big Ten"]}
        buffer = BytesIO()
        pd.DataFrame(data).to_parquet(buffer, engine="fastparquet")
        return BytesIO(buffer.getvalue())
    monkeypatch.setattr(process.urllib.request, "urlopen", download)
    process.download_season_files(workers=1)
    assert (process.TEAMS_DIR / f"cfb_teams_{year}.parquet").exists()
    manifest = json.loads((process.TEAMS_DIR / f"cfb_teams_{year}.source.json").read_text())
    assert manifest['rows'] == 1 and 'espn_cfb_teams' in manifest['url']
    calls.clear()
    process.download_season_files(workers=1)
    assert len(calls) == 3
    assert all(f"_{year}.parquet" in url for url in calls)
    # A failed refresh must leave the last valid source untouched.
    original = (process.TEAMS_DIR / f"cfb_teams_{year}.parquet").read_bytes()
    def broken(url, timeout):
        return BytesIO(b"not parquet")
    monkeypatch.setattr(process.urllib.request, "urlopen", broken)
    with pytest.raises(Exception):
        process.download_season_files(workers=1)
    assert (process.TEAMS_DIR / f"cfb_teams_{year}.parquet").read_bytes() == original


def test_conference_metadata_join_is_season_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "TEAMS_DIR", tmp_path)
    pd.DataFrame({"season": [2025, 2026], "team_id": [1, 1],
                  "conference_name": ["Old Conference", "New Conference"]}).to_parquet(tmp_path / "cfb_teams_2026.parquet")
    games = pd.DataFrame({"season": [2025, 2026], "home_id": [1, 1], "away_id": [2, 2],
                          "home_conference": [None, None], "away_conference": [None, None]})
    out = process.add_team_metadata(games)
    assert out.home_conference.tolist() == ["Old Conference", "New Conference"]
    assert len(out) == len(games)


def test_normalize_preserves_ranks_conference_alias_and_excludes_cancelled():
    raw = pd.DataFrame([{
        "game_id": 1, "season": 2026, "week": 2, "season_type": "regular",
        "start_date": "2026-09-12T16:00Z", "neutral_site": False,
        "conference_competition": None, "conference_game": True,
        "venue": "Field", "attendance": 10000, "status": "STATUS_FINAL",
        "home_id": 1, "home_team": "A", "home_abbreviation": "A",
        "home_points": 21, "home_winner": True, "home_rank": 7,
        "home_conference": "Big Ten", "away_id": 2, "away_team": "B",
        "away_abbreviation": "B", "away_points": 14, "away_winner": False,
        "away_rank": None, "away_conference": "Big Ten",
    }])
    cancelled = raw.copy().assign(game_id=2, status="STATUS_CANCELED", home_points=0, away_points=0)
    games = process.normalize_schedule(pd.concat([raw, cancelled], ignore_index=True))
    assert games.game_id.tolist() == [1]
    rows = process.expand_to_team_rows(games)
    assert rows['rank'].iloc[0] == 7
    assert pd.isna(rows['rank'].iloc[1])
    assert rows['opp_rank'].iloc[1] == 7
    assert rows['conference_competition'].all()
    assert rows['conference'].tolist() == ['Big Ten', 'Big Ten']


def test_upcoming_features_replace_stale_rank_and_recompute_previous_game(monkeypatch):
    from discovery.sports.college_football.features import enrich_team_games
    from discovery.sports.college_football.matchup import build_matchup_features
    raw = history()
    raw['team'] = raw.team_id.map({1: 'A', 2: 'B'})
    raw['opponent'] = raw.opponent_id.map({1: 'A', 2: 'B'})
    raw['is_home'] = raw.team_id.eq(1)
    raw['rank'] = 22
    raw = process.add_prior_records(process.add_datetime_parts(raw))
    df = enrich_team_games(raw)
    team = {'rank': 7, 'opp_rank': float('nan'), 'conference': 'Big Ten', 'opp_conference': 'SEC',
            'conference_competition': False, 'neutral_site': False, 'week': 5, 'season': 2026}
    opponent = {**team, 'rank': float('nan'), 'opp_rank': 7, 'conference': 'SEC', 'opp_conference': 'Big Ten'}
    monkeypatch.setattr('discovery.sports.college_football.matchup.scheduled_context',
                        lambda *args: (team, opponent, '2026-10-03T16:00Z'))
    a, b = build_matchup_features(df, team='A', opponent='B', team_id=1, opponent_id=2,
                                game_date='2026-10-03', home=True)
    assert a['rank'] == 7 and pd.isna(a['opp_rank'])
    assert pd.isna(b['rank']) and b['opp_rank'] == 7
    assert a['previous_points_for'] == 17
    assert a['previous_margin'] == 7
    assert a['rest_days'] == 6
    assert a['conference_wins'] == 2
    assert b['conference_losses'] == 2
    assert a['week'] == 5 and a['conference_competition'] is False
    assert a['conference'] == 'Big Ten' and a['opp_conference'] == 'SEC'


def test_intervening_unplayed_game_is_not_a_bye_or_a_known_result():
    raw = history()
    meta = {'conference_competition': True, 'opp_conference': 'Big Ten',
            '_prior_games': [('2026-10-03T16:00Z', True)]}
    a, b = upcoming_context(raw, team_id=1, opponent_id=2, game_date='2026-10-10T16:00Z',
                            season=2026, metadata=meta, opponent_metadata=meta)
    assert a['rest_days'] == 6 and not a['after_bye']
    assert pd.isna(a['previous_points_for'])
    assert pd.isna(a['previous_won'])
    assert pd.isna(a['conference_wins'])
    assert pd.isna(b['opp_previous_points_for'])
    assert pd.isna(b['opp_conference_losses'])
