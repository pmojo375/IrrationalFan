"""Unit tests for the generic discovery engine and CFB adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from discovery.engine.conditions import (
    condition_applies,
    evaluate_mask,
    generate_atomic_conditions,
)
from discovery.engine.models import Condition, FeatureSpec, Operator, RankingWeights, SearchConfig
from discovery.engine.ranking import deduplicate_discoveries, score_discovery
from discovery.engine.search import BeamSearchStrategy, DiscoveryEngine
from discovery.engine.streaks import recent_favorable_streak
from discovery.engine.validation import apply_conditions, recount_record, validate_discovery
from discovery.sports.college_football.features import LEAKAGE_COLUMNS, enrich_team_games, feature_specs
from discovery.sports.college_football.matchup import (
    build_matchup_features,
    clear_data_cache,
    discover,
    resolve_team_name,
)


def _synthetic_frame() -> pd.DataFrame:
    """
    Planted pattern: home + month==11 + opp_wins >= 6 ⇒ always win (8 games).
    Other games are mixed so the engine must find the multi-condition pattern.
    """
    rows = []
    gid = 1
    # Planted winning subgroup
    for i in range(8):
        rows.append(
            {
                "game_id": gid,
                "season": 2018 + i,
                "week": 12,
                "season_type": 2,
                "year": 2018 + i,
                "month": 11,
                "day": 10 + (i % 5),
                "time": "12:00",
                "team_id": 1,
                "team": "Alpha",
                "team_abbreviation": "ALP",
                "opponent_id": 100 + i,
                "opponent": f"Opp{i}",
                "opponent_abbreviation": f"O{i}",
                "is_home": True,
                "neutral_site": False,
                "conference_competition": True,
                "team_score": 30,
                "opponent_score": 17,
                "won": True,
                "wins": 5 + (i % 3),
                "losses": 2,
                "record": "5-2",
                "venue": "Home Stadium",
                "attendance": 50000,
                "status": "STATUS_FINAL",
                "game_date": f"{2018 + i}-11-{10 + (i % 5):02d}T17:00Z",
            }
        )
        gid += 1

    # Distractor games for Alpha (should lose often when pattern incomplete)
    for i in range(12):
        rows.append(
            {
                "game_id": gid,
                "season": 2015 + i,
                "week": 5,
                "season_type": 2,
                "year": 2015 + i,
                "month": 9,
                "day": 5,
                "time": "15:00",
                "team_id": 1,
                "team": "Alpha",
                "team_abbreviation": "ALP",
                "opponent_id": 200 + i,
                "opponent": f"Other{i}",
                "opponent_abbreviation": f"X{i}",
                "is_home": i % 2 == 0,
                "neutral_site": False,
                "conference_competition": False,
                "team_score": 14 if i % 3 else 28,
                "opponent_score": 21 if i % 3 else 10,
                "won": i % 3 == 0,
                "wins": i % 4,
                "losses": i % 3,
                "record": "0-0",
                "venue": "Somewhere",
                "attendance": 10000,
                "status": "STATUS_FINAL",
                "game_date": f"{2015 + i}-09-05T19:00Z",
            }
        )
        gid += 1

    # Other teams for "all" scope noise
    for i in range(20):
        rows.append(
            {
                "game_id": gid,
                "season": 2020,
                "week": 8,
                "season_type": 2,
                "year": 2020,
                "month": 10,
                "day": 1 + (i % 20),
                "time": "12:00",
                "team_id": 50 + i,
                "team": f"Team{i}",
                "team_abbreviation": f"T{i}",
                "opponent_id": 300 + i,
                "opponent": f"Foe{i}",
                "opponent_abbreviation": f"F{i}",
                "is_home": True,
                "neutral_site": False,
                "conference_competition": True,
                "team_score": 20,
                "opponent_score": 27,
                "won": False,
                "wins": 3,
                "losses": 3,
                "record": "3-3",
                "venue": "Field",
                "attendance": 1000,
                "status": "STATUS_FINAL",
                "game_date": f"2020-10-{1 + (i % 20):02d}T16:00Z",
            }
        )
        gid += 1

    df = pd.DataFrame(rows)
    # Manually attach opponent wins for planted rows (enrich will join siblings;
    # planted games have no sibling rows, so set directly after enrich).
    enriched = enrich_team_games(df)
    # For planted Alpha November home games, force opp_wins high
    mask = (enriched["team"] == "Alpha") & (enriched["month"] == 11) & (enriched["is_home"])
    enriched.loc[mask, "opp_wins"] = 6
    enriched.loc[mask, "opp_losses"] = 2
    enriched.loc[mask, "opp_games_played"] = 8
    enriched.loc[mask, "opp_win_pct"] = 0.75
    enriched.loc[mask, "opp_win_streak"] = 2
    enriched.loc[mask, "opp_loss_streak"] = 0
    enriched.loc[mask, "opp_has_winning_record"] = True
    enriched.loc[mask, "opp_has_losing_record"] = False
    # Distractors: lower opp wins
    other = (enriched["team"] == "Alpha") & (enriched["month"] != 11)
    enriched.loc[other, "opp_wins"] = 2
    enriched.loc[other, "opp_losses"] = 4
    enriched.loc[other, "opp_games_played"] = 6
    enriched.loc[other, "opp_win_pct"] = 2 / 6
    enriched.loc[other, "opp_win_streak"] = 0
    enriched.loc[other, "opp_loss_streak"] = 1
    enriched.loc[other, "opp_has_winning_record"] = False
    enriched.loc[other, "opp_has_losing_record"] = True
    enriched["win_diff"] = enriched["wins"] - enriched["opp_wins"].fillna(0)
    return enriched


def test_missing_values_never_match_as_zero():
    s = pd.Series([np.nan, 0, 5])
    cond = Condition("wins", Operator.GE, 0, "wins >= 0")
    mask = evaluate_mask(s, cond)
    assert mask.tolist() == [False, True, True]


def test_condition_applies_rejects_missing_current():
    cond = Condition("wins", Operator.GE, 3)
    assert condition_applies({"wins": 5}, cond)
    assert not condition_applies({"wins": np.nan}, cond)
    assert not condition_applies({}, cond)


def test_no_leakage_features_in_specs():
    names = {f.name for f in feature_specs()}
    assert names.isdisjoint(LEAKAGE_COLUMNS)


def test_streak_ignores_nonqualifying_games():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22", "2020-01-29"]
            ),
            "favorable": [True, False, True, True, True],
            "qual": [True, False, True, True, True],
        }
    )
    mask = df["qual"].to_numpy()
    # Qualifying outcomes chronologically: T, T, T, T — but wait, second row not qualifying.
    # Qualifying: Jan1 T, Jan15 T, Jan22 T, Jan29 T → streak 4
    # The False on Jan8 is nonqualifying and must not interrupt.
    assert recent_favorable_streak(df, mask) == 4


def test_streak_stops_at_unfavorable_qualifier():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "favorable": [True, False, True],
        }
    )
    mask = np.array([True, True, True])
    assert recent_favorable_streak(df, mask) == 1


def test_recount_and_validation_match():
    df = _synthetic_frame()
    df = df[df["team"] == "Alpha"].copy()
    df["favorable"] = df["won"].astype(bool)
    conditions = [
        Condition("is_home", Operator.EQ, True, "Home"),
        Condition("month", Operator.EQ, 11, "November"),
        Condition("opp_wins", Operator.GE, 6, "opp wins >= 6"),
    ]
    mask = apply_conditions(df, conditions)
    wins, losses, win_pct = recount_record(df, mask)
    assert wins == 8
    assert losses == 0
    assert win_pct == 1.0

    disc = validate_discovery(df, conditions, scope="team", description="test")
    assert disc is not None
    assert disc.wins == 8
    assert disc.losses == 0
    assert len(disc.matching_games) == 8


def test_beam_search_finds_planted_multi_condition_pattern():
    df = _synthetic_frame()
    team_df = df[df["team"] == "Alpha"].copy()
    team_df["favorable"] = team_df["won"].astype(bool)

    current = {
        "is_home": True,
        "month": 11,
        "month_name": "November",
        "opp_wins": 6,
        "wins": 5,
        "losses": 2,
        "games_played": 7,
        "win_pct": 5 / 7,
        "win_streak": 1,
        "loss_streak": 0,
        "has_winning_record": True,
        "has_losing_record": False,
        "neutral_site": False,
        "conference_competition": True,
        "week": 12,
        "season_type": 2,
        "opp_losses": 2,
        "opp_games_played": 8,
        "opp_win_pct": 0.75,
        "opp_win_streak": 2,
        "opp_loss_streak": 0,
        "opp_has_winning_record": True,
        "opp_has_losing_record": False,
        "win_diff": -1,
        "day_of_week": "Saturday",
    }
    features = feature_specs()
    engine = DiscoveryEngine(
        strategy=BeamSearchStrategy(),
        config=SearchConfig(
            beam_width=30,
            max_depth=3,
            min_sample_size=5,
            max_conditions=3,
            min_success_rate=0.8,
            min_streak=3,
            max_atomic_candidates=60,
        ),
        weights=RankingWeights(),
    )
    discoveries, stats = engine.run_scope(
        team_df,
        features,
        current,
        scope="team",
        target_col="favorable",
    )
    assert stats.candidates_evaluated > 0
    assert discoveries, "Expected at least one discovery"

    # Look for a discovery covering the planted 8-0 subgroup
    found = False
    for d in discoveries:
        feats = {c.feature for c in d.conditions}
        if d.wins >= 8 and d.losses == 0 and "is_home" in feats:
            found = True
            break
        if d.wins == 8 and d.losses == 0:
            found = True
            break
    assert found, f"Planted pattern not found. Top: {[(d.wins, d.losses, d.conditions) for d in discoveries[:5]]}"


def test_dedup_removes_near_duplicates():
    from discovery.engine.models import Discovery, MatchingGame

    g = MatchingGame(2020, "2020-01-01", "A", "B", 10, 7, True)
    a = Discovery(
        scope="team",
        conditions=(Condition("is_home", Operator.EQ, True), Condition("wins", Operator.GE, 3)),
        wins=5,
        losses=0,
        n_games=5,
        win_pct=1.0,
        streak=5,
        ranking_score=90,
        description="",
        matching_games=[g] * 5,
    )
    b = Discovery(
        scope="team",
        conditions=(Condition("is_home", Operator.EQ, True), Condition("wins", Operator.GE, 2)),
        wins=5,
        losses=0,
        n_games=5,
        win_pct=1.0,
        streak=5,
        ranking_score=80,
        description="",
        matching_games=[g] * 5,
    )
    kept = deduplicate_discoveries([a, b])
    assert len(kept) == 1
    assert kept[0].ranking_score == 90


def test_entering_streaks_exclude_current_game():
    df = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "season": [2026] * 4,
            "week": [1, 2, 3, 4],
            "season_type": [2] * 4,
            "year": [2026] * 4,
            "month": [9] * 4,
            "day": [1, 8, 15, 22],
            "time": ["12:00"] * 4,
            "team_id": [1, 1, 1, 1],
            "team": ["A"] * 4,
            "team_abbreviation": ["A"] * 4,
            "opponent_id": [2, 3, 4, 5],
            "opponent": ["B", "C", "D", "E"],
            "opponent_abbreviation": ["B", "C", "D", "E"],
            "is_home": [True] * 4,
            "neutral_site": [False] * 4,
            "conference_competition": [False] * 4,
            "team_score": [10, 20, 30, 5],
            "opponent_score": [7, 14, 27, 10],
            "won": [True, True, False, True],
            "wins": [0, 1, 2, 2],
            "losses": [0, 0, 0, 1],
            "record": ["0-0", "1-0", "2-0", "2-1"],
            "venue": ["V"] * 4,
            "attendance": [0] * 4,
            "status": ["STATUS_FINAL"] * 4,
            "game_date": [
                "2026-09-01T16:00Z",
                "2026-09-08T16:00Z",
                "2026-09-15T16:00Z",
                "2026-09-22T16:00Z",
            ],
        }
    )
    # Need sibling rows for merge — duplicate as opponent perspective lightly
    opp_rows = df.copy()
    opp_rows["team_id"] = df["opponent_id"]
    opp_rows["team"] = df["opponent"]
    opp_rows["opponent_id"] = 1
    opp_rows["opponent"] = "A"
    opp_rows["won"] = ~df["won"]
    opp_rows["wins"] = 0
    opp_rows["losses"] = 0
    full = pd.concat([df, opp_rows], ignore_index=True)
    enriched = enrich_team_games(full)
    a = enriched[enriched["team"] == "A"].sort_values("game_date")
    assert a["win_streak"].tolist() == [0, 1, 2, 0]
    assert a["loss_streak"].tolist() == [0, 0, 0, 1]


def test_atomic_conditions_only_match_current():
    df = _synthetic_frame()
    current = {"is_home": True, "month": 11, "wins": 5, "opp_wins": 6}
    feats = [
        FeatureSpec("is_home", "boolean", "Home"),
        FeatureSpec("month", "categorical", "Month"),
        FeatureSpec("wins", "numeric", "Wins"),
        FeatureSpec("opp_wins", "numeric", "Opp wins"),
    ]
    conds = generate_atomic_conditions(df, feats, current)
    assert conds
    for c in conds:
        assert condition_applies(current, c)


def test_score_penalizes_tiny_samples():
    tiny = score_discovery(wins=1, losses=0, streak=1, n_conditions=1, population_size=100, weights=RankingWeights())
    solid = score_discovery(wins=8, losses=0, streak=8, n_conditions=3, population_size=100, weights=RankingWeights())
    assert solid > tiny


def test_mixed_timestamps_parse_and_as_of_cutoff():
    from discovery.sports.college_football.dates import before_as_of_mask, parse_as_of, parse_game_dates

    dates = pd.Series(
        [
            "2020-12-05T20:30Z",
            "2026-10-24T16:00:00.000Z",
            "2026-11-28T17:00:00.000Z",
        ]
    )
    parsed = parse_game_dates(dates)
    assert parsed.isna().sum() == 0
    as_of = parse_as_of("2026-11-28")
    mask = before_as_of_mask(dates, as_of)
    assert mask.tolist() == [True, True, False]


def test_discoveries_must_match_current_form():
    from discovery.engine.conditions import condition_applies, conditions_apply_to_current
    from discovery.engine.models import Condition, Operator

    current = {"wins": 4, "losses": 3, "opp_win_pct": 0.571, "is_home": True}
    good = Condition("wins", Operator.GE, 4)
    bad = Condition("wins", Operator.GE, 8)
    assert condition_applies(current, good)
    assert not condition_applies(current, bad)
    assert conditions_apply_to_current(current, [good, Condition("is_home", Operator.EQ, True)])
    assert not conditions_apply_to_current(current, [good, bad])


@pytest.fixture
def real_data_available():
    from discovery.config import default_data_path

    path = default_data_path()
    if not path.exists():
        pytest.skip("Processed parquet not available")
    clear_data_cache()
    return path


def test_resolve_team_and_real_discover(real_data_available):
    # Keep scopes narrow so the multi-season dataset stays test-friendly.
    result = discover(
        team="Michigan State",
        opponent="Michigan",
        game_date="2026-11-21",
        home=True,
        top_n=10,
        min_sample_size=3,
        max_conditions=3,
        beam_width=20,
        scopes=("team", "opponent", "h2h"),
        print_report=False,
        data_path=real_data_available,
    )
    assert "Michigan State" in result.team
    assert "Michigan" in result.opponent
    assert "wins" in result.team_features
    assert result.stats.candidates_evaluated >= 0
    assert isinstance(result.report, str)
    assert "IRRATIONAL FAN" in result.report
    for disc in result.discoveries:
        assert disc.n_games == disc.wins + disc.losses
        assert len(disc.matching_games) == disc.n_games


def test_default_features_prioritize_fan_friendly_stats():
    from discovery.sports.college_football.features import advanced_feature_specs
    defaults = {f.name for f in feature_specs()}
    advanced = {f.name for f in advanced_feature_specs()}
    assert "wins" in defaults and "opp_wins" in defaults
    assert "EPAplay_off" not in defaults
    assert "EPAplay_off" in advanced


def test_diversity_removes_threshold_variations_even_with_different_records():
    from discovery.engine.models import Discovery
    from discovery.engine.search import _diverse_top

    def make(conditions, score, scope="team"):
        return Discovery(
            scope=scope, conditions=conditions, wins=7, losses=1,
            n_games=8, win_pct=0.875, streak=3, ranking_score=score,
            description="", matching_games=[]
        )

    a = make((Condition("month", Operator.EQ, 11), Condition("opp_wins", Operator.GE, 5)), 90)
    b = make((Condition("month", Operator.EQ, 11), Condition("opp_wins", Operator.GE, 6)), 85)
    c = make((Condition("is_home", Operator.EQ, True),), 80)
    result = _diverse_top([a, b, c], top_n=20)
    assert result == [a, c]


def test_more_conditions_no_longer_receive_automatic_bonus():
    weights = RankingWeights()
    simple = score_discovery(wins=8, losses=0, streak=8, n_conditions=1,
                             population_size=100, weights=weights)
    complex_ = score_discovery(wins=8, losses=0, streak=8, n_conditions=4,
                               population_size=100, weights=weights)
    assert simple > complex_
