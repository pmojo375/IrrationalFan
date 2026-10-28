"""Regression tests for report quality after candidate search."""

import pytest
import pandas as pd

from discovery.engine.models import Condition, Discovery, Operator
from discovery.engine.search import _evidence_diverse_top


def make_discovery(feature, score=90, objective="record", conditions=None):
    return Discovery(
        scope="team", conditions=conditions or (Condition(feature, Operator.EQ, True),),
        wins=7, losses=1, n_games=8, win_pct=0.875, streak=4,
        ranking_score=score, description="", matching_games=[], objective=objective,
    )


@pytest.mark.parametrize("record_count", [0, 1, 2, 3, 6])
def test_streak_budget_uses_returned_count(record_count):
    candidates = [make_discovery(f"streak_{i}", 100 - i, "streak") for i in range(8)]
    candidates += [make_discovery(f"record_{i}", 70 - i) for i in range(record_count)]
    result = _evidence_diverse_top(candidates, [], top_n=20)
    streaks = sum(d.objective == "streak" for d in result)
    # Permit one standout streak when the record pool is sparse.
    assert streaks <= max(1, len(result) * 2 // 5)
    assert sum(d.objective == "record" for d in result) == record_count
    assert len(result) == len({id(d) for d in result})


def test_narrative_subsets_ignore_threshold_changes_and_aliases():
    simple = make_discovery("unused", conditions=(
        Condition("month", Operator.EQ, 11), Condition("wins", Operator.GE, 4),
    ))
    variation = make_discovery("unused", 85, conditions=(
        Condition("month_name", Operator.EQ, "November"),
        Condition("wins", Operator.GE, 3), Condition("is_home", Operator.EQ, True),
    ))
    distinct = make_discovery("opp_losses", 80)
    result = _evidence_diverse_top([simple, variation, distinct], [], top_n=20)
    assert [d.conditions for d in result] == [simple.conditions, distinct.conditions]


def test_reranking_does_not_mutate_candidate_scores():
    candidates = [make_discovery(f"feature_{i}", 90 - i) for i in range(5)]
    original = [d.ranking_score for d in candidates]
    first = _evidence_diverse_top(candidates, [], top_n=20)
    second = _evidence_diverse_top(candidates, [], top_n=20)
    assert [d.ranking_score for d in candidates] == original
    assert first == second
    assert [d.ranking_score for d in first] == sorted(
        [d.ranking_score for d in first], reverse=True,
    )


def test_quality_floor_does_not_fill_with_weak_records():
    candidates = [make_discovery("streak", 90, "streak"), make_discovery("weak", 10)]
    result = _evidence_diverse_top(candidates, [], top_n=20)
    assert len(result) == 1
    assert result[0].objective == "streak"


def test_reduced_streak_budget_reconsiders_blocked_records():
    best = make_discovery("best", 100, "streak")
    streaks = [make_discovery(f"shared_{i}", 95 - i, "streak") for i in range(5)]
    records = [make_discovery(f"shared_{i}", 70 - i) for i in range(5)]
    result = _evidence_diverse_top([best, *streaks, *records], [], top_n=20)
    assert sum(d.objective == "record" for d in result) > 0
    assert sum(d.objective == "streak" for d in result) <= max(1, len(result) * 2 // 5)


def test_evidence_overlap_remains_scope_specific():
    a = make_discovery("home", 90)
    b = make_discovery("month", 85)
    c = make_discovery("opponent_form", 80)
    c.scope = "opponent"
    frame = pd.DataFrame({"game_id": [1, 2, 3], "home": True, "month": True, "opponent_form": True})
    scopes = [(scope, frame, {}, "wins", "favorable") for scope in ("team", "opponent")]
    result = _evidence_diverse_top([a, b, c], scopes, top_n=20)
    assert [(d.scope, d.conditions) for d in result] == [
        (a.scope, a.conditions), (c.scope, c.conditions),
    ]


@pytest.mark.parametrize("top_n", [0, 1, 2, 5])
def test_requested_limit_is_respected(top_n):
    candidates = [make_discovery(f"feature_{i}") for i in range(10)]
    assert len(_evidence_diverse_top(candidates, [], top_n=top_n)) == top_n
