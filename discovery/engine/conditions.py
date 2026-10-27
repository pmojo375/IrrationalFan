"""Automatic candidate condition generation from features and a current matchup."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from discovery.engine.models import Condition, FeatureSpec, Operator, conditions_redundant, stricter_condition


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def condition_applies(current: Mapping[str, Any], condition: Condition) -> bool:
    """Return True if the upcoming matchup satisfies the condition."""
    if condition.feature not in current:
        return False
    value = current[condition.feature]
    if _is_missing(value):
        return False
    return _compare(value, condition.operator, condition.value)


def _compare(left: Any, op: Operator, right: Any) -> bool:
    if _is_missing(left) or _is_missing(right):
        return False
    if op == Operator.EQ:
        return left == right
    if op == Operator.NE:
        return left != right
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return False
    if op == Operator.GE:
        return left_f >= right_f
    if op == Operator.LE:
        return left_f <= right_f
    if op == Operator.GT:
        return left_f > right_f
    if op == Operator.LT:
        return left_f < right_f
    return False


def evaluate_mask(series: pd.Series, condition: Condition) -> np.ndarray:
    """
    Boolean mask for rows satisfying a condition.

    Missing feature values never match (not treated as zero).
    """
    missing = pd.isna(series).to_numpy()
    op = condition.operator
    rhs = condition.value

    if op == Operator.EQ:
        if isinstance(rhs, (bool, np.bool_)):
            # Compare only non-missing values; missing never matches.
            mask = series.astype("boolean").eq(bool(rhs)).fillna(False).to_numpy()
        else:
            mask = series.eq(rhs).fillna(False).to_numpy()
    elif op == Operator.NE:
        mask = series.ne(rhs).fillna(False).to_numpy()
        # Rows with missing feature values are not "not equal" — they are unknown.
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        missing = numeric.isna().to_numpy()
        arr = numeric.to_numpy(dtype=float)
        try:
            thr = float(rhs)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=bool)
        if op == Operator.GE:
            mask = arr >= thr
        elif op == Operator.LE:
            mask = arr <= thr
        elif op == Operator.GT:
            mask = arr > thr
        elif op == Operator.LT:
            mask = arr < thr
        else:
            mask = np.zeros(len(series), dtype=bool)

    mask = np.array(mask, dtype=bool, copy=True)
    mask[missing] = False
    return mask


def _default_label(feature: FeatureSpec, op: Operator, value: Any) -> str:
    name = feature.label or feature.name
    if feature.kind == "boolean":
        truthy = bool(value)
        if op == Operator.EQ:
            return name if truthy else f"Not {name.lower()}"
    shown = value
    if feature.name in {"win_pct", "opp_win_pct"}:
        try:
            shown = f"{float(value):.0%}"
        except (TypeError, ValueError):
            shown = value
    return f"{name} {op.value} {shown}"


def generate_atomic_conditions(
    df: pd.DataFrame,
    features: Sequence[FeatureSpec],
    current: Mapping[str, Any],
    max_candidates: int = 80,
) -> list[Condition]:
    """
    Generate single-feature conditions that the current matchup satisfies.

    Numeric thresholds are drawn from values present in the data (and the
    current matchup), not from a hand-written grid of every comparison.
    """
    candidates: list[Condition] = []
    seen: set[tuple] = set()

    def add(cond: Condition) -> None:
        if cond.key() in seen:
            return
        if not condition_applies(current, cond):
            return
        seen.add(cond.key())
        candidates.append(cond)

    for feature in features:
        if feature.name not in df.columns:
            continue
        cur = current.get(feature.name)
        if _is_missing(cur):
            continue

        series = df[feature.name]
        if feature.kind == "boolean":
            add(
                Condition(
                    feature=feature.name,
                    operator=Operator.EQ,
                    value=bool(cur),
                    label=_default_label(feature, Operator.EQ, bool(cur)),
                )
            )
            continue

        if feature.kind == "categorical":
            add(
                Condition(
                    feature=feature.name,
                    operator=Operator.EQ,
                    value=cur,
                    label=_default_label(feature, Operator.EQ, cur),
                )
            )
            continue

        # Numeric
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            continue
        try:
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue

        # Equality to current value
        add(
            Condition(
                feature=feature.name,
                operator=Operator.EQ,
                value=_maybe_int(cur_f),
                label=_default_label(feature, Operator.EQ, _maybe_int(cur_f)),
            )
        )

        # Candidate thresholds from observed unique values + a few rounded anchors
        uniques = sorted({float(x) for x in numeric.unique().tolist()})
        anchors = _numeric_anchors(cur_f, uniques)

        for thr in anchors:
            if thr <= cur_f:
                add(
                    Condition(
                        feature=feature.name,
                        operator=Operator.GE,
                        value=_maybe_int(thr),
                        label=_default_label(feature, Operator.GE, _maybe_int(thr)),
                    )
                )
                if thr < cur_f:
                    add(
                        Condition(
                            feature=feature.name,
                            operator=Operator.GT,
                            value=_maybe_int(thr),
                            label=_default_label(feature, Operator.GT, _maybe_int(thr)),
                        )
                    )
            if thr >= cur_f:
                add(
                    Condition(
                        feature=feature.name,
                        operator=Operator.LE,
                        value=_maybe_int(thr),
                        label=_default_label(feature, Operator.LE, _maybe_int(thr)),
                    )
                )
                if thr > cur_f:
                    add(
                        Condition(
                            feature=feature.name,
                            operator=Operator.LT,
                            value=_maybe_int(thr),
                            label=_default_label(feature, Operator.LT, _maybe_int(thr)),
                        )
                    )

    # Drop clearly redundant weaker thresholds among atomics on the same feature,
    # keeping a compact diverse set (equality + a few inequality strengths).
    pruned = _prune_atomic_redundancy(candidates)
    if len(pruned) > max_candidates:
        pruned = pruned[:max_candidates]
    return pruned


def _maybe_int(value: float) -> Any:
    if float(value).is_integer():
        return int(value)
    return float(value)


def _numeric_anchors(cur: float, uniques: Sequence[float]) -> list[float]:
    """Select a compact set of thresholds relevant to the current value."""
    anchors: set[float] = {cur}
    # Nearby observed values at or below / above current
    below = [u for u in uniques if u <= cur]
    above = [u for u in uniques if u >= cur]
    # Keep extremes and a few evenly spaced points near current
    for pool in (below, above):
        if not pool:
            continue
        anchors.add(pool[0])
        anchors.add(pool[-1])
        if len(pool) <= 8:
            anchors.update(pool)
        else:
            idxs = np.linspace(0, len(pool) - 1, num=8, dtype=int)
            anchors.update(pool[i] for i in idxs)

    # Small integer / half-step neighborhoods for count-like features
    for delta in (0, 1, 2, 3, 4, 5):
        anchors.add(cur - delta)
        anchors.add(cur + delta)
    return sorted({a for a in anchors if np.isfinite(a)})


def _prune_atomic_redundancy(conditions: Sequence[Condition]) -> list[Condition]:
    """Keep equality plus a few distinct inequality strengths per feature."""
    by_feature: dict[str, list[Condition]] = {}
    for c in conditions:
        by_feature.setdefault(c.feature, []).append(c)

    kept: list[Condition] = []
    for feature, group in by_feature.items():
        eqs = [c for c in group if c.operator == Operator.EQ]
        kept.extend(eqs)

        for direction in (
            (Operator.GE, Operator.GT),
            (Operator.LE, Operator.LT),
        ):
            subset = [c for c in group if c.operator in direction]
            if not subset:
                continue
            # Sort so more extreme thresholds come first for GE/GT, etc.
            reverse = direction[0] in (Operator.GE, Operator.GT)
            subset = sorted(subset, key=lambda c: float(c.value), reverse=reverse)
            selected: list[Condition] = []
            for c in subset:
                if any(conditions_redundant(c, s) for s in selected):
                    # Replace with stricter if needed
                    for i, s in enumerate(selected):
                        if conditions_redundant(c, s):
                            selected[i] = stricter_condition(c, s)
                            break
                else:
                    selected.append(c)
                if len(selected) >= 4:
                    break
            kept.extend(selected)

        # Preserve any NE conditions
        kept.extend(c for c in group if c.operator == Operator.NE)

    return kept


# Feature pairs that are nearly always redundant together for CFB/home-away data.
_REDUNDANT_FEATURE_PAIRS = frozenset(
    {
        frozenset({"is_home", "neutral_site"}),
        frozenset({"month", "month_name"}),
        frozenset({"wins", "win_pct"}),
        frozenset({"losses", "win_pct"}),
        frozenset({"opp_wins", "opp_win_pct"}),
        frozenset({"opp_losses", "opp_win_pct"}),
        frozenset({"has_winning_record", "has_losing_record"}),
        frozenset({"opp_has_winning_record", "opp_has_losing_record"}),
        frozenset({"win_streak", "loss_streak"}),
        frozenset({"opp_win_streak", "opp_loss_streak"}),
    }
)


def can_combine(existing: Sequence[Condition], new: Condition) -> bool:
    """Reject combinations that add a redundant condition on an existing feature."""
    for cond in existing:
        if cond.feature == new.feature:
            return False
        pair = frozenset({cond.feature, new.feature})
        if pair in _REDUNDANT_FEATURE_PAIRS:
            return False
    return True


def precompute_masks(
    df: pd.DataFrame,
    conditions: Sequence[Condition],
) -> dict[tuple, np.ndarray]:
    """Cache boolean masks keyed by condition identity."""
    masks: dict[tuple, np.ndarray] = {}
    for cond in conditions:
        if cond.feature not in df.columns:
            masks[cond.key()] = np.zeros(len(df), dtype=bool)
            continue
        masks[cond.key()] = evaluate_mask(df[cond.feature], cond)
    return masks
