"""Core data models for the discovery engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence


class Operator(str, Enum):
    EQ = "=="
    NE = "!="
    GE = ">="
    LE = "<="
    GT = ">"
    LT = "<"


@dataclass(frozen=True)
class Condition:
    """A single predicate over one feature."""

    feature: str
    operator: Operator
    value: Any
    label: str = ""

    def key(self) -> tuple:
        return (self.feature, self.operator.value, _freeze(self.value))

    def display(self) -> str:
        return self.label or f"{self.feature} {self.operator.value} {self.value}"


@dataclass(frozen=True)
class FeatureSpec:
    """Describe a searchable feature column."""

    name: str
    kind: str  # "numeric" | "categorical" | "boolean"
    label: str = ""
    # Human-facing phrasing helpers used by sports formatting layers.
    phrasing: str = ""


@dataclass
class RankingWeights:
    """Configurable weights for entertaining / surprising discoveries."""

    win_pct: float = 35.0
    sample_size: float = 18.0
    streak: float = 22.0
    rarity: float = 12.0
    n_conditions: float = 8.0
    perfect_record_bonus: float = 10.0
    # Soft-penalize tiny samples even if they pass min_sample_size.
    tiny_sample_penalty: float = 15.0
    tiny_sample_threshold: int = 4


@dataclass
class SearchConfig:
    """Beam-search and filtering parameters."""

    beam_width: int = 40
    max_depth: int = 4
    min_sample_size: int = 3
    max_conditions: int = 4
    max_atomic_candidates: int = 80
    scopes: Sequence[str] = ("team", "opponent", "all", "h2h")
    # Minimum win rate (or loss rate for opponent-loss searches) to keep.
    min_success_rate: float = 0.55
    # Prefer streaks at least this long when ranking streak discoveries.
    min_streak: int = 3


@dataclass
class MatchingGame:
    """One historical observation supporting a discovery."""

    season: Any
    date: Any
    team: str
    opponent: str
    team_score: Any
    opponent_score: Any
    won: bool
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Discovery:
    """A validated, ranked historical pattern."""

    scope: str
    conditions: tuple[Condition, ...]
    wins: int
    losses: int
    n_games: int
    win_pct: float
    streak: int
    ranking_score: float
    description: str
    matching_games: list[MatchingGame]
    objective: str = "record"  # "record" | "streak"
    favorable_label: str = "wins"  # "wins" | "losses"
    population_size: int = 0
    candidates_evaluated: int = 0

    @property
    def record_str(self) -> str:
        if self.favorable_label == "losses":
            # Present as opponent W-L from their perspective: losses-wins inverted wording
            # is handled in formatting; keep raw favorable/unfavorable counts here.
            return f"{self.wins}-{self.losses}"
        return f"{self.wins}-{self.losses}"


@dataclass
class SearchStats:
    """Runtime metrics for a discovery run."""

    elapsed_seconds: float = 0.0
    candidates_evaluated: int = 0
    atomic_conditions: int = 0
    discoveries_before_dedup: int = 0
    discoveries_after_dedup: int = 0


@dataclass
class ObservationFrame:
    """
    Domain-agnostic view of historical observations.

    `df` must include:
      - target column (boolean favorable outcome for the active scope)
      - feature columns listed in `features`
      - optional metadata columns for evidence formatting
    """

    df: Any  # pandas.DataFrame
    features: Sequence[FeatureSpec]
    target_col: str = "favorable"
    date_col: str = "game_date"
    meta_cols: Sequence[str] = ()


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    # Normalize numpy / pandas scalars
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def conditions_redundant(a: Condition, b: Condition) -> bool:
    """True if one numeric threshold condition implies the other on the same feature."""
    if a.feature != b.feature:
        return False
    if a.operator == b.operator == Operator.EQ:
        return a.value == b.value
    # Same direction inequalities: stricter implies weaker.
    try:
        av = float(a.value)
        bv = float(b.value)
    except (TypeError, ValueError):
        return a.key() == b.key()

    pairs = {
        (Operator.GE, Operator.GE): lambda x, y: x >= y,  # a implies b if a.value >= b.value
        (Operator.GT, Operator.GT): lambda x, y: x >= y,
        (Operator.LE, Operator.LE): lambda x, y: x <= y,
        (Operator.LT, Operator.LT): lambda x, y: x <= y,
        (Operator.GE, Operator.GT): lambda x, y: x > y,
        (Operator.GT, Operator.GE): lambda x, y: x >= y,
        (Operator.LE, Operator.LT): lambda x, y: x < y,
        (Operator.LT, Operator.LE): lambda x, y: x <= y,
    }
    fn = pairs.get((a.operator, b.operator))
    if fn is None:
        return a.key() == b.key()
    # a is redundant with b (or vice versa) if one implies the other.
    a_implies_b = fn(av, bv)
    b_implies_a = pairs.get((b.operator, a.operator), lambda *_: False)(bv, av)
    return a_implies_b or b_implies_a


def stricter_condition(a: Condition, b: Condition) -> Condition:
    """Prefer the more specific of two redundant conditions."""
    if a.feature != b.feature:
        return a
    try:
        av = float(a.value)
        bv = float(b.value)
    except (TypeError, ValueError):
        return a

    if a.operator in (Operator.GE, Operator.GT) and b.operator in (Operator.GE, Operator.GT):
        return a if av >= bv else b
    if a.operator in (Operator.LE, Operator.LT) and b.operator in (Operator.LE, Operator.LT):
        return a if av <= bv else b
    return a
