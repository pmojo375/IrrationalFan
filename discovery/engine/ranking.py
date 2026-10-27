"""Ranking and deduplication of discoveries for entertainment value."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from discovery.engine.models import Condition, Discovery, RankingWeights


def score_discovery(
    *,
    wins: int,
    losses: int,
    streak: int,
    n_conditions: int,
    population_size: int,
    weights: RankingWeights,
    objective: str = "record",
) -> float:
    """
    Score a pattern for how interesting / entertaining it is.

    Not intended as a statistical significance test.
    """
    n = wins + losses
    if n <= 0:
        return 0.0

    win_pct = wins / n
    # Dampen extreme rates with tiny samples.
    sample_factor = min(1.0, math.log1p(n) / math.log1p(20))
    rarity = 0.0
    if population_size > 0:
        rarity = 1.0 - (n / population_size)
        rarity = max(0.0, min(1.0, rarity))

    score = 0.0
    score += weights.win_pct * win_pct * sample_factor
    # Prefer meaningful but not gigantic samples (entertainment sweet spot ~5-25).
    sweet = math.exp(-0.5 * ((math.log1p(n) - math.log1p(12)) ** 2))
    score += weights.sample_size * sweet
    score += weights.rarity * rarity
    score += weights.n_conditions * min(1.0, n_conditions / 4.0)

    if objective == "streak":
        score += weights.streak * min(1.0, streak / 8.0) * (1.0 if streak >= 3 else 0.4)
    else:
        score += weights.streak * min(1.0, streak / 10.0) * 0.5

    if losses == 0 and wins >= 3:
        score += weights.perfect_record_bonus * sample_factor

    if n < weights.tiny_sample_threshold:
        score -= weights.tiny_sample_penalty * (weights.tiny_sample_threshold - n) / weights.tiny_sample_threshold

    # Mild preference for undefeated / perfect failure patterns
    if win_pct >= 0.999 and n >= 3:
        score += 5.0
    # Soft-penalize bloated near-coin-flip "patterns"
    if n >= 100 and win_pct < 0.8:
        score -= 12.0 * (1.0 - win_pct)

    return float(score)


def deduplicate_discoveries(
    discoveries: Sequence[Discovery],
    similarity_threshold: float = 0.75,
) -> list[Discovery]:
    """
    Remove duplicate or nearly identical discoveries.

    Prefer higher-scoring items. Two discoveries are near-duplicates when their
    condition sets heavily overlap and their evidence sets are similar.
    """
    ordered = sorted(discoveries, key=lambda d: d.ranking_score, reverse=True)
    kept: list[Discovery] = []

    for disc in ordered:
        if any(_too_similar(disc, other, similarity_threshold) for other in kept):
            continue
        kept.append(disc)
    return kept


def _too_similar(a: Discovery, b: Discovery, threshold: float) -> bool:
    if a.scope != b.scope:
        return False
    if a.objective != b.objective:
        return False

    cond_sim = _jaccard(_cond_keys(a.conditions), _cond_keys(b.conditions))
    if cond_sim >= threshold and a.wins == b.wins and a.losses == b.losses:
        return True

    # Same feature set with only threshold tweaks and nearly identical records
    feat_a = {c.feature for c in a.conditions}
    feat_b = {c.feature for c in b.conditions}
    if feat_a == feat_b and abs(a.win_pct - b.win_pct) < 1e-9 and a.n_games == b.n_games:
        return True

    # Same / overlapping features with nearly identical evidence → keep the better score only
    games_a = _game_keys(a)
    games_b = _game_keys(b)
    if games_a and games_b and _jaccard(games_a, games_b) >= 0.8:
        if a.wins == b.wins and a.losses == b.losses:
            if feat_a == feat_b or cond_sim >= 0.4 or _jaccard(feat_a, feat_b) >= 0.5:
                return True
            # Identical evidence sets are enough even with divergent condition wording
            if _jaccard(games_a, games_b) >= 0.99:
                return True

    return False


def _cond_keys(conditions: Iterable[Condition]) -> set[tuple]:
    return {c.key() for c in conditions}


def _game_keys(discovery: Discovery) -> set[tuple]:
    keys: set[tuple] = set()
    for g in discovery.matching_games:
        keys.add((str(g.date), str(g.team), str(g.opponent), str(g.team_score), str(g.opponent_score)))
    return keys


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
