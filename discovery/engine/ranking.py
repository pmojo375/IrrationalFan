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
    prefilter_top_k: int = 250,
) -> list[Discovery]:
    """
    Remove duplicate or nearly identical discoveries.

    Prefer higher-scoring items. Uses cheap condition/record fingerprints only —
    evidence-list comparisons are intentionally avoided (they were O(n² * games)).
    """
    if not discoveries:
        return []

    ordered = sorted(discoveries, key=lambda d: d.ranking_score, reverse=True)
    if len(ordered) > prefilter_top_k:
        ordered = ordered[:prefilter_top_k]

    kept: list[Discovery] = []
    # Cache fingerprints so we never rebuild condition sets repeatedly.
    kept_meta: list[tuple[set[tuple], frozenset[str], int, int, int]] = []

    for disc in ordered:
        cond_keys = _cond_keys(disc.conditions)
        feats = frozenset(c.feature for c in disc.conditions)
        meta = (cond_keys, feats, disc.wins, disc.losses, disc.n_games)

        if any(
            _too_similar_fast(
                disc,
                other,
                cond_keys,
                feats,
                other_meta,
                similarity_threshold,
            )
            for other, other_meta in zip(kept, kept_meta)
        ):
            continue
        kept.append(disc)
        kept_meta.append(meta)
    return kept


def _too_similar_fast(
    a: Discovery,
    b: Discovery,
    cond_a: set[tuple],
    feats_a: frozenset[str],
    b_meta: tuple[set[tuple], frozenset[str], int, int, int],
    threshold: float,
) -> bool:
    if a.scope != b.scope or a.objective != b.objective:
        return False

    cond_b, feats_b, b_wins, b_losses, b_n = b_meta
    if a.wins == b_wins and a.losses == b_losses and a.n_games == b_n:
        cond_sim = _jaccard(cond_a, cond_b)
        if cond_sim >= threshold:
            return True
        if feats_a == feats_b:
            return True
        if _jaccard(feats_a, feats_b) >= 0.8 and cond_sim >= 0.4:
            return True
    return False


def _cond_keys(conditions: Iterable[Condition]) -> set[tuple]:
    return {c.key() for c in conditions}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
