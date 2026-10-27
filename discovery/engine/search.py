"""Subgroup discovery via beam search (strategy-pluggable)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from discovery.engine.conditions import (
    can_combine,
    conditions_apply_to_current,
    generate_atomic_conditions,
    precompute_masks,
)
from discovery.engine.models import (
    Condition,
    Discovery,
    FeatureSpec,
    RankingWeights,
    SearchConfig,
    SearchStats,
)
from discovery.engine.ranking import deduplicate_discoveries, score_discovery, narrative_signature
from discovery.engine.streaks import recent_favorable_streak
from discovery.engine.validation import recount_record


DescriptionFn = Callable[[str, Sequence[Condition], Discovery], str]


@dataclass
class Candidate:
    conditions: tuple[Condition, ...]
    mask: np.ndarray
    wins: int
    losses: int
    streak: int
    score: float


class SearchStrategy(ABC):
    """Interface so alternative search strategies can replace beam search later."""

    @abstractmethod
    def search(
        self,
        df: pd.DataFrame,
        features: Sequence[FeatureSpec],
        current: Mapping[str, Any],
        *,
        scope: str,
        target_col: str,
        date_col: str,
        config: SearchConfig,
        weights: RankingWeights,
        favorable_label: str = "wins",
    ) -> tuple[list[Discovery], SearchStats]:
        raise NotImplementedError


class BeamSearchStrategy(SearchStrategy):
    """Efficient beam search over condition combinations using cached masks."""

    def search(
        self,
        df: pd.DataFrame,
        features: Sequence[FeatureSpec],
        current: Mapping[str, Any],
        *,
        scope: str,
        target_col: str,
        date_col: str,
        config: SearchConfig,
        weights: RankingWeights,
        favorable_label: str = "wins",
    ) -> tuple[list[Discovery], SearchStats]:
        t0 = time.perf_counter()
        stats = SearchStats()
        population_size = len(df)
        if population_size == 0:
            return [], stats

        atomics = generate_atomic_conditions(
            df, features, current, max_candidates=config.max_atomic_candidates
        )
        stats.atomic_conditions = len(atomics)
        if not atomics:
            return [], stats

        masks = precompute_masks(df, atomics)
        evaluated = 0

        # Depth-1 candidates
        beam: list[Candidate] = []
        discoveries: list[Discovery] = []

        for cond in atomics:
            mask = masks[cond.key()]
            cand, evals = self._evaluate_candidate(
                df,
                (cond,),
                mask,
                target_col=target_col,
                date_col=date_col,
                config=config,
                weights=weights,
                population_size=population_size,
            )
            evaluated += evals
            if cand is None:
                continue
            beam.append(cand)
            discoveries.extend(
                self._candidates_to_discoveries(
                    df,
                    cand,
                    scope=scope,
                    target_col=target_col,
                    date_col=date_col,
                    favorable_label=favorable_label,
                    population_size=population_size,
                    config=config,
                )
            )

        beam = sorted(beam, key=lambda c: c.score, reverse=True)[: config.beam_width]

        max_depth = min(config.max_depth, config.max_conditions)
        for _depth in range(2, max_depth + 1):
            next_beam: list[Candidate] = []
            seen_sets: set[frozenset] = set()

            for parent in beam:
                for cond in atomics:
                    if not can_combine(parent.conditions, cond):
                        continue
                    keyset = frozenset(c.key() for c in parent.conditions) | {cond.key()}
                    if keyset in seen_sets:
                        continue
                    seen_sets.add(keyset)

                    child_mask = parent.mask & masks[cond.key()]
                    # Skip if intersection unchanged (new condition adds nothing)
                    if child_mask.sum() == parent.mask.sum():
                        continue

                    child_conds = tuple(sorted(parent.conditions + (cond,), key=lambda c: c.key()))
                    cand, evals = self._evaluate_candidate(
                        df,
                        child_conds,
                        child_mask,
                        target_col=target_col,
                        date_col=date_col,
                        config=config,
                        weights=weights,
                        population_size=population_size,
                    )
                    evaluated += evals
                    if cand is None:
                        continue
                    next_beam.append(cand)
                    discoveries.extend(
                        self._candidates_to_discoveries(
                            df,
                            cand,
                            scope=scope,
                            target_col=target_col,
                            date_col=date_col,
                            favorable_label=favorable_label,
                            population_size=population_size,
                            config=config,
                        )
                    )

            next_beam = sorted(next_beam, key=lambda c: c.score, reverse=True)[: config.beam_width]
            if not next_beam:
                break
            beam = next_beam

        stats.candidates_evaluated = evaluated
        stats.discoveries_before_dedup = len(discoveries)
        stats.elapsed_seconds = time.perf_counter() - t0
        return discoveries, stats

    def _evaluate_candidate(
        self,
        df: pd.DataFrame,
        conditions: tuple[Condition, ...],
        mask: np.ndarray,
        *,
        target_col: str,
        date_col: str,
        config: SearchConfig,
        weights: RankingWeights,
        population_size: int,
    ) -> tuple[Optional[Candidate], int]:
        n = int(mask.sum())
        if n < config.min_sample_size:
            return None, 1

        wins, losses, win_pct = recount_record(df, mask, target_col=target_col)
        if wins + losses < config.min_sample_size:
            return None, 1
        if win_pct < config.min_success_rate and not (losses == 0 and wins >= config.min_sample_size):
            # Still allow strong streaks even if overall rate is middling
            streak = recent_favorable_streak(df, mask, target_col=target_col, date_col=date_col)
            if streak < config.min_streak:
                return None, 1
        else:
            streak = recent_favorable_streak(df, mask, target_col=target_col, date_col=date_col)

        score = score_discovery(
            wins=wins,
            losses=losses,
            streak=streak,
            n_conditions=len(conditions),
            population_size=population_size,
            weights=weights,
            objective="record",
        )
        return (
            Candidate(
                conditions=conditions,
                mask=mask,
                wins=wins,
                losses=losses,
                streak=streak,
                score=score,
            ),
            1,
        )

    def _candidates_to_discoveries(
        self,
        df: pd.DataFrame,
        cand: Candidate,
        *,
        scope: str,
        target_col: str,
        date_col: str,
        favorable_label: str,
        population_size: int,
        config: SearchConfig,
    ) -> list[Discovery]:
        # Defer building full evidence lists until after ranking/dedup — materializing
        # matching games for every beam candidate dominated runtime.
        n_games = cand.wins + cand.losses
        win_pct = cand.wins / n_games if n_games else 0.0

        out: list[Discovery] = []
        # Record-based discovery
        if n_games >= config.min_sample_size and (
            win_pct >= config.min_success_rate or cand.losses == 0
        ):
            out.append(
                Discovery(
                    scope=scope,
                    conditions=cand.conditions,
                    wins=cand.wins,
                    losses=cand.losses,
                    n_games=n_games,
                    win_pct=win_pct,
                    streak=cand.streak,
                    ranking_score=cand.score,
                    description="",
                    matching_games=[],
                    objective="record",
                    favorable_label=favorable_label,
                    population_size=population_size,
                )
            )

        # Streak discoveries are most meaningful for a single team's timeline.
        # Cross-team chronological streaks in the "all" scope are usually noise.
        if cand.streak >= config.min_streak and scope in {"team", "opponent", "h2h"}:
            streak_weights = RankingWeights(
                win_pct=20.0,
                sample_size=10.0,
                streak=40.0,
                rarity=10.0,
                n_conditions=8.0,
                perfect_record_bonus=5.0,
            )
            streak_score = score_discovery(
                wins=cand.wins,
                losses=cand.losses,
                streak=cand.streak,
                n_conditions=len(cand.conditions),
                population_size=population_size,
                weights=streak_weights,
                objective="streak",
            )
            out.append(
                Discovery(
                    scope=scope,
                    conditions=cand.conditions,
                    wins=cand.wins,
                    losses=cand.losses,
                    n_games=n_games,
                    win_pct=win_pct,
                    streak=cand.streak,
                    ranking_score=streak_score,
                    description="",
                    matching_games=[],
                    objective="streak",
                    favorable_label=favorable_label,
                    population_size=population_size,
                )
            )
        return out


@dataclass
class DiscoveryEngine:
    """Facade that runs one or more scopes and returns ranked discoveries."""

    strategy: SearchStrategy = field(default_factory=BeamSearchStrategy)
    config: SearchConfig = field(default_factory=SearchConfig)
    weights: RankingWeights = field(default_factory=RankingWeights)
    describe: Optional[DescriptionFn] = None

    def run_scope(
        self,
        df: pd.DataFrame,
        features: Sequence[FeatureSpec],
        current: Mapping[str, Any],
        *,
        scope: str,
        target_col: str = "favorable",
        date_col: str = "game_date",
        favorable_label: str = "wins",
    ) -> tuple[list[Discovery], SearchStats]:
        discoveries, stats = self.strategy.search(
            df,
            features,
            current,
            scope=scope,
            target_col=target_col,
            date_col=date_col,
            config=self.config,
            weights=self.weights,
            favorable_label=favorable_label,
        )
        # Attach descriptions and re-score with configured weights for record objs.
        # Hard requirement: every condition must still describe the CURRENT matchup.
        final: list[Discovery] = []
        for disc in discoveries:
            if not conditions_apply_to_current(current, disc.conditions):
                continue
            score = score_discovery(
                wins=disc.wins,
                losses=disc.losses,
                streak=disc.streak,
                n_conditions=len(disc.conditions),
                population_size=disc.population_size,
                weights=self.weights,
                objective=disc.objective,
            )
            disc.ranking_score = score
            if self.describe is not None:
                disc.description = self.describe(scope, disc.conditions, disc)
            final.append(disc)
        return final, stats

    def run_many(
        self,
        scopes: Sequence[tuple[str, pd.DataFrame, Mapping[str, Any], str, str]],
        features: Sequence[FeatureSpec],
        *,
        date_col: str = "game_date",
        top_n: int = 20,
    ) -> tuple[list[Discovery], SearchStats]:
        """
        scopes items: (scope_name, dataframe, current_features, favorable_label, target_col)
        """
        t0 = time.perf_counter()
        all_disc: list[Discovery] = []
        total = SearchStats()

        for scope, df, current, favorable_label, target_col in scopes:
            discs, stats = self.run_scope(
                df,
                features,
                current,
                scope=scope,
                target_col=target_col,
                date_col=date_col,
                favorable_label=favorable_label,
            )
            all_disc.extend(discs)
            total.candidates_evaluated += stats.candidates_evaluated
            total.atomic_conditions += stats.atomic_conditions
            total.discoveries_before_dedup += stats.discoveries_before_dedup

        deduped = deduplicate_discoveries(all_disc)
        deduped = _diverse_top(deduped, top_n=top_n)
        total.discoveries_after_dedup = len(deduped)
        total.elapsed_seconds = time.perf_counter() - t0
        return deduped, total


def _diverse_top(discoveries: list[Discovery], top_n: int) -> list[Discovery]:
    """Greedy diversity selection: avoid near-identical stories in the final report.

    A scope/objective can contribute several results, but only one version of a
    feature combination. Do not pad the output with repetitions to reach top_n.
    """
    if top_n <= 0:
        return []
    ordered = sorted(discoveries, key=lambda d: d.ranking_score, reverse=True)
    selected: list[Discovery] = []
    signatures: set[tuple[str, frozenset[str]]] = set()
    # Reserve one opportunity per scope, then select the strongest distinct stories.
    for scope in dict.fromkeys(d.scope for d in ordered):
        for d in ordered:
            signature = (d.scope, narrative_signature(d))
            if d.scope == scope and signature not in signatures:
                selected.append(d)
                signatures.add(signature)
                break
        if len(selected) >= top_n:
            return selected
    for d in ordered:
        signature = (d.scope, narrative_signature(d))
        if signature in signatures:
            continue
        selected.append(d)
        signatures.add(signature)
        if len(selected) >= top_n:
            break
    return sorted(selected, key=lambda d: d.ranking_score, reverse=True)
