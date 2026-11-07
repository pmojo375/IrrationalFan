"""Subgroup discovery via beam search (strategy-pluggable)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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
from discovery.engine.validation import apply_conditions, recount_record



# Families are intentionally broad: records and previous-game scoring must not
# monopolize the beam before rarer contextual features can combine.
def _feature_family(name: str) -> str:
    n = name.removeprefix("opp_")
    if n in {"wins", "losses", "games_played", "win_pct", "win_streak",
             "loss_streak", "has_winning_record", "has_losing_record", "win_diff"}:
        return "record"
    if n.startswith("previous_"):
        return "previous"
    if n in {"rest_days", "after_bye", "days_since_last_game"}:
        return "rest"
    if n in {"month", "month_name", "day_of_week", "week"}:
        return "calendar"
    if n in {"is_home", "neutral_site", "venue", "attendance",
             "conference", "division", "conference_competition"}:
        return "setting"
    if "rank" in n or n in {"rank", "ranking"}:
        return "ranking"
    return "advanced"


def _novelty(conditions: Sequence[Condition]) -> float:
    families = {_feature_family(c.feature) for c in conditions}
    uncommon = families - {"record", "previous"}
    # A single unusual dimension is worthwhile, but several distinct
    # dimensions make a more interesting story than extra score thresholds.
    return min(2.0, len(uncommon) * 0.75 + max(0, len(families) - 1) * 0.25)


def _diverse_beam(candidates: list[Candidate], width: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    if width <= 0:
        return []
    # Reserve up to one third of slots for distinct non-record families.
    reserved = max(1, width // 3)
    chosen: list[Candidate] = []
    used: set[tuple] = set()
    for family in ("advanced", "setting", "rest", "ranking", "calendar"):
        if len(chosen) >= reserved:
            break
        for cand in ordered:
            key = tuple(c.key() for c in cand.conditions)
            if key not in used and any(_feature_family(c.feature) == family for c in cand.conditions):
                chosen.append(cand)
                used.add(key)
                break
    for cand in ordered:
        if len(chosen) >= width:
            break
        key = tuple(c.key() for c in cand.conditions)
        if key not in used:
            chosen.append(cand)
            used.add(key)
    return chosen


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

        required = config.required_features
        atomics = generate_atomic_conditions(
            df, features, current, max_candidates=config.max_atomic_candidates
        )
        # Reserve candidates for every required stat independently of the global budget.
        for name in required:
            extra = generate_atomic_conditions(df, [f for f in features if f.name == name], current,
                                               max_candidates=12)
            atomics.extend(c for c in extra if c.key() not in {a.key() for a in atomics})
        if any(not any(c.feature == name for c in atomics) for name in required):
            return [], stats
        stats.atomic_conditions = len(atomics)
        if not atomics:
            return [], stats

        masks = precompute_masks(df, atomics)
        evaluated = 0

        # Depth-1 candidates
        beam: list[Candidate] = []
        discoveries: list[Discovery] = []

        for cond in atomics:
            if required and cond.feature != required[0]:
                continue
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

        beam = (_diverse_beam(beam, config.beam_width) if config.diverse_search
                else sorted(beam, key=lambda c: c.score, reverse=True)[: config.beam_width])

        max_depth = min(config.max_depth, config.max_conditions)
        for _depth in range(2, max_depth + 1):
            next_beam: list[Candidate] = []
            seen_sets: set[frozenset] = set()

            for parent in beam:
                for cond in atomics:
                    if _depth <= len(required) and cond.feature != required[_depth - 1]:
                        continue
                    if not can_combine(parent.conditions, cond):
                        continue
                    keyset = frozenset(c.key() for c in parent.conditions) | {cond.key()}
                    if keyset in seen_sets:
                        continue
                    seen_sets.add(keyset)

                    child_mask = parent.mask & masks[cond.key()]
                    # Skip if intersection unchanged (new condition adds nothing)
                    if child_mask.sum() == parent.mask.sum() and cond.feature not in required:
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

            next_beam = (_diverse_beam(next_beam, config.beam_width) if config.diverse_search
                         else sorted(next_beam, key=lambda c: c.score, reverse=True)[: config.beam_width])
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
        if set(config.required_features).issubset(c.feature for c in conditions) and win_pct < config.min_success_rate and not (losses == 0 and wins >= config.min_sample_size):
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
        if config.novelty_bonus:
            score += config.novelty_bonus * _novelty(conditions)
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
        if not set(config.required_features).issubset(c.feature for c in cand.conditions):
            return []
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
        config = self.config
        if scope == "opponent" and config.opponent_feature_map:
            remap = lambda names: tuple(config.opponent_feature_map.get(n, n) for n in names)
            config = replace(config,
                             allowed_features=remap(config.allowed_features) if config.allowed_features is not None else None,
                             required_features=remap(config.required_features),
                             excluded_features=remap(config.excluded_features))
        features = [f for f in features
                    if (config.allowed_features is None or f.name in config.allowed_features)
                    and f.name not in config.excluded_features]
        discoveries, stats = self.strategy.search(
            df,
            features,
            current,
            scope=scope,
            target_col=target_col,
            date_col=date_col,
            config=config,
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
            if self.config.novelty_bonus:
                score += self.config.novelty_bonus * _novelty(disc.conditions)
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
        deduped = _evidence_diverse_top(deduped, scopes, top_n=top_n)
        total.discoveries_after_dedup = len(deduped)
        total.elapsed_seconds = time.perf_counter() - t0
        return deduped, total


def _evidence_diverse_top(
    discoveries: list[Discovery],
    scopes: Sequence[tuple[str, pd.DataFrame, Mapping[str, Any], str, str]],
    top_n: int,
    pool_size: int = 200,
    max_overlap: float = 0.75,
    min_selection_score: float = 20.0,
) -> list[Discovery]:
    """Rerank a small shortlist by quality, distinct stories and actual game overlap.

    Only materialize game-id sets for the top candidates. Keep evidence separated
    by scope: opponent losses and team wins refer to different narratives.
    """
    if top_n <= 0 or not discoveries:
        return []
    frames = {name: frame for name, frame, *_ in scopes}
    ordered = sorted(discoveries, key=lambda d: d.ranking_score, reverse=True)[:pool_size]
    evidence: list[frozenset] = []
    for d in ordered:
        frame = frames.get(d.scope)
        if frame is None or "game_id" not in frame.columns:
            evidence.append(frozenset())
            continue
        mask = apply_conditions(frame, d.conditions)
        evidence.append(frozenset(frame.loc[mask, "game_id"].dropna().tolist()))

    signatures = [narrative_signature(d) for d in ordered]

    def select(max_streaks: int) -> list[tuple[int, float]]:
        selected: list[tuple[int, float]] = []
        objective_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        while len(selected) < min(top_n, len(ordered)):
            best_index = None
            best_score = float("-inf")
            for i, d in enumerate(ordered):
                if any(i == j for j, _ in selected):
                    continue
                if d.objective == "streak" and objective_counts.get("streak", 0) >= max_streaks:
                    continue
                # Threshold changes, aliases and extra clauses do not make a
                # nested version of an already selected story a new narrative.
                if any(
                    ordered[j].scope == d.scope
                    and (signatures[i] <= signatures[j] or signatures[j] <= signatures[i])
                    for j, _ in selected
                ):
                    continue
                overlaps = []
                for j, _ in selected:
                    if ordered[j].scope != d.scope:
                        continue
                    union = evidence[i] | evidence[j]
                    if union:
                        overlaps.append(len(evidence[i] & evidence[j]) / len(union))
                overlap = max(overlaps, default=0.0)
                if overlap >= max_overlap:
                    continue
                score = d.ranking_score - 22.0 * overlap
                score -= 5.0 * max(0, len(d.conditions) - 2) ** 2
                score -= min(8.0, 2.0 * objective_counts.get(d.objective, 0))
                score -= min(6.0, 2.0 * scope_counts.get(d.scope, 0))
                features = signatures[i]
                dimensions = sum(bool(features & group) for group in (
                    {"month", "day_of_week", "week"},
                    {"wins", "losses", "win_streak"},
                    {"opp_wins", "opp_losses", "opp_win_streak"},
                    {"is_home", "neutral_site", "venue"},
                    {"previous_points_for", "previous_points_against", "previous_margin", "previous_won"},
                    {"rest_days", "after_bye", "opp_rest_days", "opp_after_bye"},
                    {"rank", "opp_rank"},
                    {"conference", "opp_conference", "conference_wins", "conference_losses"},
                ))
                score += 2.0 * max(0, dimensions - 1)
                if score > best_score:
                    best_score, best_index = score, i
            if best_index is None or best_score < min_selection_score:
                break
            d = ordered[best_index]
            selected.append((best_index, best_score))
            objective_counts[d.objective] = objective_counts.get(d.objective, 0) + 1
            scope_counts[d.scope] = scope_counts.get(d.scope, 0) + 1
        return selected

    # A requested-length cap alone lets a short report consist entirely of
    # streaks. Tighten against the actual result count and rerun so records
    # previously blocked by removed streaks can compete. Allow one standout
    # streak for tiny reports, including when no strong records exist.
    max_streaks = max(1, top_n * 2 // 5)
    while True:
        selected = select(max_streaks)
        allowed = max(1, len(selected) * 2 // 5)
        streaks = sum(ordered[i].objective == "streak" for i, _ in selected)
        if streaks <= allowed:
            break
        max_streaks = allowed

    # Keep raw candidates intact across selection passes and repeat calls.
    return [replace(ordered[i], ranking_score=score) for i, score in selected]
