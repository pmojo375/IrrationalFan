"""Independent validation of discoveries against matching historical rows."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from discovery.engine.conditions import evaluate_mask
from discovery.engine.models import Condition, Discovery, MatchingGame
from discovery.engine.streaks import recent_favorable_streak


def apply_conditions(df: pd.DataFrame, conditions: Sequence[Condition]) -> np.ndarray:
    """Intersect condition masks; missing values never satisfy a condition."""
    if not conditions:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for cond in conditions:
        if cond.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)
        mask &= evaluate_mask(df[cond.feature], cond)
    return mask


def recount_record(
    df: pd.DataFrame,
    mask: np.ndarray,
    target_col: str = "favorable",
) -> tuple[int, int, float]:
    """Recalculate wins/losses (favorable/unfavorable) from matching rows."""
    if not mask.any():
        return 0, 0, 0.0
    outcomes = df.loc[mask, target_col].astype(bool)
    wins = int(outcomes.sum())
    losses = int((~outcomes).sum())
    n = wins + losses
    win_pct = wins / n if n else 0.0
    return wins, losses, win_pct


def build_matching_games(
    df: pd.DataFrame,
    mask: np.ndarray,
    *,
    date_col: str = "game_date",
    team_col: str = "team",
    opponent_col: str = "opponent",
    team_score_col: str = "team_score",
    opponent_score_col: str = "opponent_score",
    season_col: str = "season",
    won_col: str = "won",
    target_col: str = "favorable",
) -> list[MatchingGame]:
    """Materialize evidence rows sorted chronologically."""
    if not mask.any():
        return []
    subset = df.loc[mask].copy()
    if date_col in subset.columns:
        subset = subset.sort_values(date_col, kind="mergesort")

    games: list[MatchingGame] = []
    for _, row in subset.iterrows():
        won_val = bool(row[won_col]) if won_col in subset.columns else bool(row[target_col])
        games.append(
            MatchingGame(
                season=row[season_col] if season_col in subset.columns else None,
                date=row[date_col] if date_col in subset.columns else None,
                team=str(row[team_col]) if team_col in subset.columns else "",
                opponent=str(row[opponent_col]) if opponent_col in subset.columns else "",
                team_score=row[team_score_col] if team_score_col in subset.columns else None,
                opponent_score=row[opponent_score_col] if opponent_score_col in subset.columns else None,
                won=won_val,
            )
        )
    return games


def validate_discovery(
    df: pd.DataFrame,
    conditions: Sequence[Condition],
    *,
    scope: str,
    target_col: str = "favorable",
    date_col: str = "game_date",
    objective: str = "record",
    favorable_label: str = "wins",
    description: str = "",
    ranking_score: float = 0.0,
    population_size: int = 0,
) -> Optional[Discovery]:
    """Rebuild a discovery from scratch; return None if empty."""
    mask = apply_conditions(df, conditions)
    wins, losses, win_pct = recount_record(df, mask, target_col=target_col)
    if wins + losses == 0:
        return None
    streak = recent_favorable_streak(df, mask, target_col=target_col, date_col=date_col)
    games = build_matching_games(df, mask, date_col=date_col, target_col=target_col, won_col="won")
    return Discovery(
        scope=scope,
        conditions=tuple(conditions),
        wins=wins,
        losses=losses,
        n_games=wins + losses,
        win_pct=win_pct,
        streak=streak,
        ranking_score=ranking_score,
        description=description,
        matching_games=games,
        objective=objective,
        favorable_label=favorable_label,
        population_size=population_size,
    )
