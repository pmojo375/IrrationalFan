"""Streak discovery over chronologically ordered qualifying games."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def recent_favorable_streak(
    df: pd.DataFrame,
    mask: np.ndarray,
    target_col: str = "favorable",
    date_col: str = "game_date",
) -> int:
    """
    Count consecutive favorable outcomes backward from the most recent
    qualifying game. Non-qualifying games do not interrupt the streak.
    """
    if not mask.any():
        return 0

    subset = df.loc[mask].copy()
    if date_col in subset.columns:
        subset = subset.sort_values(date_col, kind="mergesort")
    else:
        subset = subset.sort_index(kind="mergesort")

    outcomes = subset[target_col].astype(bool).to_numpy()
    streak = 0
    for outcome in outcomes[::-1]:
        if bool(outcome):
            streak += 1
        else:
            break
    return int(streak)


def streak_game_indices(
    df: pd.DataFrame,
    mask: np.ndarray,
    target_col: str = "favorable",
    date_col: str = "game_date",
) -> list[Any]:
    """Return index labels of games that form the current favorable streak."""
    if not mask.any():
        return []

    subset = df.loc[mask].copy()
    if date_col in subset.columns:
        subset = subset.sort_values(date_col, kind="mergesort")
    else:
        subset = subset.sort_index(kind="mergesort")

    streak_idxs: list[Any] = []
    for idx, outcome in zip(subset.index[::-1], subset[target_col].astype(bool).to_numpy()[::-1]):
        if bool(outcome):
            streak_idxs.append(idx)
        else:
            break
    streak_idxs.reverse()
    return streak_idxs
