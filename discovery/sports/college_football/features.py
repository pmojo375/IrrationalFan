"""Feature definitions and safe pregame enrichment for CFB observations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from discovery.engine.models import FeatureSpec

# Columns that reflect postgame information and must never be used as conditions.
LEAKAGE_COLUMNS = frozenset(
    {
        "team_score",
        "opponent_score",
        "won",
        "favorable",
        "attendance",  # typically finalized after the game
        "status",
    }
)

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def feature_specs() -> list[FeatureSpec]:
    """Pregame features available for automatic condition generation."""
    return [
        FeatureSpec("is_home", "boolean", "Home game", "home game"),
        FeatureSpec("neutral_site", "boolean", "Neutral site", "neutral-site game"),
        FeatureSpec("conference_competition", "boolean", "Conference game", "conference game"),
        FeatureSpec("month", "categorical", "Month", "month"),
        FeatureSpec("month_name", "categorical", "Month", "month"),
        FeatureSpec("week", "numeric", "Week", "week"),
        FeatureSpec("wins", "numeric", "Pregame wins", "entering with wins"),
        FeatureSpec("losses", "numeric", "Pregame losses", "entering with losses"),
        FeatureSpec("games_played", "numeric", "Games played", "games played"),
        FeatureSpec("win_pct", "numeric", "Pregame win percentage", "win percentage"),
        FeatureSpec("win_streak", "numeric", "Win streak", "win streak"),
        FeatureSpec("loss_streak", "numeric", "Loss streak", "loss streak"),
        FeatureSpec("has_winning_record", "boolean", "Winning record", "winning record"),
        FeatureSpec("has_losing_record", "boolean", "Losing record", "losing record"),
        FeatureSpec("opp_wins", "numeric", "Opponent pregame wins", "opponent wins"),
        FeatureSpec("opp_losses", "numeric", "Opponent pregame losses", "opponent losses"),
        FeatureSpec("opp_games_played", "numeric", "Opponent games played", "opponent games played"),
        FeatureSpec("opp_win_pct", "numeric", "Opponent win percentage", "opponent win percentage"),
        FeatureSpec("opp_win_streak", "numeric", "Opponent win streak", "opponent win streak"),
        FeatureSpec("opp_loss_streak", "numeric", "Opponent loss streak", "opponent loss streak"),
        FeatureSpec("opp_has_winning_record", "boolean", "Opponent winning record", "opponent winning record"),
        FeatureSpec("opp_has_losing_record", "boolean", "Opponent losing record", "opponent losing record"),
        FeatureSpec("win_diff", "numeric", "Win differential vs opponent", "win differential"),
        FeatureSpec("day_of_week", "categorical", "Day of week", "day of week"),
    ]


def enrich_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional pregame features without touching postgame outcomes as inputs.

    Opponent records are joined from the sibling team-perspective row of the same game.
    Streaks are computed from prior games only (shift by one within each team).
    """
    out = df.copy()

    # Ensure chronological ordering for streak math
    if "game_date" in out.columns:
        out["_sort_date"] = pd.to_datetime(out["game_date"], utc=True, errors="coerce")
    else:
        out["_sort_date"] = pd.NaT

    out = out.sort_values(
        ["team_id", "_sort_date", "game_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Entering streaks use only prior game results (current game excluded).
    out["win_streak"] = (
        out.groupby("team_id", group_keys=False)["won"]
        .apply(lambda s: _entering_streaks(s, positive=True))
    )
    out["loss_streak"] = (
        out.groupby("team_id", group_keys=False)["won"]
        .apply(lambda s: _entering_streaks(s, positive=False))
    )
    out["win_streak"] = out["win_streak"].astype(int)
    out["loss_streak"] = out["loss_streak"].astype(int)

    out["games_played"] = out["wins"] + out["losses"]
    out["win_pct"] = np.where(
        out["games_played"] > 0,
        out["wins"] / out["games_played"],
        np.nan,
    )
    out["has_winning_record"] = out["wins"] > out["losses"]
    out["has_losing_record"] = out["losses"] > out["wins"]

    if "month" in out.columns:
        out["month_name"] = out["month"].map(MONTH_NAMES)
    else:
        out["month_name"] = pd.NA

    out["day_of_week"] = out["_sort_date"].dt.day_name()

    # Opponent pregame features via same-game sibling row
    sibling = out[
        [
            "game_id",
            "team_id",
            "wins",
            "losses",
            "games_played",
            "win_pct",
            "win_streak",
            "loss_streak",
            "has_winning_record",
            "has_losing_record",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "wins": "opp_wins",
            "losses": "opp_losses",
            "games_played": "opp_games_played",
            "win_pct": "opp_win_pct",
            "win_streak": "opp_win_streak",
            "loss_streak": "opp_loss_streak",
            "has_winning_record": "opp_has_winning_record",
            "has_losing_record": "opp_has_losing_record",
        }
    )
    out = out.merge(sibling, on=["game_id", "opponent_id"], how="left")

    out["win_diff"] = out["wins"] - out["opp_wins"]

    out = out.drop(columns=["_sort_date"])
    return out


def _entering_streaks(won: pd.Series, positive: bool = True) -> pd.Series:
    """
    Entering streak length before each game.

    For each row, the streak reflects only earlier games for that team.
    """
    values: list[int] = []
    streak = 0
    for result in won.astype(bool):
        values.append(streak)
        good = bool(result) if positive else not bool(result)
        streak = streak + 1 if good else 0
    return pd.Series(values, index=won.index)


def assert_no_leakage(condition_features: Sequence[str]) -> None:
    bad = [f for f in condition_features if f in LEAKAGE_COLUMNS]
    if bad:
        raise ValueError(f"Pregame conditions must not use postgame columns: {bad}")
