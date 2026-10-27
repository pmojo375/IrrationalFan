"""Feature definitions and safe pregame enrichment for CFB observations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from discovery.engine.models import FeatureSpec
from discovery.sports.college_football.dates import parse_game_dates

# True postgame targets / identifiers — never searchable conditions.
LEAKAGE_COLUMNS = frozenset(
    {
        "team_score",
        "opponent_score",
        "won",
        "favorable",
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

# Curated advanced / goofy columns from the weekly summaries + schedule metadata.
# Kept selective so beam search stays tractable while still using the richer data.
_ADVANCED_TEAM_FEATURES: list[tuple[str, str, str]] = [
    # (column, kind, label)
    ("conference", "categorical", "Conference"),
    ("division", "categorical", "Division"),
    ("venue", "categorical", "Venue"),
    ("attendance", "numeric", "Attendance"),
    ("EPAplay_off", "numeric", "Offensive EPA/play"),
    ("EPAplay_def", "numeric", "Defensive EPA/play allowed"),
    ("EPAplay_margin", "numeric", "EPA/play margin"),
    ("success_off", "numeric", "Offensive success rate"),
    ("success_def", "numeric", "Defensive success rate allowed"),
    ("success_margin", "numeric", "Success-rate margin"),
    ("explosive_off", "numeric", "Offensive explosive play rate"),
    ("explosive_def", "numeric", "Explosive plays allowed"),
    ("havoc_off", "numeric", "Havoc rate allowed on offense"),
    ("havoc_def", "numeric", "Defensive havoc rate"),
    ("havoc_margin", "numeric", "Havoc margin"),
    ("yardsplay_off", "numeric", "Yards per play"),
    ("yardsplay_def", "numeric", "Yards per play allowed"),
    ("third_down_success_off", "numeric", "Third-down conversion rate"),
    ("third_down_success_def", "numeric", "Opponent third-down rate"),
    ("red_zone_success_off", "numeric", "Red-zone success rate"),
    ("red_zone_success_def", "numeric", "Red-zone success allowed"),
    ("turnovers_off", "numeric", "Turnovers committed per game"),
    ("turnovers_def", "numeric", "Turnovers forced per game"),
    ("turnover_margin", "numeric", "Turnover margin"),
    ("passrate_off", "numeric", "Pass rate"),
    ("rushrate_off", "numeric", "Rush rate"),
    ("playsgame_off", "numeric", "Plays per game"),
    ("EPAplay_off_rank", "numeric", "Offensive EPA/play rank"),
    ("EPAplay_def_rank", "numeric", "Defensive EPA/play rank"),
    ("success_off_rank", "numeric", "Offensive success-rate rank"),
    ("success_def_rank", "numeric", "Defensive success-rate rank"),
    ("havoc_def_rank", "numeric", "Havoc rank"),
    ("explosive_off_rank", "numeric", "Explosive-play rank"),
]


def feature_specs() -> list[FeatureSpec]:
    """Pregame features available for automatic condition generation."""
    specs = [
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

    # Advanced efficiency metrics are useful for analysis, but overwhelm fan-facing
    # discoveries. Keep the definitions available for an explicit opt-in.
    return specs


def advanced_feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for name, kind, label in _ADVANCED_TEAM_FEATURES:
        specs.append(FeatureSpec(name, kind, label, label.lower()))
        # Opponent copies already live on rows as opp_* from process.py
        if name not in {"venue", "attendance"}:
            specs.append(
                FeatureSpec(
                    f"opp_{name}",
                    kind,
                    f"Opponent {label[0].lower() + label[1:]}",
                    f"opponent {label.lower()}",
                )
            )

    return specs


def searchable_feature_names() -> set[str]:
    return {f.name for f in [*feature_specs(), *advanced_feature_specs()] if f.name not in LEAKAGE_COLUMNS}


def enrich_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional pregame features without touching postgame outcomes as inputs.

    Opponent records are joined from the sibling team-perspective row of the same game.
    Streaks are computed from prior games only (shift by one within each team).
    """
    out = df.copy()

    # Ensure chronological ordering for streak math
    if "game_date" in out.columns:
        out["_sort_date"] = parse_game_dates(out["game_date"])
    else:
        out["_sort_date"] = pd.NaT

    sort_cols = ["season", "team_id", "_sort_date", "game_id"] if "season" in out.columns else ["team_id", "_sort_date", "game_id"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Entering streaks use only prior game results within a season (current game excluded).
    streak_group = ["season", "team_id"] if "season" in out.columns else ["team_id"]
    out["win_streak"] = (
        out.groupby(streak_group, group_keys=False)["won"]
        .apply(lambda s: _entering_streaks(s, positive=True))
    )
    out["loss_streak"] = (
        out.groupby(streak_group, group_keys=False)["won"]
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

    # Opponent W-L / streak features via same-game sibling row.
    # Skip columns that process.py already attached from weekly summaries.
    record_cols = [
        "wins",
        "losses",
        "games_played",
        "win_pct",
        "win_streak",
        "loss_streak",
        "has_winning_record",
        "has_losing_record",
    ]
    sibling = out[["game_id", "team_id", *record_cols]].rename(
        columns={
            "team_id": "opponent_id",
            **{c: f"opp_{c}" for c in record_cols},
        }
    )
    existing_opp = [c for c in sibling.columns if c in out.columns and c not in {"game_id", "opponent_id"}]
    sibling = sibling.drop(columns=existing_opp, errors="ignore")
    out = out.merge(sibling, on=["game_id", "opponent_id"], how="left")

    if "opp_wins" in out.columns:
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
