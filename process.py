"""Process CFB schedule parquet into one row per team per game with prior records."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).with_name("cfb_schedule_2026.parquet")
SUMMARIES_PATH = Path(__file__).with_name("cfb_team_summaries_weekly_2026.parquet")
OUTPUT_PATH = Path(__file__).with_name("cfb_team_games_2026.parquet")

# Identity / join keys from weekly summaries — not copied onto game rows as-is.
_SUMMARY_KEY_COLS = ("team_id", "pos_team", "season", "through_week")


def expand_to_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Turn each matchup into two rows: one for home, one for away."""
    common = [
        "game_id",
        "season",
        "week",
        "season_type",
        "game_date",
        "neutral_site",
        "conference_competition",
        "venue",
        "attendance",
        "status",
    ]

    home = games[common].copy()
    home["team_id"] = games["home_id"]
    home["team"] = games["home_team"]
    home["team_abbreviation"] = games["home_abbreviation"]
    home["opponent_id"] = games["away_id"]
    home["opponent"] = games["away_team"]
    home["opponent_abbreviation"] = games["away_abbreviation"]
    home["team_score"] = games["home_score"]
    home["opponent_score"] = games["away_score"]
    home["won"] = games["home_winner"]
    home["is_home"] = True

    away = games[common].copy()
    away["team_id"] = games["away_id"]
    away["team"] = games["away_team"]
    away["team_abbreviation"] = games["away_abbreviation"]
    away["opponent_id"] = games["home_id"]
    away["opponent"] = games["home_team"]
    away["opponent_abbreviation"] = games["home_abbreviation"]
    away["team_score"] = games["away_score"]
    away["opponent_score"] = games["home_score"]
    away["won"] = games["away_winner"]
    away["is_home"] = False

    return pd.concat([home, away], ignore_index=True)


def add_datetime_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Split game_date into year, month, day, and time (US Eastern)."""
    # America/New_York follows EDT/EST for the calendar (EDT through early Nov).
    dt = pd.to_datetime(df["game_date"], utc=True).dt.tz_convert("America/New_York")
    df = df.copy()
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["time"] = dt.dt.strftime("%H:%M")
    df["game_datetime"] = dt
    return df


def add_prior_records(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's W-L record entering the game (prior games only)."""
    df = df.sort_values(
        ["team_id", "game_datetime", "game_id"],
        kind="mergesort",
    ).copy()

    # Only final games count toward record; all rows here are final, but keep the guard.
    final_mask = df["status"].eq("STATUS_FINAL")
    win = (df["won"] & final_mask).astype(int)
    loss = ((~df["won"]) & final_mask).astype(int)

    # Prior record = cumulative through previous games only (exclude this game).
    df["wins"] = win.groupby(df["team_id"]).cumsum() - win
    df["losses"] = loss.groupby(df["team_id"]).cumsum() - loss
    df["record"] = df["wins"].astype(str) + "-" + df["losses"].astype(str)
    return df


def _prepare_summary_frame(summaries: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Normalize weekly summary keys and optionally prefix feature columns.

    `through_week` means cumulative stats after that week has been played.
    """
    out = summaries.copy()
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["season"] = out["season"].astype("int64")
    out["through_week"] = out["through_week"].astype("int64")

    feature_cols = [c for c in out.columns if c not in _SUMMARY_KEY_COLS]
    keep = ["team_id", "season", "through_week", *feature_cols]
    out = out[keep]

    if prefix:
        rename = {c: f"{prefix}{c}" for c in feature_cols}
        out = out.rename(columns=rename)

    # Drop duplicate team-week rows if any (keep last).
    out = out.drop_duplicates(subset=["team_id", "season", "through_week"], keep="last")
    return out


def add_weekly_summaries(df: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each team's (and opponent's) weekly summary as of the prior week.

    Join key: team_id + season + through_week == game.week - 1

    Week 1 and teams missing from the summary file remain null — never filled with zero.
    """
    base = df.copy()
    base["team_id"] = pd.to_numeric(base["team_id"], errors="coerce").astype("Int64")
    base["opponent_id"] = pd.to_numeric(base["opponent_id"], errors="coerce").astype("Int64")
    base["season"] = base["season"].astype("int64")
    base["week"] = base["week"].astype("int64")
    base["_prior_week"] = base["week"] - 1

    team_sum = _prepare_summary_frame(summaries, prefix="")
    team_sum = team_sum.rename(columns={"through_week": "_prior_week"})

    opp_sum = _prepare_summary_frame(summaries, prefix="opp_")
    opp_sum = opp_sum.rename(
        columns={
            "team_id": "opponent_id",
            "through_week": "_prior_week",
        }
    )

    before_cols = set(base.columns)

    merged = base.merge(team_sum, on=["team_id", "season", "_prior_week"], how="left")
    merged = merged.merge(opp_sum, on=["opponent_id", "season", "_prior_week"], how="left")
    merged = merged.drop(columns=["_prior_week"])

    added = [c for c in merged.columns if c not in before_cols]
    marker = "conference" if "conference" in merged.columns else (added[0] if added else None)
    team_hit = int(merged[marker].notna().sum()) if marker else 0
    print(
        f"Weekly summaries: attached {len(added)} columns; "
        f"{team_hit}/{len(merged)} rows have team prior-week stats "
        f"(week 1 and non-covered teams stay null)"
    )
    return merged


def main() -> None:
    games = pd.read_parquet(INPUT_PATH)
    team_games = expand_to_team_rows(games)
    team_games = add_datetime_parts(team_games)
    team_games = add_prior_records(team_games)

    if SUMMARIES_PATH.exists():
        summaries = pd.read_parquet(SUMMARIES_PATH)
        team_games = add_weekly_summaries(team_games, summaries)
    else:
        print(f"No summaries file at {SUMMARIES_PATH.name}; skipping weekly stats join")

    team_games = team_games.sort_values(
        ["game_datetime", "game_id", "is_home"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Core game columns first, then any attached summary features.
    core_columns = [
        "game_id",
        "season",
        "week",
        "season_type",
        "year",
        "month",
        "day",
        "time",
        "team_id",
        "team",
        "team_abbreviation",
        "opponent_id",
        "opponent",
        "opponent_abbreviation",
        "is_home",
        "neutral_site",
        "conference_competition",
        "team_score",
        "opponent_score",
        "won",
        "wins",
        "losses",
        "record",
        "venue",
        "attendance",
        "status",
        "game_date",
    ]
    extra = [c for c in team_games.columns if c not in core_columns and c != "game_datetime"]
    columns = core_columns + extra
    out = team_games[columns]
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to {OUTPUT_PATH.name}")
    preview_cols = [c for c in core_columns if c in out.columns][:12]
    if "conference" in out.columns:
        preview_cols += ["conference", "EPAplay_off", "opp_EPAplay_off"]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print(out.loc[out["week"] >= 2, preview_cols].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
