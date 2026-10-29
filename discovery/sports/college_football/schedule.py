"""Resolve upcoming games from the unfiltered source schedule, not scored history."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from discovery.sports.college_football.dates import parse_game_dates

ROOT = Path(__file__).resolve().parents[3]
SCHEDULE_DIR = ROOT / "data" / "schedules"
EASTERN = ZoneInfo("America/New_York")


def resolve_scheduled_game(
    team_id: int, opponent_id: int, date: str, *,
    home: bool | None = None, schedule_dir: Path | None = None,
    now: datetime | None = None,
) -> bool:
    """Return selected team's home flag or raise on absent, past, or contradictory games.

    The processed team-game file excludes unplayed games. Read raw schedule files
    so an arbitrary matchup can never be silently invented.
    """
    requested = pd.to_datetime(date, errors="coerce")
    if pd.isna(requested):
        raise ValueError(f"Invalid matchup date: {date!r}")
    requested_day = requested.date()
    today = (now or datetime.now(EASTERN)).astimezone(EASTERN).date()
    if requested_day <= today:
        raise ValueError(f"Discovery requires a future game date (after {today}).")

    directory = Path(schedule_dir) if schedule_dir is not None else SCHEDULE_DIR
    source = directory / f"cfb_schedules_{requested_day.year}.parquet"
    if not source.exists():
        raise ValueError(f"Cannot verify the matchup: missing raw schedule {source}.")
    raw = pd.read_parquet(source)
    required = {"game_date", "home_id", "away_id"} if "game_date" in raw else {"start_date", "home_id", "away_id"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Raw schedule is missing required columns: {sorted(required - set(raw.columns))}")
    date_col = "game_date" if "game_date" in raw else "start_date"
    dates = parse_game_dates(raw[date_col])
    local_dates = dates.dt.tz_convert(EASTERN).dt.date
    selected_home = pd.to_numeric(raw["home_id"], errors="coerce").eq(team_id) & pd.to_numeric(raw["away_id"], errors="coerce").eq(opponent_id)
    selected_away = pd.to_numeric(raw["away_id"], errors="coerce").eq(team_id) & pd.to_numeric(raw["home_id"], errors="coerce").eq(opponent_id)
    matches = raw.loc[(selected_home | selected_away) & local_dates.eq(requested_day)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one scheduled game between team IDs {team_id} and {opponent_id} on {requested_day}; found {len(matches)}.")
    row = matches.iloc[0]
    if "home_score" in matches.columns and "away_score" in matches.columns:
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            raise ValueError("The scheduled game already has a final score.")
    if "home_points" in matches.columns and "away_points" in matches.columns:
        if pd.notna(row["home_points"]) and pd.notna(row["away_points"]):
            raise ValueError("The scheduled game already has a final score.")
    if "status" in matches.columns and str(row["status"]).upper() in {"STATUS_FINAL", "FINAL", "STATUS_IN_PROGRESS", "IN_PROGRESS"}:
        raise ValueError("The scheduled game is not upcoming.")
    actual_home = bool(row["home_id"] == team_id)
    if home is not None and bool(home) != actual_home:
        raise ValueError(f"Incorrect venue: selected team is {'home' if actual_home else 'away'} according to the schedule.")
    return actual_home


def scheduled_context(team_id: int, opponent_id: int, date: str) -> tuple[dict, dict, object]:
    """Read metadata for the already-validated matchup from the saved schedule."""
    from discovery.sports.college_football.context import schedule_metadata

    requested = pd.Timestamp(date)
    raw = pd.read_parquet(SCHEDULE_DIR / f"cfb_schedules_{requested.year}.parquet")
    date_col = "game_date" if "game_date" in raw else "start_date"
    dates = parse_game_dates(raw[date_col])
    match = ((raw["home_id"].eq(team_id) & raw["away_id"].eq(opponent_id)) |
             (raw["away_id"].eq(team_id) & raw["home_id"].eq(opponent_id)))
    matches = raw.loc[match & dates.dt.tz_convert(EASTERN).dt.date.eq(requested.date())]
    if len(matches) != 1:
        raise ValueError("Schedule changed during discovery; rerun with a consistent cache")
    row = matches.iloc[0]
    home = bool(row["home_id"] == team_id)
    team, opponent = schedule_metadata(row, home=home), schedule_metadata(row, home=not home)
    # Seasonal team metadata fills only missing schedule conference names.
    directory = SCHEDULE_DIR.parent / "teams"
    source = directory / f"cfb_teams_{int(row.get('season', requested.year))}.parquet"
    if source.exists():
        from discovery.sports.college_football.context import normalize_conference
        teams = pd.read_parquet(source)
        column = "conference_name" if "conference_name" in teams else "conference"
        if column in teams:
            def conference(tid):
                values = teams.loc[teams["team_id"].eq(tid), column].dropna()
                return normalize_conference(values.iloc[-1]) if len(values) else pd.NA
            for values, tid, oid in [(team, team_id, opponent_id), (opponent, opponent_id, team_id)]:
                if pd.isna(values["conference"]):
                    values["conference"] = conference(tid)
                if pd.isna(values["opp_conference"]):
                    values["opp_conference"] = conference(oid)
    # A later matchup may have unplayed games before it. Their dates are known,
    # but their results must not be replaced by the latest completed result.
    kickoff = parse_game_dates(row[date_col])
    for values, tid in [(team, team_id), (opponent, opponent_id)]:
        prior = raw.loc[(raw["home_id"].eq(tid) | raw["away_id"].eq(tid))
                        & raw["season"].eq(row["season"]) & dates.lt(kickoff)]
        if "status" in prior:
            prior = prior.loc[~prior["status"].astype("string").str.upper().str.contains(
                "CANCEL|POSTPON", regex=True, na=False)]
        values["_prior_games"] = [
            (game[date_col], schedule_metadata(game, home=bool(game["home_id"] == tid))["conference_competition"])
            for _, game in prior.iterrows()
        ]
    return team, opponent, row[date_col]
