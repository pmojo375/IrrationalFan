"""Process CFB schedule + weekly summary parquets into team-game rows.

Source data (sportsdataverse / cfbfastR):
  https://github.com/sportsdataverse/cfbfastR-cfb-data/tree/main/cfb/cfb_schedules/parquet
  https://github.com/sportsdataverse/cfbfastR-cfb-data/tree/main/cfb/cfb_team_summaries_weekly/parquet
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from discovery.sports.college_football.context import add_pregame_context, normalize_conference

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEDULES_DIR = DATA_DIR / "schedules"
SUMMARIES_DIR = DATA_DIR / "summaries"
TEAMS_DIR = DATA_DIR / "teams"
OUTPUT_PATH = ROOT / "cfb_team_games.parquet"

RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

# Identity / join keys from weekly summaries — not copied onto game rows as-is.
_SUMMARY_KEY_COLS = ("team_id", "pos_team", "season", "through_week")


def download_season_files(force: bool = False, workers: int = 6) -> None:
    """Cache source Parquets; refresh current-season files on every online build.

    Historical missing team assets (404) are optional; network/schema failures
    abort the build. Atomic replacement leaves existing caches intact on failure.
    """
    now = datetime.now(timezone.utc)
    # January bowls belong to the prior season, not the new calendar year.
    current = now.year if now.month >= 8 else now.year - 1
    jobs = []
    for directory, dataset, years in [
        (SCHEDULES_DIR, "cfb_schedules", range(2001, current + 1)),
        (SUMMARIES_DIR, "cfb_team_summaries_weekly", range(2004, current + 1)),
        (TEAMS_DIR, "cfb_teams", range(2004, current + 1)),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        tag = "espn_cfb_teams" if dataset == "cfb_teams" else dataset
        for year in years:
            name = f"{dataset}_{year}.parquet"
            jobs.append((directory / name, f"{RELEASES}/{tag}/{name}", year, dataset))

    def fetch(dest, url, year, dataset):
        if dest.exists() and not force and year < current:
            return f"skip {dest.name}"
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".parquet", delete=False) as f:
                temp = Path(f.name)
                with urllib.request.urlopen(url, timeout=90) as response:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
            raw = pd.read_parquet(temp)
            required = {"season", "team_id"} if dataset == "cfb_teams" else (
                {"season", "game_id"} if dataset == "cfb_schedules" else {"season", "team_id", "through_week"})
            if not required.issubset(raw.columns) or raw.empty:
                raise ValueError(f"Invalid {dataset} schema or empty dataset: {dest.name}")
            os.replace(temp, dest)
            dest.with_suffix(".source.json").write_text(json.dumps({
                "url": url, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(raw), "columns": len(raw.columns),
            }, indent=2) + "\n")
            return f"ok   {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)"
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and dataset == "cfb_teams" and year < current:
                return f"unavailable {dest.name} (using schedule/summary conference data)"
            raise
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    print(f"Refreshing source datasets ({len(jobs)} files; historical caches reused) ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result())


def add_team_metadata(games: pd.DataFrame) -> pd.DataFrame:
    """Fill conference names using team IDs AND season, respecting realignment."""
    files = sorted(TEAMS_DIR.glob("cfb_teams_*.parquet"))
    if not files:
        print("No cached team metadata; using schedule/summary conferences")
        return games
    frames = []
    for path in files:
        raw = pd.read_parquet(path)
        conference = "conference_name" if "conference_name" in raw else "conference"
        if conference not in raw:
            continue
        frames.append(raw[["season", "team_id", conference]].rename(columns={conference: "_team_conference"}))
    if not frames:
        return games
    teams = pd.concat(frames).drop_duplicates(["season", "team_id"], keep="last")
    out = games.copy()
    for side in ("home", "away"):
        out = out.merge(teams.rename(columns={"team_id": f"{side}_id"}),
                        on=["season", f"{side}_id"], how="left", validate="many_to_one")
        col = f"{side}_conference"
        out[col] = out[col].combine_first(out.pop("_team_conference"))
    return out


def normalize_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """Map sportsdataverse schedule columns onto the processor schema."""
    df = raw.copy()

    rename = {}
    if "start_date" in df.columns and "game_date" not in df.columns:
        rename["start_date"] = "game_date"
    if "home_points" in df.columns and "home_score" not in df.columns:
        rename["home_points"] = "home_score"
    if "away_points" in df.columns and "away_score" not in df.columns:
        rename["away_points"] = "away_score"
    if rename:
        df = df.rename(columns=rename)

    required = [
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
        "home_id",
        "home_team",
        "home_abbreviation",
        "home_score",
        "home_winner",
        "away_id",
        "away_team",
        "away_abbreviation",
        "away_score",
        "away_winner",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Schedule missing columns: {missing}")

    # Prefer the provider's explicit conference flag, then its legacy alias.
    if "conference_game" in df:
        df["conference_competition"] = df["conference_competition"].astype("boolean").fillna(df["conference_game"].astype("boolean"))
    optional = ["home_rank", "away_rank", "home_conference", "away_conference"]
    for col in optional:
        if col not in df:
            df[col] = pd.NA
    out = df[required + optional].copy()
    for col in ["home_rank", "away_rank"]:
        value = pd.to_numeric(out[col], errors="coerce")
        out[col] = value.where(value.between(1, 25))

    # Some canceled games have placeholder 0-0 scores and winner flags.
    # A valid result requires an explicitly final game status.
    final_status = out["status"].astype("string").str.upper().str.strip().isin(
        {"STATUS_FINAL", "FINAL", "STATUS_FINAL_OVERTIME"}
    )
    out = out.loc[final_status].copy()

    # Keep only games with known final scores / winners.
    scored = out["home_score"].notna() & out["away_score"].notna()
    decided = out["home_winner"].notna() & out["away_winner"].notna()
    out = out.loc[scored & decided].copy()

    # Normalize dtypes used downstream.
    out["season"] = out["season"].astype("int64")
    out["week"] = out["week"].astype("int64")
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out["home_winner"] = out["home_winner"].astype(bool)
    out["away_winner"] = out["away_winner"].astype(bool)
    out["neutral_site"] = out["neutral_site"].fillna(0).astype(bool)
    out["conference_competition"] = out["conference_competition"].astype("boolean")
    if out["status"].isna().any():
        out["status"] = out["status"].fillna("STATUS_FINAL")

    return out


def load_all_schedules() -> pd.DataFrame:
    files = sorted(SCHEDULES_DIR.glob("cfb_schedules_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No schedule parquets in {SCHEDULES_DIR}. Run: python process.py --download"
        )
    frames = []
    for path in files:
        raw = pd.read_parquet(path)
        frames.append(normalize_schedule(raw))
        print(f"  schedule {path.name}: {len(frames[-1])} scored games")
    games = pd.concat(frames, ignore_index=True)
    games = games.drop_duplicates(subset=["game_id"], keep="last")
    print(f"Loaded {len(games)} scored schedule rows across {games['season'].nunique()} seasons")
    return games


def load_all_summaries() -> pd.DataFrame:
    files = sorted(SUMMARIES_DIR.glob("cfb_team_summaries_weekly_*.parquet"))
    if not files:
        print(f"No summary parquets in {SUMMARIES_DIR}; continuing without weekly stats")
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    summaries = pd.concat(frames, ignore_index=True)
    print(
        f"Loaded {len(summaries)} weekly summary rows "
        f"across {summaries['season'].nunique()} seasons "
        f"({len(summaries.columns)} cols)"
    )
    return summaries


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
    # Designated home team is not necessarily playing on its home field.
    home["is_home"] = ~games["neutral_site"].astype(bool)

    home["rank"] = games["home_rank"]
    home["opp_rank"] = games["away_rank"]
    home["conference"] = games["home_conference"]
    home["opp_conference"] = games["away_conference"]

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

    away["rank"] = games["away_rank"]
    away["opp_rank"] = games["home_rank"]
    away["conference"] = games["away_conference"]
    away["opp_conference"] = games["home_conference"]
    return pd.concat([home, away], ignore_index=True)


def add_datetime_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Split game_date into year, month, day, and time (US Eastern)."""
    dt = pd.to_datetime(df["game_date"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df.copy()
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["time"] = dt.dt.strftime("%H:%M")
    df["game_datetime"] = dt
    return df


def add_prior_records(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's W-L record entering the game (prior games only, within season)."""
    df = df.sort_values(
        ["season", "team_id", "game_datetime", "game_id"],
        kind="mergesort",
    ).copy()

    # All rows here already have final scores; count every decided game.
    win = df["won"].astype(bool).astype(int)
    loss = (~df["won"].astype(bool)).astype(int)

    group = [df["season"], df["team_id"]]
    df["wins"] = win.groupby(group).cumsum() - win
    df["losses"] = loss.groupby(group).cumsum() - loss
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

    out = out.drop_duplicates(subset=["team_id", "season", "through_week"], keep="last")
    return out


def add_weekly_summaries(df: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each team's (and opponent's) weekly summary as of the prior week.

    Join key: team_id + season + through_week == game.week - 1

    Week 1 and teams missing from the summary file remain null — never filled with zero.
    """
    if summaries.empty:
        return df

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

    # Schedule metadata is preferred; summary values only fill missing cells.
    overlaps = (set(team_sum.columns) | set(opp_sum.columns)) & set(base.columns) - {"team_id", "opponent_id", "season", "_prior_week"}
    saved = {col: base[col].copy() for col in overlaps}
    base = base.drop(columns=list(overlaps))
    merged = base.merge(team_sum, on=["team_id", "season", "_prior_week"], how="left")
    merged = merged.merge(opp_sum, on=["opponent_id", "season", "_prior_week"], how="left")
    merged = merged.drop(columns=["_prior_week"])
    for col, values in saved.items():
        merged[col] = values.reset_index(drop=True).combine_first(merged[col])

    added = [c for c in merged.columns if c not in before_cols]
    marker = "EPAplay_off" if "EPAplay_off" in merged.columns else (added[0] if added else None)
    team_hit = int(merged[marker].notna().sum()) if marker else 0
    print(
        f"Weekly summaries: attached {len(added)} columns; "
        f"{team_hit}/{len(merged)} rows have team prior-week stats "
        f"(week 1 and non-covered teams stay null)"
    )
    return merged


def build_team_games() -> pd.DataFrame:
    print("Loading schedules...")
    games = add_team_metadata(load_all_schedules())
    team_games = expand_to_team_rows(games)
    team_games = add_datetime_parts(team_games)
    team_games = add_prior_records(team_games)

    print("Loading weekly summaries...")
    summaries = load_all_summaries()
    team_games = add_weekly_summaries(team_games, summaries)
    for col in ["conference", "opp_conference"]:
        team_games[col] = team_games[col].map(normalize_conference)
    team_games = add_pregame_context(team_games)

    team_games = team_games.sort_values(
        ["game_datetime", "game_id", "is_home"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

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
    return team_games[core_columns + extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build multi-season CFB team-game dataset")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/refresh schedule + summary parquets from sportsdataverse",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even when local files already exist",
    )
    parser.add_argument("--offline", action="store_true", help="Build only from cached source files; no downloads")
    args = parser.parse_args(argv)
    if args.offline and (args.download or args.force_download):
        parser.error("--offline cannot be combined with download flags")
    if not args.offline:
        download_season_files(force=args.force_download)

    out = build_team_games()
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        out.to_parquet(temporary, index=False)
        os.replace(temporary, OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Wrote {len(out)} rows x {len(out.columns)} cols to {OUTPUT_PATH.name} "
        f"(seasons {int(out['season'].min())}-{int(out['season'].max())})"
    )
    preview_cols = [
        "season",
        "week",
        "team",
        "opponent",
        "wins",
        "losses",
        "conference",
        "EPAplay_off",
        "opp_EPAplay_off",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    sample = out.loc[(out["week"] >= 2) & (out["season"] >= 2024), preview_cols].head(8)
    print(sample.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
