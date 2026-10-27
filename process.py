"""Process CFB schedule + weekly summary parquets into team-game rows.

Source data (sportsdataverse / cfbfastR):
  https://github.com/sportsdataverse/cfbfastR-cfb-data/tree/main/cfb/cfb_schedules/parquet
  https://github.com/sportsdataverse/cfbfastR-cfb-data/tree/main/cfb/cfb_team_summaries_weekly/parquet
"""

from __future__ import annotations

import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEDULES_DIR = DATA_DIR / "schedules"
SUMMARIES_DIR = DATA_DIR / "summaries"
OUTPUT_PATH = ROOT / "cfb_team_games.parquet"

GITHUB_RAW = "https://github.com/sportsdataverse/cfbfastR-cfb-data/raw/main/cfb"
SCHEDULE_YEARS = range(2001, 2027)
SUMMARY_YEARS = range(2004, 2027)

# Identity / join keys from weekly summaries — not copied onto game rows as-is.
_SUMMARY_KEY_COLS = ("team_id", "pos_team", "season", "through_week")


def download_season_files(force: bool = False, workers: int = 6) -> None:
    """Download schedule + weekly summary parquets into data/."""
    jobs: list[tuple[Path, str]] = []
    for year in SCHEDULE_YEARS:
        name = f"cfb_schedules_{year}.parquet"
        dest = SCHEDULES_DIR / name
        remote = f"{GITHUB_RAW}/cfb_schedules/parquet/{name}"
        jobs.append((dest, remote))
    for year in SUMMARY_YEARS:
        name = f"cfb_team_summaries_weekly_{year}.parquet"
        dest = SUMMARIES_DIR / name
        remote = f"{GITHUB_RAW}/cfb_team_summaries_weekly/parquet/{name}"
        jobs.append((dest, remote))

    SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(dest: Path, url: str) -> str:
        if dest.exists() and dest.stat().st_size > 1000 and not force:
            return f"skip {dest.name}"
        urllib.request.urlretrieve(url, dest)
        return f"ok   {dest.name} ({dest.stat().st_size / 1e6:.2f} MB)"

    print(f"Downloading {len(jobs)} files into {DATA_DIR} ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fetch, dest, url) for dest, url in jobs]
        for fut in as_completed(futs):
            print(fut.result())


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

    out = df[required].copy()

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
    out["conference_competition"] = out["conference_competition"].fillna(0).astype(bool)
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


def build_team_games() -> pd.DataFrame:
    print("Loading schedules...")
    games = load_all_schedules()
    team_games = expand_to_team_rows(games)
    team_games = add_datetime_parts(team_games)
    team_games = add_prior_records(team_games)

    print("Loading weekly summaries...")
    summaries = load_all_summaries()
    team_games = add_weekly_summaries(team_games, summaries)

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
    args = parser.parse_args(argv)

    if args.download or args.force_download:
        download_season_files(force=args.force_download)

    out = build_team_games()
    out.to_parquet(OUTPUT_PATH, index=False)
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
