"""Matchup resolution and public discover() API for college football."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from discovery.config import EngineConfig
from discovery.engine.models import Discovery, FeatureSpec, SearchStats
from discovery.engine.search import DiscoveryEngine
from discovery.engine.validation import apply_conditions, build_matching_games, recount_record
from discovery.sports.college_football.dates import before_as_of_mask, parse_as_of, parse_game_dates
from discovery.sports.college_football.schedule import resolve_scheduled_game
from discovery.sports.college_football.features import (
    LEAKAGE_COLUMNS,
    MONTH_NAMES,
    assert_no_leakage,
    enrich_team_games,
    feature_specs,
    searchable_feature_names,
)
from discovery.sports.college_football.formatting import describe_discovery, format_report
from discovery.engine.streaks import recent_favorable_streak


@dataclass
class DiscoverResult:
    """Structured output from a discovery run."""

    team: str
    opponent: str
    game_date: str
    home: bool
    discoveries: list[Discovery]
    stats: SearchStats
    team_features: dict[str, Any]
    opponent_features: dict[str, Any]
    report: str

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.report, encoding="utf-8")
        return path


_DATA_CACHE: dict[str, pd.DataFrame] = {}


def load_observations(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load and enrich the processed team-games parquet (cached by path)."""
    key = str(Path(data_path).resolve())
    if key not in _DATA_CACHE:
        raw = pd.read_parquet(data_path)
        _DATA_CACHE[key] = enrich_team_games(raw)
    return _DATA_CACHE[key]


def clear_data_cache() -> None:
    _DATA_CACHE.clear()


def resolve_team_id(df: pd.DataFrame, query: str) -> int:
    """Resolve a user-facing team string to a stable team_id."""
    q = query.strip().lower()
    teams = df[["team_id", "team"]].dropna().drop_duplicates()
    names = teams["team"].astype(str)

    exact = teams.loc[names.str.lower() == q]
    pool = exact
    if pool.empty:
        starts = teams.loc[names.str.lower().str.startswith(q)]
        pool = starts
    if pool.empty:
        contains = teams.loc[names.str.lower().str.contains(q, regex=False)]
        pool = contains
    if pool.empty:
        raise ValueError(f"Could not resolve team name {query!r}.")

    # Prefer the team_id with the most games, then the most recent display name.
    counts = df.groupby("team_id").size()
    ranked = (
        pool.assign(n=pool["team_id"].map(counts))
        .sort_values(["n", "team"], ascending=[False, True])
    )
    return int(ranked.iloc[0]["team_id"])


def resolve_team_name(df: pd.DataFrame, query: str) -> str:
    """Resolve a user-facing team string to the latest display name for that team_id."""
    team_id = resolve_team_id(df, query)
    return display_name_for_team_id(df, team_id)


def display_name_for_team_id(df: pd.DataFrame, team_id: int) -> str:
    sub = df.loc[df["team_id"] == team_id, ["team", "game_date"]].copy()
    if sub.empty:
        raise ValueError(f"No rows for team_id={team_id}")
    sub["_dt"] = parse_game_dates(sub["game_date"])
    sub = sub.dropna(subset=["_dt"]).sort_values("_dt", kind="mergesort")
    if sub.empty:
        raise ValueError(f"No dated rows for team_id={team_id}")
    return str(sub.iloc[-1]["team"])


def _team_state_after_latest(
    df: pd.DataFrame,
    team_id: int,
    as_of: Optional[pd.Timestamp] = None,
) -> dict[str, Any]:
    """Pregame features for a team after its most recent completed game before as_of."""
    sub = df[df["team_id"] == team_id].copy()
    if sub.empty:
        raise ValueError(f"No historical rows for team_id={team_id}")
    sub["_dt"] = parse_game_dates(sub["game_date"])
    sub = sub.loc[before_as_of_mask(sub["game_date"], as_of)].copy()
    sub = sub.dropna(subset=["_dt"]).sort_values(["_dt", "game_id"], kind="mergesort")
    if sub.empty:
        raise ValueError(f"No historical rows for team_id={team_id} before {as_of}")
    last = sub.iloc[-1]
    team = str(last["team"])

    # After the last completed game, the next entering record is last.wins/losses + result
    wins = int(last["wins"]) + (1 if bool(last["won"]) else 0)
    losses = int(last["losses"]) + (0 if bool(last["won"]) else 1)
    games_played = wins + losses
    if bool(last["won"]):
        win_streak = int(last["win_streak"]) + 1
        loss_streak = 0
    else:
        win_streak = 0
        loss_streak = int(last["loss_streak"]) + 1

    # Records and entering streaks are season-specific. The latest completed
    # game can be from the previous season for an early-season matchup.
    if as_of is not None and "season" in last.index and int(last["season"]) < as_of.year:
        wins = losses = games_played = win_streak = loss_streak = 0

    win_pct = wins / games_played if games_played else float("nan")
    state: dict[str, Any] = {
        "wins": wins,
        "losses": losses,
        "games_played": games_played,
        "win_pct": win_pct,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "has_winning_record": wins > losses,
        "has_losing_record": losses > wins,
        "team": team,
        "team_id": int(last["team_id"]),
    }

    # Carry forward advanced / goofy fields from the latest available snapshot.
    # Summary cols on the last game row are prior-week; still the best in-row estimate
    # until the next week's prior join exists.
    wanted = searchable_feature_names()
    for col in wanted:
        if col.startswith("opp_") or col in state:
            continue
        if col in last.index and not pd.isna(last[col]):
            state[col] = last[col]

    # Prefer latest non-null conference/division/EPA-style fields from this season.
    season = int(last["season"]) if "season" in last.index and not pd.isna(last["season"]) else None
    if season is not None:
        season_rows = sub[sub["season"] == season]
        for col in (
            "conference",
            "division",
            "EPAplay_off",
            "EPAplay_def",
            "EPAplay_margin",
            "success_off",
            "success_def",
            "havoc_def",
            "explosive_off",
            "yardsplay_off",
            "EPAplay_off_rank",
            "EPAplay_def_rank",
        ):
            if col not in season_rows.columns:
                continue
            series = season_rows[col].dropna()
            if not series.empty:
                state[col] = series.iloc[-1]

    # Expected attendance / home venue for goofy home-game patterns.
    home_games = sub[sub["is_home"] == True] if "is_home" in sub.columns else sub.iloc[0:0]
    if not home_games.empty:
        if "venue" in home_games.columns:
            venues = home_games["venue"].dropna()
            if not venues.empty:
                state["venue"] = venues.iloc[-1]
        if "attendance" in home_games.columns:
            atts = pd.to_numeric(home_games["attendance"], errors="coerce")
            atts = atts[atts > 0]
            if not atts.empty:
                state["attendance"] = float(atts.tail(5).median())

    return state


def build_matchup_features(
    df: pd.DataFrame,
    *,
    team: str,
    opponent: str,
    team_id: int,
    opponent_id: int,
    game_date: str,
    home: bool,
    overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build pregame feature vectors for both perspectives of the upcoming game.

    If an exact row already exists in the dataset for this matchup/date, use it.
    Otherwise synthesize features from each team's latest completed state.
    """
    overrides = dict(overrides or {})
    dt = parse_as_of(game_date)
    month = int(dt.month) if dt is not None and not pd.isna(dt) else None
    month_name = MONTH_NAMES.get(month) if month else None
    day_of_week = dt.day_name() if dt is not None and not pd.isna(dt) else None

    # Try to find an existing scheduled/played row for this matchup
    if dt is not None and not pd.isna(dt):
        same_day = parse_game_dates(df["game_date"]).dt.strftime("%Y-%m-%d") == dt.strftime("%Y-%m-%d")
        existing = df[(df["team_id"] == team_id) & (df["opponent_id"] == opponent_id) & same_day]
    else:
        existing = df.iloc[0:0]
    if not existing.empty:
        row = existing.iloc[0].to_dict()
        team_feats = _row_to_features(row)
        opp_row = df[(df["game_id"] == row["game_id"]) & (df["team_id"] == opponent_id)]
        opponent_feats = _row_to_features(opp_row.iloc[0].to_dict()) if not opp_row.empty else {}
    else:
        # Only use games strictly before this matchup (backtest-safe).
        team_state = _team_state_after_latest(df, team_id, as_of=dt)
        opp_state = _team_state_after_latest(df, opponent_id, as_of=dt)

        team_feats = {
            **team_state,
            "is_home": bool(home),
            "neutral_site": False,
            "conference_competition": None,  # unknown unless overridden
            "month": month,
            "month_name": month_name,
            "day_of_week": day_of_week,
            "week": None,
            "opp_wins": opp_state["wins"],
            "opp_losses": opp_state["losses"],
            "opp_games_played": opp_state["games_played"],
            "opp_win_pct": opp_state["win_pct"],
            "opp_win_streak": opp_state["win_streak"],
            "opp_loss_streak": opp_state["loss_streak"],
            "opp_has_winning_record": opp_state["has_winning_record"],
            "opp_has_losing_record": opp_state["has_losing_record"],
            "win_diff": team_state["wins"] - opp_state["wins"],
            "opponent": opponent,
        }
        # Mirror opponent advanced snapshot onto opp_* keys for the selected team view.
        for key, value in opp_state.items():
            if key in {
                "wins",
                "losses",
                "games_played",
                "win_pct",
                "win_streak",
                "loss_streak",
                "has_winning_record",
                "has_losing_record",
                "team",
                "team_id",
                "venue",
                "attendance",
            }:
                continue
            team_feats[f"opp_{key}"] = value

        if not home:
            # Road game: venue/attendance should reflect the opponent's home spot when known.
            if "venue" in opp_state:
                team_feats["venue"] = opp_state["venue"]
            if "attendance" in opp_state:
                team_feats["attendance"] = opp_state["attendance"]

        opponent_feats = {
            **opp_state,
            "is_home": not bool(home),
            "neutral_site": False,
            "conference_competition": None,
            "month": month,
            "month_name": month_name,
            "day_of_week": day_of_week,
            "week": None,
            "opp_wins": team_state["wins"],
            "opp_losses": team_state["losses"],
            "opp_games_played": team_state["games_played"],
            "opp_win_pct": team_state["win_pct"],
            "opp_win_streak": team_state["win_streak"],
            "opp_loss_streak": team_state["loss_streak"],
            "opp_has_winning_record": team_state["has_winning_record"],
            "opp_has_losing_record": team_state["has_losing_record"],
            "win_diff": opp_state["wins"] - team_state["wins"],
            "opponent": team,
        }
        for key, value in team_state.items():
            if key in {
                "wins",
                "losses",
                "games_played",
                "win_pct",
                "win_streak",
                "loss_streak",
                "has_winning_record",
                "has_losing_record",
                "team",
                "team_id",
                "venue",
                "attendance",
            }:
                continue
            opponent_feats[f"opp_{key}"] = value

    # Apply explicit overrides to the selected team's feature vector (and mirrored keys)
    for key, value in overrides.items():
        team_feats[key] = value
        # Common mirrors
        if key == "is_home":
            opponent_feats["is_home"] = not bool(value)
        elif key.startswith("opp_"):
            bare = key[len("opp_") :]
            opponent_feats[bare] = value
        elif key in {
            "wins",
            "losses",
            "games_played",
            "win_pct",
            "win_streak",
            "loss_streak",
            "has_winning_record",
            "has_losing_record",
        }:
            opponent_feats[f"opp_{key}" if key != "has_winning_record" and key != "has_losing_record" else f"opp_{key}"] = value
            if key in ("wins", "losses", "games_played", "win_pct", "win_streak", "loss_streak",
                       "has_winning_record", "has_losing_record"):
                opponent_feats[f"opp_{key}"] = value

    return team_feats, opponent_feats


def _row_to_features(row: Mapping[str, Any]) -> dict[str, Any]:
    specs = searchable_feature_names()
    return {k: row.get(k) for k in specs if k in row}


def _prepare_scope_frames(
    df: pd.DataFrame,
    *,
    team_id: int,
    opponent_id: int,
    game_date: str,
    scopes: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Filter historical populations and attach a scope-specific favorable target."""
    hist = df.copy()
    # Backtest-safe: only games strictly before the matchup kickoff / as-of date.
    as_of = parse_as_of(game_date)
    if as_of is not None:
        hist = hist.loc[before_as_of_mask(hist["game_date"], as_of)].copy()

    frames: dict[str, pd.DataFrame] = {}
    if "all" in scopes or "both" in scopes:
        all_df = hist.copy()
        all_df["favorable"] = all_df["won"].astype(bool)
        frames["all"] = all_df

    if "team" in scopes or "both" in scopes:
        team_df = hist[hist["team_id"] == team_id].copy()
        team_df["favorable"] = team_df["won"].astype(bool)
        frames["team"] = team_df

    if "opponent" in scopes or "both" in scopes:
        opp_df = hist[hist["team_id"] == opponent_id].copy()
        # Favorable for our narrative = opponent lost
        opp_df["favorable"] = ~opp_df["won"].astype(bool)
        frames["opponent"] = opp_df

    if "h2h" in scopes:
        h2h = hist[(hist["team_id"] == team_id) & (hist["opponent_id"] == opponent_id)].copy()
        h2h["favorable"] = h2h["won"].astype(bool)
        frames["h2h"] = h2h

    return frames


def discover(
    team: str,
    opponent: str,
    game_date: str,
    home: Optional[bool] = None,
    *,
    top_n: int = 20,
    scopes: Optional[Sequence[str]] = None,
    min_sample_size: Optional[int] = None,
    max_conditions: Optional[int] = None,
    beam_width: Optional[int] = None,
    max_depth: Optional[int] = None,
    feature_overrides: Optional[Mapping[str, Any]] = None,
    data_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[EngineConfig] = None,
    print_report: bool = True,
) -> DiscoverResult:
    """
    Discover entertaining historical statistics for an upcoming CFB matchup.

    Parameters
    ----------
    team, opponent:
        Team names (canonical or partial, e.g. "Michigan State").
    game_date:
        Upcoming game date (YYYY-MM-DD or parseable timestamp).
    home:
        Whether `team` is at home.
    feature_overrides:
        Explicit pregame feature values when current-game info is unavailable.
    """
    cfg = config or EngineConfig()
    path = Path(data_path) if data_path else cfg.data_path
    df = load_observations(path)

    team_id = resolve_team_id(df, team)
    opponent_id = resolve_team_id(df, opponent)
    team_name = display_name_for_team_id(df, team_id)
    opp_name = display_name_for_team_id(df, opponent_id)
    home = resolve_scheduled_game(team_id, opponent_id, game_date, home=home)

    search_cfg = replace(cfg.search)
    if scopes is not None:
        search_cfg.scopes = tuple(scopes)
    if min_sample_size is not None:
        search_cfg.min_sample_size = min_sample_size
    if max_conditions is not None:
        search_cfg.max_conditions = max_conditions
        search_cfg.max_depth = max_conditions
    if beam_width is not None:
        search_cfg.beam_width = beam_width
    if max_depth is not None:
        search_cfg.max_depth = max_depth

    team_feats, opp_feats = build_matchup_features(
        df,
        team=team_name,
        opponent=opp_name,
        team_id=team_id,
        opponent_id=opponent_id,
        game_date=game_date,
        home=home,
        overrides=feature_overrides,
    )

    features = [f for f in feature_specs() if f.name not in LEAKAGE_COLUMNS]
    assert_no_leakage([f.name for f in features])

    frames = _prepare_scope_frames(
        df,
        team_id=team_id,
        opponent_id=opponent_id,
        game_date=game_date,
        scopes=search_cfg.scopes,
    )

    scope_jobs: list[tuple[str, pd.DataFrame, Mapping[str, Any], str, str]] = []
    for scope_name in search_cfg.scopes:
        if scope_name == "both":
            # Expand into team + opponent if not already present
            continue
        if scope_name not in frames:
            continue
        if scope_name == "opponent":
            scope_jobs.append((scope_name, frames[scope_name], opp_feats, "losses", "favorable"))
        else:
            current = team_feats if scope_name in {"team", "all", "h2h"} else team_feats
            scope_jobs.append((scope_name, frames[scope_name], current, "wins", "favorable"))

    if "both" in search_cfg.scopes:
        if "team" in frames:
            scope_jobs.append(("team", frames["team"], team_feats, "wins", "favorable"))
        if "opponent" in frames:
            scope_jobs.append(("opponent", frames["opponent"], opp_feats, "losses", "favorable"))

    def _describe(scope: str, conditions, disc: Discovery) -> str:
        return describe_discovery(disc, team=team_name, opponent=opp_name)

    engine = DiscoveryEngine(
        config=search_cfg,
        weights=cfg.ranking,
        describe=_describe,
    )
    discoveries, stats = engine.run_many(
        scope_jobs,
        features,
        date_col="game_date",
        top_n=top_n,
    )

    # Final guard: keep only discoveries whose conditions match current form,
    # then attach historical evidence for the survivors only.
    from discovery.engine.conditions import conditions_apply_to_current

    filtered: list[Discovery] = []
    for disc in discoveries:
        current = opp_feats if disc.scope == "opponent" else team_feats
        if not conditions_apply_to_current(current, disc.conditions):
            continue
        frame = frames.get(disc.scope)
        if frame is not None:
            mask = apply_conditions(frame, disc.conditions)
            wins, losses, win_pct = recount_record(frame, mask, target_col="favorable")
            disc.wins = wins
            disc.losses = losses
            disc.n_games = wins + losses
            disc.win_pct = win_pct
            disc.streak = recent_favorable_streak(frame, mask, target_col="favorable", date_col="game_date")
            disc.matching_games = build_matching_games(
                frame, mask, date_col="game_date", target_col="favorable", won_col="won"
            )
        if not disc.description:
            disc.description = describe_discovery(disc, team=team_name, opponent=opp_name)
        filtered.append(disc)
    discoveries = filtered
    stats.discoveries_after_dedup = len(discoveries)

    report = format_report(
        team=team_name,
        opponent=opp_name,
        game_date=game_date,
        home=home,
        discoveries=discoveries,
        stats=stats,
        team_features=team_feats,
        opponent_features=opp_feats,
    )
    result = DiscoverResult(
        team=team_name,
        opponent=opp_name,
        game_date=game_date,
        home=home,
        discoveries=discoveries,
        stats=stats,
        team_features=team_feats,
        opponent_features=opp_feats,
        report=report,
    )
    if print_report:
        print(report)
    if output_path:
        result.save(output_path)
    return result
