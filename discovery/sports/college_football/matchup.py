"""Matchup resolution and public discover() API for college football."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from discovery.config import EngineConfig
from discovery.engine.models import Discovery, FeatureSpec, SearchStats
from discovery.engine.search import DiscoveryEngine
from discovery.sports.college_football.features import (
    LEAKAGE_COLUMNS,
    MONTH_NAMES,
    assert_no_leakage,
    enrich_team_games,
    feature_specs,
)
from discovery.sports.college_football.formatting import describe_discovery, format_report


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


def resolve_team_name(df: pd.DataFrame, query: str) -> str:
    """Resolve a user-facing team string to the canonical dataset name."""
    names = sorted(df["team"].dropna().unique().tolist())
    q = query.strip().lower()
    exact = [n for n in names if n.lower() == q]
    if exact:
        return exact[0]
    starts = [n for n in names if n.lower().startswith(q)]
    if len(starts) == 1:
        return starts[0]
    contains = [n for n in names if q in n.lower()]
    if len(contains) == 1:
        return contains[0]
    # Prefer FBS-looking shortest contains match when several share a prefix
    if starts:
        return sorted(starts, key=len)[0]
    if contains:
        return sorted(contains, key=len)[0]
    raise ValueError(f"Could not resolve team name {query!r}. Examples: {names[:5]}")


def _team_state_after_latest(df: pd.DataFrame, team: str) -> dict[str, Any]:
    """Current pregame features for a team as of after its most recent completed game."""
    sub = df[df["team"] == team].copy()
    if sub.empty:
        raise ValueError(f"No historical rows for team {team!r}")
    sub["_dt"] = pd.to_datetime(sub["game_date"], utc=True, errors="coerce")
    sub = sub.sort_values(["_dt", "game_id"], kind="mergesort")
    last = sub.iloc[-1]

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

    win_pct = wins / games_played if games_played else float("nan")
    return {
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


def build_matchup_features(
    df: pd.DataFrame,
    *,
    team: str,
    opponent: str,
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
    dt = pd.to_datetime(game_date, utc=True, errors="coerce")
    month = int(dt.month) if not pd.isna(dt) else None
    month_name = MONTH_NAMES.get(month) if month else None
    day_of_week = dt.day_name() if not pd.isna(dt) else None

    # Try to find an existing scheduled/played row for this matchup
    existing = df[
        (df["team"] == team)
        & (df["opponent"] == opponent)
        & (pd.to_datetime(df["game_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
           == (dt.strftime("%Y-%m-%d") if not pd.isna(dt) else ""))
    ]
    if not existing.empty:
        row = existing.iloc[0].to_dict()
        team_feats = _row_to_features(row)
        opp_row = df[(df["game_id"] == row["game_id"]) & (df["team"] == opponent)]
        opponent_feats = _row_to_features(opp_row.iloc[0].to_dict()) if not opp_row.empty else {}
    else:
        team_state = _team_state_after_latest(df, team)
        opp_state = _team_state_after_latest(df, opponent)

        team_feats = {
            **team_state,
            "is_home": bool(home),
            "neutral_site": False,
            "conference_competition": None,  # unknown unless overridden
            "month": month,
            "month_name": month_name,
            "day_of_week": day_of_week,
            "week": None,
            "season_type": 2,
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
        opponent_feats = {
            **opp_state,
            "is_home": not bool(home),
            "neutral_site": False,
            "conference_competition": None,
            "month": month,
            "month_name": month_name,
            "day_of_week": day_of_week,
            "week": None,
            "season_type": 2,
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
    specs = {f.name for f in feature_specs()}
    return {k: row.get(k) for k in specs if k in row}


def _prepare_scope_frames(
    df: pd.DataFrame,
    *,
    team: str,
    opponent: str,
    game_date: str,
    scopes: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Filter historical populations and attach a scope-specific favorable target."""
    hist = df.copy()
    # Exclude the upcoming game itself if present (no leakage / circular evidence)
    gd = pd.to_datetime(game_date, utc=True, errors="coerce")
    if not pd.isna(gd):
        same_day = pd.to_datetime(hist["game_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d") == gd.strftime(
            "%Y-%m-%d"
        )
        upcoming_mask = same_day & (
            ((hist["team"] == team) & (hist["opponent"] == opponent))
            | ((hist["team"] == opponent) & (hist["opponent"] == team))
        )
        hist = hist.loc[~upcoming_mask].copy()

    frames: dict[str, pd.DataFrame] = {}
    if "all" in scopes or "both" in scopes:
        all_df = hist.copy()
        all_df["favorable"] = all_df["won"].astype(bool)
        frames["all"] = all_df

    if "team" in scopes or "both" in scopes:
        team_df = hist[hist["team"] == team].copy()
        team_df["favorable"] = team_df["won"].astype(bool)
        frames["team"] = team_df

    if "opponent" in scopes or "both" in scopes:
        opp_df = hist[hist["team"] == opponent].copy()
        # Favorable for our narrative = opponent lost
        opp_df["favorable"] = ~opp_df["won"].astype(bool)
        frames["opponent"] = opp_df

    if "h2h" in scopes:
        h2h = hist[(hist["team"] == team) & (hist["opponent"] == opponent)].copy()
        h2h["favorable"] = h2h["won"].astype(bool)
        frames["h2h"] = h2h

    return frames


def discover(
    team: str,
    opponent: str,
    game_date: str,
    home: bool = True,
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

    team_name = resolve_team_name(df, team)
    opp_name = resolve_team_name(df, opponent)

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
        game_date=game_date,
        home=home,
        overrides=feature_overrides,
    )

    features = [f for f in feature_specs() if f.name not in LEAKAGE_COLUMNS]
    assert_no_leakage([f.name for f in features])

    frames = _prepare_scope_frames(
        df,
        team=team_name,
        opponent=opp_name,
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

    # Ensure every discovery has a description
    for disc in discoveries:
        if not disc.description:
            disc.description = describe_discovery(disc, team=team_name, opponent=opp_name)

    report = format_report(
        team=team_name,
        opponent=opp_name,
        game_date=game_date,
        home=home,
        discoveries=discoveries,
        stats=stats,
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
