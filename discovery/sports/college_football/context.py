"""Pregame context shared by historical processing and upcoming matchups."""
from __future__ import annotations

import pandas as pd

from discovery.sports.college_football.dates import parse_game_dates

CONTEXT_FEATURES = [
    ("previous_points_for", "numeric", "Points scored in previous game"),
    ("previous_points_against", "numeric", "Points allowed in previous game"),
    ("previous_margin", "numeric", "Previous-game scoring margin"),
    ("previous_won", "boolean", "Won previous game"),
    ("previous_conference_game", "boolean", "Previous game was a conference game"),
    ("previous_opponent_conference", "categorical", "Previous opponent conference"),
    ("rest_days", "numeric", "Full rest days between games"),
    ("after_bye", "boolean", "At least 12 days between games"),
    ("conference_wins", "numeric", "Pregame conference wins"),
    ("conference_losses", "numeric", "Pregame conference losses"),
    ("rank", "numeric", "Pregame poll rank"),
    ("conference", "categorical", "Conference"),
]


CONFERENCE_ALIASES = {
    "Big Ten Conference": "Big Ten", "Big 12 Conference": "Big 12",
    "Southeastern Conference": "SEC", "Atlantic Coast Conference": "ACC",
    "Mountain West Conference": "Mountain West", "American Athletic Conference": "American",
    "Mid-American Conference": "MAC", "Sun Belt Conference": "Sun Belt",
    "Conference USA": "CUSA", "Conference USA Conference": "CUSA",
    "Pac-12 Conference": "Pac-12",
}


def normalize_conference(value):
    return CONFERENCE_ALIASES.get(value, value) if pd.notna(value) else pd.NA


def schedule_metadata(row, *, home: bool) -> dict:
    """One perspective of a raw schedule; unknown ranks stay unknown."""
    side, other = ("home", "away") if home else ("away", "home")
    def rank(name):
        value = pd.to_numeric(row.get(f"{name}_rank"), errors="coerce")
        return value if pd.notna(value) and 1 <= value <= 25 else float("nan")
    conference = row.get("conference_competition")
    if pd.isna(conference):
        conference = row.get("conference_game", pd.NA)
    return {
        "rank": rank(side), "opp_rank": rank(other),
        "conference": normalize_conference(row.get(f"{side}_conference", pd.NA)),
        "opp_conference": normalize_conference(row.get(f"{other}_conference", pd.NA)),
        "conference_competition": conference,
        "neutral_site": row.get("neutral_site", pd.NA),
        "week": row.get("week"), "season": row.get("season"),
        "season_type": row.get("season_type"),
    }


def add_pregame_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Shift completed results within season; never use the current outcome.

    rest_days excludes both game dates: Saturday to Saturday is six full days.
    after_bye means a gap of at least twelve calendar days, not an offseason.
    """
    out = frame.copy()
    out["_context_date"] = parse_game_dates(out["game_date"])
    out = out.sort_values(["season", "team_id", "_context_date", "game_id"], kind="mergesort")
    group = [out["season"], out["team_id"]]
    for source, dest in [
        ("team_score", "previous_points_for"), ("opponent_score", "previous_points_against"),
        ("won", "previous_won"), ("conference_competition", "previous_conference_game"),
        ("opp_conference", "previous_opponent_conference"),
    ]:
        values = out[source] if source in out else pd.Series(pd.NA, index=out.index)
        out[dest] = values.groupby(group).shift()
    out["previous_margin"] = out["previous_points_for"] - out["previous_points_against"]
    days = out["_context_date"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    gap = (days - days.groupby(group).shift()).dt.days
    out["rest_days"] = (gap - 1).where(gap > 0)
    out["after_bye"] = gap.ge(12).astype("boolean").where(gap.notna())
    conference = out.get("conference_competition", pd.Series(pd.NA, index=out.index)).astype("boolean")
    # An unknown conference flag makes subsequent conference totals unknown.
    unknown = conference.isna().astype(int).groupby(group).cumsum() - conference.isna().astype(int)
    for outcome, dest in [(True, "conference_wins"), (False, "conference_losses")]:
        result = (conference.fillna(False) & out["won"].eq(outcome).fillna(False)).astype(int)
        out[dest] = (result.groupby(group).cumsum() - result).astype("Int64").where(unknown.eq(0))
    for col in ["previous_won", "previous_conference_game"]:
        out[col] = out[col].astype("boolean")
    # Mirror only derived fields; schedule ranks/conferences already have both perspectives.
    cols = [name for name, *_ in CONTEXT_FEATURES if name not in {"rank", "conference"}]
    stale = [f"opp_{c}" for c in cols if f"opp_{c}" in out]
    out = out.drop(columns=stale)
    sibling = out[["game_id", "team_id", *cols]].rename(columns={
        "team_id": "opponent_id", **{c: f"opp_{c}" for c in cols},
    })
    return out.drop(columns="_context_date").merge(sibling, on=["game_id", "opponent_id"], how="left", validate="one_to_one")


def upcoming_context(history: pd.DataFrame, *, team_id: int, opponent_id: int,
                     game_date, season: int, metadata: dict, opponent_metadata: dict) -> tuple[dict, dict]:
    """Run the same derivation on history plus two outcome-free matchup rows."""
    cutoff = parse_game_dates(game_date)
    dates = parse_game_dates(history["game_date"])
    hist = history.loc[(dates < cutoff) & history["season"].eq(season)].copy()
    # Work on narrow source columns rather than copying 1,600 summary fields.
    sources = ["game_id", "season", "team_id", "opponent_id", "game_date", "team_score",
               "opponent_score", "won", "conference_competition", "opp_conference"]
    hist = hist.reindex(columns=sources)
    rows = []
    # Negative ID is local to this calculation and never saved as a real game.
    for tid, oid, meta in [(team_id, opponent_id, metadata), (opponent_id, team_id, opponent_metadata)]:
        rows.append({"game_id": -1, "season": season, "team_id": tid, "opponent_id": oid,
                     "game_date": game_date, "team_score": float("nan"), "opponent_score": float("nan"),
                     "won": pd.NA, "conference_competition": meta.get("conference_competition", pd.NA),
                     "opp_conference": meta.get("opp_conference", pd.NA)})
    combined = add_pregame_context(pd.concat([hist, pd.DataFrame(rows)], ignore_index=True))
    derived = [name for name, *_ in CONTEXT_FEATURES if name not in {"rank", "conference"}]
    keys = derived + [f"opp_{name}" for name in derived]
    results = [combined.loc[combined["game_id"].eq(-1) & combined["team_id"].eq(tid), keys].iloc[0].to_dict()
               for tid in (team_id, opponent_id)]
    for tid, meta, values in [(team_id, metadata, results[0]), (opponent_id, opponent_metadata, results[1])]:
        completed = parse_game_dates(hist.loc[hist["team_id"].eq(tid), "game_date"])
        latest = completed.max()
        pending = [(date, conf) for date, conf in meta.get("_prior_games", [])
                   if pd.isna(latest) or parse_game_dates(date) > latest]
        if not pending:
            continue
        for name in derived:
            if name.startswith("previous_"):
                values[name] = pd.NA
        previous = max(parse_game_dates(date) for date, _ in pending)
        def eastern_day(timestamp):
            return timestamp.tz_convert("America/New_York").date()
        gap = (eastern_day(cutoff) - eastern_day(previous)).days
        values["rest_days"] = gap - 1 if gap > 0 else pd.NA
        values["after_bye"] = gap >= 12 if gap > 0 else pd.NA
        if any(pd.isna(conf) or bool(conf) for _, conf in pending):
            values["conference_wins"] = values["conference_losses"] = pd.NA
    for values, other in [(results[0], results[1]), (results[1], results[0])]:
        for name in derived:
            values[f"opp_{name}"] = other[name]
    return tuple(results)
