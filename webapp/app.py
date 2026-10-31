"""Read-only FastAPI UI for scheduled Irrational Fan matchups."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from discovery.sports.college_football.dates import parse_game_dates
from discovery.sports.college_football.matchup import discover
from discovery.sports.college_football.schedule import SCHEDULE_DIR

EASTERN = ZoneInfo("America/New_York")
PAGE = Path(__file__).with_name("index.html")
app = FastAPI(title="Irrational Fan", version="0.1.0")

# Keep Uvicorn at one worker so this guard covers every request.
_DISCOVERY_LOCK = asyncio.Lock()
_REPORT_CACHE: OrderedDict[tuple[str, int], dict] = OrderedDict()
_CACHE_LIMIT = 32



class MatchupRequest(BaseModel):
    game_id: str = Field(min_length=1, max_length=100)
    team_id: int | None = None


def _schedule_files(now: datetime) -> list[Path]:
    # January bowls can belong to the previous season.
    return [SCHEDULE_DIR / f"cfb_schedules_{year}.parquet"
            for year in (now.year - 1, now.year, now.year + 1)]


def _read_games(now: datetime) -> list[dict]:
    frames = []
    for path in _schedule_files(now):
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No cached schedules found. Run python process.py first.")
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last").reset_index(drop=True)
    date_col = "game_date" if "game_date" in raw else "start_date"
    required = {"game_id", "home_id", "away_id", "home_team", "away_team", "week", "season", date_col}
    if not required.issubset(raw.columns):
        raise ValueError(f"Schedule missing columns: {sorted(required - set(raw.columns))}")
    dates = parse_game_dates(raw[date_col]).dt.tz_convert(EASTERN)
    today = now.astimezone(EASTERN).date()
    rows = []
    for i, row in raw.iterrows():
        dt = dates.loc[i]
        if pd.isna(dt) or dt.date() < today:
            continue
        status = str(row.get("status", "")).upper()
        if any(word in status for word in ("FINAL", "CANCEL", "POSTPON")):
            continue
        score_home = row.get("home_score", row.get("home_points"))
        score_away = row.get("away_score", row.get("away_points"))
        if pd.notna(score_home) and pd.notna(score_away) and "IN_PROGRESS" not in status:
            continue
        rows.append({
            "game_id": str(row["game_id"]),
            "home_id": int(row["home_id"]), "away_id": int(row["away_id"]),
            "home_team": str(row["home_team"]), "away_team": str(row["away_team"]),
            "date": dt.date().isoformat(), "kickoff": dt.isoformat(),
            "week": int(row["week"]), "season": int(row["season"]),
            "neutral_site": bool(row["neutral_site"]) if pd.notna(row.get("neutral_site")) else False,
        })
    return sorted(rows, key=lambda r: (r["kickoff"], r["game_id"]))


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/games")
def games(mode: str = Query("week", pattern="^(week|future)$"),
          team: str = Query("", max_length=100)):
    now = datetime.now(EASTERN)
    try:
        all_games = _read_games(now)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, str(exc)) from exc
    # The current slate is the earliest upcoming season/week, not a calendar-week guess.
    if mode == "week" and all_games:
        first = all_games[0]
        all_games = [g for g in all_games if
                     (g["season"], g["week"]) == (first["season"], first["week"])]
    if team.strip():
        q = team.casefold().strip()
        all_games = [g for g in all_games if
                     q in g["home_team"].casefold() or q in g["away_team"].casefold()]
    return {"games": all_games, "mode": mode, "as_of": now.isoformat()}


@app.get("/api/cache")
def cached_reports():
    return {"reports": [{"game_id": game_id, "team_id": team_id, **report}
            for (game_id, team_id), report in reversed(list(_REPORT_CACHE.items()))],
            "count": len(_REPORT_CACHE), "limit": _CACHE_LIMIT}


@app.delete("/api/cache")
async def clear_cached_reports():
    async with _DISCOVERY_LOCK:
        cleared = len(_REPORT_CACHE)
        _REPORT_CACHE.clear()
    return {"cleared": cleared}


@app.post("/api/discover")
async def run_discovery(request: MatchupRequest):
    try:
        all_games = await asyncio.to_thread(_read_games, datetime.now(EASTERN))
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, str(exc)) from exc
    matches = [g for g in all_games if g["game_id"] == request.game_id]
    if len(matches) != 1:
        raise HTTPException(404, "Upcoming scheduled game not found")
    game = matches[0]
    if request.team_id is not None and request.team_id not in (game["home_id"], game["away_id"]):
        raise HTTPException(422, "Selected team is not in this matchup")
    selected_home = request.team_id is None or request.team_id == game["home_id"]
    team = game["home_team"] if selected_home else game["away_team"]
    opponent = game["away_team"] if selected_home else game["home_team"]
    cache_key = (game["game_id"], int(game["home_id"] if selected_home else game["away_id"]))
    cached = _REPORT_CACHE.get(cache_key)
    if cached is not None:
        _REPORT_CACHE.move_to_end(cache_key)
        return cached

    if _DISCOVERY_LOCK.locked():
        raise HTTPException(429, "Another matchup is being analyzed. Please retry shortly.")
    async with _DISCOVERY_LOCK:
        cached = _REPORT_CACHE.get(cache_key)
        if cached is not None:
            _REPORT_CACHE.move_to_end(cache_key)
            return cached
        try:
            result = await asyncio.to_thread(
                discover, team=team, opponent=opponent, game_date=game["date"],
                home=selected_home, print_report=False,
            )
        except FileNotFoundError as exc:
            raise HTTPException(503, "Processed data unavailable; run python process.py") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        response = {"game": game, "team": result.team, "opponent": result.opponent,
                    "report": result.report, "count": len(result.discoveries)}
        _REPORT_CACHE[cache_key] = response
        if len(_REPORT_CACHE) > _CACHE_LIMIT:
            _REPORT_CACHE.popitem(last=False)
        return response
