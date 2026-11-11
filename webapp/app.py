"""Read-only FastAPI UI for scheduled Irrational Fan matchups."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
import math
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from webapp.jobs import JobStore, QueueFull
from pydantic import BaseModel, Field

from discovery.config import EngineConfig
from discovery.sports.college_football.feature_groups import GROUPS, feature_catalog
from discovery.sports.college_football.dates import parse_game_dates
from discovery.sports.college_football.matchup import discover
from discovery.sports.college_football.schedule import SCHEDULE_DIR

EASTERN = ZoneInfo("America/New_York")
PAGE = Path(__file__).with_name("index.html")
_JOBS = None
_JOBS_LOCK = threading.Lock()
_CACHE_LIMIT = 32


def _job_store():
    global _JOBS
    with _JOBS_LOCK:
        if _JOBS is None:
            path = os.environ.get("IRRATIONALFAN_JOB_DB", str(PAGE.parent.parent / "data" / "web_jobs.sqlite3"))
            _JOBS = JobStore(path, limit=_CACHE_LIMIT)
        return _JOBS


@asynccontextmanager
async def lifespan(app):
    global _JOBS
    _job_store()
    try:
        yield
    finally:
        if _JOBS is not None:
            await asyncio.to_thread(_JOBS.close)
            _JOBS = None


app = FastAPI(title="Irrational Fan", version="0.3.0", lifespan=lifespan)


class MatchupRequest(BaseModel):
    game_id: str = Field(min_length=1, max_length=100)
    team_id: int | None = None
    min_sample_size: int = Field(default=3, ge=3, le=30)
    max_conditions: int = Field(default=4, ge=1, le=8)
    discovery_mode: str = Field(default="entertaining", pattern="^(entertaining|balanced|evidence|perfect|rivalry|chaos|custom)$")
    evidence_strength: int = Field(default=50, ge=0, le=100)
    weirdness: int = Field(default=50, ge=0, le=100)
    feature_groups: list[str] | None = None
    require_all_groups: bool = False
    only_stats_ui: bool = False
    selected_stats_ui: list[str] = Field(default_factory=list, max_length=200)
    allowed_stats: list[str] | None = Field(default=None, max_length=200)
    required_stats: list[str] = Field(default_factory=list, max_length=8)
    excluded_stats: list[str] = Field(default_factory=list, max_length=200)


def _discovery_config(mode: str, evidence_strength: int = 50, weirdness: int = 50) -> EngineConfig:
    """Different search objectives, not just a re-sort of the same results."""
    cfg = EngineConfig()
    if mode == "balanced":
        cfg.ranking = replace(
            cfg.ranking, win_pct=30.0, sample_size=30.0, streak=12.0,
            rarity=5.0, n_conditions=3.0, perfect_record_bonus=5.0,
            tiny_sample_penalty=24.0, tiny_sample_threshold=7,
        )
        cfg.search = replace(cfg.search, beam_width=55, min_success_rate=0.60)
    elif mode == "evidence":
        cfg.ranking = replace(
            cfg.ranking, win_pct=22.0, sample_size=50.0, streak=5.0,
            rarity=0.0, n_conditions=0.0, perfect_record_bonus=2.0,
            tiny_sample_penalty=35.0, tiny_sample_threshold=10,
        )
        cfg.search = replace(cfg.search, beam_width=65, min_success_rate=0.65)
    elif mode == "perfect":
        cfg.ranking = replace(cfg.ranking, perfect_record_bonus=45.0,
                              sample_size=28.0, n_conditions=1.0,
                              tiny_sample_penalty=30.0, tiny_sample_threshold=6)
        cfg.search = replace(cfg.search, beam_width=60, min_success_rate=0.80)
    elif mode == "rivalry":
        cfg.search = replace(cfg.search, scopes=("h2h", "team", "opponent"),
                             beam_width=60, min_success_rate=0.60)
        cfg.ranking = replace(cfg.ranking, sample_size=30.0, rarity=3.0,
                              n_conditions=2.0, tiny_sample_penalty=25.0)
    elif mode == "chaos":
        cfg.search = replace(cfg.search, beam_width=85, max_atomic_candidates=160,
                             min_success_rate=0.55, diverse_search=True,
                             include_advanced=True, novelty_bonus=9.0)
        cfg.ranking = replace(cfg.ranking, rarity=24.0, n_conditions=20.0,
                              sample_size=12.0, tiny_sample_penalty=22.0)
    elif mode == "custom":
        e, w = evidence_strength / 100, weirdness / 100
        cfg.search = replace(cfg.search, min_success_rate=round(.50 + .30 * e, 3),
                             beam_width=30 + round(70 * w),
                             max_atomic_candidates=60 + round(80 * w),
                             diverse_search=w >= .5, include_advanced=w >= .65,
                             novelty_bonus=round(12 * w, 2))
        cfg.ranking = replace(cfg.ranking, sample_size=8 + 52 * e,
                              win_pct=20 + 20 * e, rarity=2 + 30 * w,
                              n_conditions=2 + 22 * w,
                              tiny_sample_penalty=8 + 36 * e,
                              tiny_sample_threshold=3 + round(9 * e))
    return cfg


def _effective_config(request: MatchupRequest) -> EngineConfig:
    cfg = _discovery_config(request.discovery_mode, request.evidence_strength, request.weirdness)
    cfg.search = replace(cfg.search, min_sample_size=request.min_sample_size,
                         max_conditions=request.max_conditions, max_depth=request.max_conditions)
    if request.feature_groups is not None:
        if not request.feature_groups or len(request.feature_groups) != len(set(request.feature_groups)) or any(g not in GROUPS for g in request.feature_groups):
            raise ValueError('Select one or more valid, nonduplicate feature categories')
        if request.require_all_groups and len(request.feature_groups) > request.max_conditions:
            raise ValueError('Maximum conditions must be at least the number of required categories')
        cfg.search = replace(cfg.search, selected_feature_groups=tuple(sorted(request.feature_groups)),
                             require_all_feature_groups=request.require_all_groups,
                             include_advanced=cfg.search.include_advanced or 'advanced' in request.feature_groups)
    elif request.require_all_groups:
        raise ValueError('Select categories before requiring every category')
    catalog = {f["id"]: f for f in feature_catalog()}
    # The API must not accept selections that the deployed parquet cannot support.
    deployed = {f["id"]: f for f in selectable_features()["features"]}
    for values in (request.allowed_stats, request.required_stats, request.excluded_stats):
        if values is not None and (len(values) != len(set(values)) or any(n not in catalog for n in values)):
            raise ValueError("Choose valid, nonduplicate stat IDs")
    allowed, required, excluded = request.allowed_stats, set(request.required_stats), set(request.excluded_stats)
    if required & excluded:
        raise ValueError("A stat cannot be both required and excluded")
    if len(required) > request.max_conditions:
        raise ValueError("Maximum conditions must be at least the number of required stats")
    if allowed is not None and (not allowed or not required.issubset(allowed)):
        raise ValueError("Select at least one allowed stat and include every required stat")
    selected = set(allowed if allowed is not None else catalog) - excluded
    if request.feature_groups is not None:
        selected = {n for n in selected if catalog[n]["group"] in request.feature_groups}
    unavailable = {n for n in selected if deployed[n]["group"] == "team_stadium" and not deployed[n]["available"]}
    if required & unavailable:
        raise ValueError("Required stats are not present in the deployed dataset: " + ", ".join(sorted(required & unavailable)))
    if allowed is not None and unavailable:
        raise ValueError("Selected stats are not present in the deployed dataset: " + ", ".join(sorted(unavailable)))
    if not selected or not required.issubset(selected):
        raise ValueError("Required stats must belong to enabled categories; allow at least one stat")
    cfg.search = replace(cfg.search,
                         allowed_features=tuple(sorted(allowed)) if allowed is not None else None,
                         required_features=tuple(sorted(required)), excluded_features=tuple(sorted(excluded)),
                         include_advanced=cfg.search.include_advanced or any(catalog[n]["advanced"] for n in
                             (selected if allowed is not None else required)))
    return cfg


@app.get("/api/features")
def selectable_features():
    from fastparquet import ParquetFile
    try:
        columns = set(ParquetFile(str(EngineConfig().data_path)).columns)
    except (OSError, ValueError):
        columns = set()
    catalog = feature_catalog()
    for feature in catalog:
        feature["available"] = (feature["id"] in columns if feature["group"] == "team_stadium" else
                                not feature["advanced"] or feature["id"] in columns)
        if not feature["available"]:
            feature["reason"] = "Not present in the processed dataset"
    return {"features": catalog, "groups": GROUPS}


def _config_details(cfg: EngineConfig) -> dict:
    from dataclasses import asdict
    return {"search": asdict(cfg.search), "ranking": asdict(cfg.ranking), "top_n": cfg.top_n}


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


@app.get("/api/jobs")
def recent_jobs():
    return {"jobs": _job_store().list(), "limit": _CACHE_LIMIT}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_store().get(job_id)
    if job is None:
        raise HTTPException(404, "Discovery no longer available. Start a new search.")
    return job


@app.get("/api/cache")
def cached_reports():
    store = _job_store()
    reports = []
    for item in store.list():
        if item["status"] == "complete":
            job = store.get(item["id"])
            if job:  # A concurrent clear may remove a completed report.
                reports.append({**job["result"], **job["settings"], "created_at": job["created_at"]})
    return {"reports": reports, "count": len(reports), "limit": _CACHE_LIMIT}


@app.delete("/api/cache")
def clear_cached_reports():
    return {"cleared": _job_store().clear_completed()}


@app.post("/api/jobs")
async def start_discovery(request: MatchupRequest):
    job, reused = await _submit_discovery(request)
    return JSONResponse(job, status_code=200 if job["status"] == "complete" else 202,
                        headers={"Location": "api/jobs/" + job["id"]})


@app.post("/api/discover")
async def run_discovery(request: MatchupRequest):
    """Compatibility endpoint. The UI uses short-lived /api/jobs requests."""
    job, _ = await _submit_discovery(request)
    while job["status"] in ("queued", "running"):
        await asyncio.sleep(.1)
        job = _job_store().get(job["id"])
        if job is None:
            raise HTTPException(404, "Discovery cleared. Start a new search.")
    if job["status"] == "failed":
        raise HTTPException(job["error_status"] or 500, job["error"])
    return job["result"]


def _file_version(path):
    path = Path(path)
    try:
        stat = path.stat()
        return [str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino]
    except FileNotFoundError:
        return [str(path.resolve()), None]


async def _submit_discovery(request: MatchupRequest):
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
    try:
        config = _effective_config(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    settings = request.model_dump()
    settings["team_id"] = game["home_id"] if selected_home else game["away_id"]
    # Normalize settings that do not affect this search, including category order.
    normalized = dict(settings)
    normalized.pop("only_stats_ui", None)
    normalized.pop("selected_stats_ui", None)
    for name in ("allowed_stats", "required_stats", "excluded_stats"):
        if normalized[name] is not None:
            normalized[name] = sorted(normalized[name])
    if request.discovery_mode != "custom":
        normalized.pop("evidence_strength")
        normalized.pop("weirdness")
    if normalized["feature_groups"] is not None:
        normalized["feature_groups"] = sorted(normalized["feature_groups"])
    identity = {"version": app.version, "settings": normalized, "game": game,
                "config": _config_details(config),
                "data": _file_version(config.data_path),
                "schedules": [_file_version(path) for path in _schedule_files(datetime.now(EASTERN))]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata = {"game": game, "team": team, "opponent": opponent, "settings": settings}
    try:
        return _job_store().submit(key, metadata, lambda: _build_report(request, game, selected_home, config))
    except QueueFull:
        raise HTTPException(429, "The discovery queue is full. Please try again shortly.",
                            headers={"Retry-After": "10"})


def _build_report(request, game, selected_home, config):
    team = game["home_team"] if selected_home else game["away_team"]
    opponent = game["away_team"] if selected_home else game["home_team"]
    try:
        result = discover(
            team=team, opponent=opponent, game_date=game["date"],
            home=selected_home, print_report=False,
            min_sample_size=request.min_sample_size, max_conditions=request.max_conditions,
            config=config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, "Processed data unavailable; run python process.py") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    selected = result.discoveries
    if request.discovery_mode == "perfect":
        selected = [d for d in selected if d.losses == 0 and d.wins >= request.min_sample_size
                    and d.objective == "record"]
    elif request.discovery_mode == "rivalry":
        # Prioritize direct matchup evidence; fall back to team/opponent history
        # when head-to-head samples are too small.
        selected = sorted(selected, key=lambda d: (d.scope == "h2h", d.ranking_score),
                          reverse=True)
    elif request.discovery_mode == "chaos":
        selected = sorted(selected, key=lambda d: (len(d.conditions), d.ranking_score),
                          reverse=True)
    discoveries = [_serialize_discovery(d) for d in selected]
    response = {"game": game, "team": result.team, "opponent": result.opponent,
                "discoveries": discoveries, "settings": request.model_dump(),
                "report": result.report, "count": len(discoveries), "effective_config": _config_details(config)}
    return response


def _serialize_discovery(d):
    """Return a JSON-safe evidence view without losing the original report."""
    def safe(v):
        if v is None:
            return None
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (datetime,)):
            return v.isoformat()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        return str(v)

    return {
        "scope": d.scope, "description": d.description, "record": d.record_str,
        "wins": d.wins, "losses": d.losses, "n_games": d.n_games,
        "win_pct": safe(d.win_pct), "streak": d.streak,
        "ranking_score": safe(d.ranking_score), "objective": d.objective,
        "favorable_label": d.favorable_label,
        "population_size": d.population_size, "candidates_evaluated": d.candidates_evaluated,
        "conditions": [c.display() for c in d.conditions],
        "matching_games": [{
            "season": safe(g.season), "date": safe(g.date), "team": g.team,
            "opponent": g.opponent, "team_score": safe(g.team_score),
            "opponent_score": safe(g.opponent_score), "won": bool(g.won),
        } for g in d.matching_games],
    }
