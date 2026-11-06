"""D1-only CFBD team/venue metadata enrichment (2026 snapshot, not historical)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "cfbd_teams_2026.json"
D1 = {"fbs", "fcs"}
FIELDS = {
    "mascot": "mascot", "color": "primary_color", "alternateColor": "alternate_color",
    "classification": "classification", "conference": "snapshot_conference",
}
VENUE = {
    "name": "stadium_name", "city": "stadium_city", "state": "stadium_state",
    "capacity": "stadium_capacity", "constructionYear": "stadium_built",
    "elevation": "stadium_elevation_m", "latitude": "stadium_latitude",
    "longitude": "stadium_longitude", "grass": "stadium_grass", "dome": "stadium_dome",
}

def _color(value):
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        return None
    try:
        int(value[1:], 16)
    except ValueError:
        return None
    return value.lower()

def load_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["team_id", *FIELDS.values(), *VENUE.values()])
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("CFBD teams JSON must be an array")
    rows = []
    for item in records:
        if not isinstance(item, dict) or str(item.get("classification", "")).lower() not in D1:
            continue
        venue = item.get("location") or {}
        if not isinstance(venue, dict):
            venue = {}
        row = {"team_id": item.get("id")}
        row.update({dest: item.get(source) for source, dest in FIELDS.items()})
        row.update({dest: venue.get(source) for source, dest in VENUE.items()})
        row["primary_color"] = _color(row["primary_color"])
        row["alternate_color"] = _color(row["alternate_color"])
        rows.append(row)
    out = pd.DataFrame(rows, columns=["team_id", *FIELDS.values(), *VENUE.values()])
    if out.empty:
        raise ValueError("No FBS/FCS teams found in CFBD JSON")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="raise").astype("int64")
    if out["team_id"].duplicated().any():
        raise ValueError("Duplicate CFBD team IDs")
    return out

def _distance(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    a, b, c, d = map(math.radians, (float(lat1), float(lon1), float(lat2), float(lon2)))
    h = math.sin((c-a)/2)**2 + math.cos(a)*math.cos(c)*math.sin((d-b)/2)**2
    return 3958.7613 * 2 * math.asin(min(1, math.sqrt(h)))

def enrich_metadata(df: pd.DataFrame, path: Path = METADATA_PATH) -> pd.DataFrame:
    """Match both perspectives by CFBD team ID; preserve unknowns, never backfill non-D1."""
    meta = load_metadata(path)
    if meta.empty:
        return df
    out = df.copy()
    for key, prefix in (("team_id", ""), ("opponent_id", "opp_")):
        side = meta.rename(columns={"team_id": key, **{c: prefix+c for c in meta if c != "team_id"}})
        out = out.merge(side, on=key, how="left", validate="many_to_one")
    for col in ("stadium_capacity", "stadium_built", "stadium_elevation_m", "opp_stadium_capacity"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["same_mascot"] = (out["mascot"].notna() & out["opp_mascot"].notna() & out["mascot"].str.casefold().eq(out["opp_mascot"].str.casefold())).astype("boolean")
    out["same_mascot"] = out["same_mascot"].where(out["mascot"].notna() & out["opp_mascot"].notna())
    out["same_primary_color"] = out["primary_color"].eq(out["opp_primary_color"]).astype("boolean").where(out["primary_color"].notna() & out["opp_primary_color"].notna())
    out["opponent_stadium_larger"] = (out["opp_stadium_capacity"] > out["stadium_capacity"]).astype("boolean").where(out["stadium_capacity"].notna() & out["opp_stadium_capacity"].notna())
    out["campus_distance_miles"] = [
        _distance(a,b,c,d) for a,b,c,d in zip(out["stadium_latitude"],out["stadium_longitude"],out["opp_stadium_latitude"],out["opp_stadium_longitude"])
    ]
    # Snapshot features describe the schools' 2026 venues, not where historical games occurred.
    return out
