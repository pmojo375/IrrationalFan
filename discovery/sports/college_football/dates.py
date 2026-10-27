"""Robust game-date parsing for mixed sportsdataverse timestamp formats."""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd


def parse_game_dates(values: Any) -> pd.DatetimeIndex | pd.Timestamp:
    """
    Parse schedule timestamps that mix `...Z` and `....000Z` styles.

    Always prefer ISO8601 so a few odd rows do not become NaT and sort to the end.
    """
    return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")


def parse_as_of(game_date: Union[str, pd.Timestamp, None]) -> Optional[pd.Timestamp]:
    """Parse the matchup cutoff timestamp used for backtesting."""
    if game_date is None or (isinstance(game_date, float) and pd.isna(game_date)):
        return None
    ts = parse_game_dates(game_date)
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0] if len(ts) else pd.NaT
    if pd.isna(ts):
        # Allow plain YYYY-MM-DD
        ts = pd.to_datetime(game_date, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def before_as_of_mask(game_dates: Any, as_of: Optional[pd.Timestamp]) -> pd.Series:
    """True for rows strictly before the matchup kickoff / as-of timestamp."""
    parsed = parse_game_dates(game_dates)
    if as_of is None or pd.isna(as_of):
        return pd.Series(True, index=getattr(parsed, "index", None))
    return parsed.notna() & (parsed < as_of)
