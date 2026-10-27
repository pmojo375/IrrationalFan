"""Default configuration for the discovery engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from discovery.engine.models import RankingWeights, SearchConfig


_ROOT = Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    """Prefer the multi-season build; fall back to legacy single-season file."""
    multi = _ROOT / "cfb_team_games.parquet"
    if multi.exists():
        return multi
    return _ROOT / "cfb_team_games_2026.parquet"


# Resolved at import for convenience; EngineConfig re-resolves via default_data_path.
DATA_PATH = default_data_path()


@dataclass
class EngineConfig:
    """Top-level knobs for a discovery run."""

    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            beam_width=40,
            max_depth=4,
            min_sample_size=3,
            max_conditions=4,
            max_atomic_candidates=120,
            scopes=("team", "opponent", "all", "h2h"),
        )
    )
    ranking: RankingWeights = field(default_factory=RankingWeights)
    top_n: int = 20
    data_path: Path = field(default_factory=default_data_path)
