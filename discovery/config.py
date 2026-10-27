"""Default configuration for the discovery engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from discovery.engine.models import RankingWeights, SearchConfig


DATA_PATH = Path(__file__).resolve().parent.parent / "cfb_team_games_2026.parquet"


@dataclass
class EngineConfig:
    """Top-level knobs for a discovery run."""

    search: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            beam_width=40,
            max_depth=4,
            min_sample_size=3,
            max_conditions=4,
            max_atomic_candidates=80,
            scopes=("team", "opponent", "all", "h2h"),
        )
    )
    ranking: RankingWeights = field(default_factory=RankingWeights)
    top_n: int = 20
    data_path: Path = field(default_factory=lambda: DATA_PATH)
