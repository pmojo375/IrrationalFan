"""User-selectable feature families. Future CFBD metadata stays disabled until deployed."""
from __future__ import annotations

from discovery.engine.models import FeatureSpec

# Explicit names/prefixes prevent newly ingested columns from becoming searchable by accident.
GROUPS = {
    "records": "Pregame records & streaks",
    "schedule": "Schedule & timing",
    "previous": "Previous games & rest",
    "rankings": "Rankings & conferences",
    "advanced": "Advanced performance",
}
RECORD = {"wins", "losses", "games_played", "win_pct", "win_streak", "loss_streak",
          "has_winning_record", "has_losing_record", "win_diff"}
SCHEDULE = {"is_home", "neutral_site", "conference_competition", "month",
            "month_name", "week", "day_of_week"}
RANKING = {"rank", "conference", "division"}
PREVIOUS = {"previous_points_for", "previous_points_against", "previous_margin",
            "previous_won", "previous_conference_game", "previous_opponent_conference",
            "rest_days", "after_bye", "conference_wins", "conference_losses"}

def feature_group(name: str) -> str | None:
    core = name[4:] if name.startswith("opp_") else name
    if core in RECORD:
        return "records"
    if core in SCHEDULE:
        return "schedule"
    if core in PREVIOUS:
        return "previous"
    if core in RANKING:
        return "rankings"
    # Existing curated advanced stats, not the not-yet-deployed team/venue snapshot.
    if core in {"venue", "attendance"} or core.startswith((
        "EPAplay_", "success_", "explosive_", "havoc_", "yardsplay_",
        "third_down_", "red_zone_", "turnover", "passrate_", "rushrate_", "playsgame_"
    )):
        return "advanced"
    return None

def filter_features(features: list[FeatureSpec], groups: tuple[str, ...] | None) -> list[FeatureSpec]:
    if groups is None:
        return features
    allowed = set(groups)
    return [f for f in features if feature_group(f.name) in allowed]


def feature_catalog():
    """Curated user-facing IDs; aliases and outcome fields are never selectable."""
    from discovery.sports.college_football.features import feature_specs, advanced_feature_specs
    unique = {f.name: f for f in [*feature_specs(), *advanced_feature_specs()]
              if f.name != "month_name"}
    return [{"id": f.name, "label": f.label, "group": feature_group(f.name),
             "perspective": "opponent" if f.name.startswith("opp_") else "team",
             "advanced": f.name not in {x.name for x in feature_specs()}}
            for f in unique.values()]
