from dataclasses import replace

import pytest

from discovery.engine.models import Condition, FeatureSpec, Operator
from discovery.sports.college_football.feature_groups import GROUPS, feature_group, filter_features
from webapp.app import MatchupRequest, _effective_config


def test_groups_classify_existing_and_cfbd_features():
    assert feature_group("opp_wins") == "records"
    assert feature_group("rest_days") == "previous"
    assert feature_group("opp_EPAplay_off") == "advanced"
    assert feature_group("same_primary_color") == "team_stadium"
    assert feature_group("stadium_capacity") == "team_stadium"


def test_group_filter_does_not_leak_other_features():
    specs = [FeatureSpec("wins", "numeric"), FeatureSpec("month", "categorical"),
             FeatureSpec("rest_days", "numeric"), FeatureSpec("EPAplay_off", "numeric")]
    assert [f.name for f in filter_features(specs, ("records", "previous"))] == ["wins", "rest_days"]
    assert filter_features(specs, None) == specs


def test_request_config_and_validation():
    base = MatchupRequest(game_id="123")
    assert _effective_config(base).search.selected_feature_groups is None
    custom = MatchupRequest(game_id="123", feature_groups=["records", "advanced"])
    config = _effective_config(custom)
    assert config.search.include_advanced
    assert config.search.selected_feature_groups == ("advanced", "records")
    with pytest.raises(ValueError):
        _effective_config(MatchupRequest(game_id="123", feature_groups=["stadium"]))
    with pytest.raises(ValueError):
        _effective_config(MatchupRequest(game_id="123", feature_groups=[]))
    with pytest.raises(ValueError):
        _effective_config(MatchupRequest(game_id="123", feature_groups=["records", "schedule"], max_conditions=1, require_all_groups=True))
