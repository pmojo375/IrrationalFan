from dataclasses import replace
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from discovery.engine.models import FeatureSpec, SearchConfig, RankingWeights
from discovery.engine.search import DiscoveryEngine
from webapp.app import app, MatchupRequest, _effective_config


def test_stat_catalog_is_curated_and_separates_perspectives():
    catalog = TestClient(app).get('/api/features').json()['features']
    ids = {f['id'] for f in catalog}
    assert {'wins', 'opp_wins', 'rest_days', 'opp_rest_days'} <= ids
    assert not {'won', 'team_score', 'month_name', 'stadium_capacity'} & ids
    assert len(ids) == len(catalog)


@pytest.mark.parametrize('settings', [
    {'required_stats':['wins'], 'excluded_stats':['wins']},
    {'required_stats':['wins','losses'], 'max_conditions':1},
    {'allowed_stats':[]},
    {'allowed_stats':['wins'], 'required_stats':['losses']},
    {'excluded_stats':['won']},
    {'allowed_stats':['wins','wins']},
    {'required_stats':['rest_days'], 'feature_groups':['records']},
])
def test_invalid_stat_combinations_rejected(settings):
    with pytest.raises(ValueError):
        _effective_config(MatchupRequest(game_id='1', **settings))


def test_explicit_advanced_selection_enables_generation():
    config = _effective_config(MatchupRequest(game_id='1', allowed_stats=['EPAplay_off']))
    assert config.search.include_advanced
    assert config.search.allowed_features == ('EPAplay_off',)


def fixture_frame():
    return pd.DataFrame({'wins':[2]*12, 'opp_wins':[3]*12, 'rest_days':[6]*12,
                         'month':[11]*12, 'favorable':[True]*12,
                         'game_date':pd.date_range('2025-01-01',periods=12)})


def test_required_stats_seed_search_even_when_redundant_and_budget_is_small():
    frame = fixture_frame()
    config = SearchConfig(required_features=('rest_days','wins'),
                          allowed_features=('rest_days','wins'), max_atomic_candidates=1,
                          min_sample_size=3, max_conditions=2, max_depth=2)
    engine = DiscoveryEngine(config=config)
    ds,_ = engine.run_scope(frame,[FeatureSpec(n,'numeric',n) for n in frame if n not in ('favorable','game_date')],
                            {'wins':2,'rest_days':6,'month':11},scope='team')
    assert ds
    assert all({c.feature for c in d.conditions} == {'rest_days','wins'} for d in ds)


def test_missing_required_stat_does_not_silently_relax_search():
    frame=fixture_frame()
    engine=DiscoveryEngine(config=SearchConfig(required_features=('rest_days',)))
    ds,_=engine.run_scope(frame,[FeatureSpec('rest_days','numeric')], {'rest_days':None},scope='team')
    assert ds == []


def test_team_and_opponent_selections_are_remapped_in_opponent_history():
    frame=fixture_frame()
    config=SearchConfig(allowed_features=('wins','opp_wins'), required_features=('wins',),
                        excluded_features=('opp_wins',), opponent_feature_map={'wins':'opp_wins','opp_wins':'wins'})
    engine=DiscoveryEngine(config=config)
    features=[FeatureSpec('wins','numeric'), FeatureSpec('opp_wins','numeric')]
    current={'wins':2,'opp_wins':3}
    own,_=engine.run_scope(frame,features,current,scope='team')
    opp,_=engine.run_scope(frame,features,current,scope='opponent')
    assert own and opp
    assert all({c.feature for c in d.conditions} == {'wins'} for d in own)
    assert all({c.feature for c in d.conditions} == {'opp_wins'} for d in opp)
