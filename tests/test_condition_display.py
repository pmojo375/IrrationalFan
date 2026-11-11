from discovery.engine.models import Condition, Operator
from discovery.sports.college_football.formatting import condition_phrase


def test_generated_advanced_label_does_not_duplicate_threshold():
    condition = Condition("opp_stadium_capacity", Operator.LE, 70138,
                          "Opponent home stadium capacity <= 70138")
    assert condition_phrase(condition) == "Opponent home stadium capacity <= 70,138"


def test_decimal_thresholds_are_readable():
    condition = Condition("EPAplay_off", Operator.GE, 0.123456789,
                          "Offensive EPA/play >= 0.123456789")
    assert condition_phrase(condition) == "Offensive EPA/play >= 0.123"


def test_integer_thresholds_keep_thousands_separators():
    condition = Condition("stadium_capacity", Operator.GE, 100000,
                          "School home stadium capacity >= 100000")
    assert condition_phrase(condition) == "School home stadium capacity >= 100,000"
