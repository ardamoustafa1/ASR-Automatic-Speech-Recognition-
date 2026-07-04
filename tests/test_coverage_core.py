from asr_pro.core.alert_engine import _condition_matches
from asr_pro.core.topic_classifier import classify_topics_from_hits


def test_classify_topics_from_hits():
    # Empty hits should return empty list
    res = classify_topics_from_hits([], None)
    assert res == []


def test_condition_matches():
    trend_mock = type("obj", (object,), {"pct_change": 25.0, "current_count": 100})
    condition = {
        "metric": "pct_increase",
        "operator": "pct_increase",
        "threshold": 20.0,
        "min_count": 5,
    }
    assert _condition_matches(trend_mock, condition) is True

    condition_lt = {
        "metric": "pct_increase",
        "operator": "pct_increase",
        "threshold": 30.0,
        "min_count": 5,
    }
    assert _condition_matches(trend_mock, condition_lt) is False

    condition_gte = {"metric": "hit_count", "operator": "gte", "threshold": 100, "min_count": 5}
    assert _condition_matches(trend_mock, condition_gte) is True

    condition_false = {"metric": "hit_count", "operator": "gte", "threshold": 200, "min_count": 5}
    assert _condition_matches(trend_mock, condition_false) is False
