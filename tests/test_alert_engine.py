from asr_pro.core.alert_engine import evaluate_alerts


def test_evaluate_alerts(db_session):
    # Tests that evaluate_alerts runs without crashing when there are no active rules
    events = evaluate_alerts(db_session)
    assert events == []
