import pytest
from datetime import datetime, timedelta
from asr_pro.core.trend_engine import get_trend_data, detect_anomalies, forecast_tomorrow, init_db

def test_trend_data_generation():
    init_db()
    data = get_trend_data(days=7)
    assert isinstance(data, dict)
    assert len(data) >= 7

def test_anomaly_detection_thresholds():
    # Mock data to simulate an anomaly
    mock_data = {
        "2023-10-01": {"App Crash": 5},
        "2023-10-02": {"App Crash": 5},
        "2023-10-03": {"App Crash": 6},
        "2023-10-04": {"App Crash": 25}, # 400% increase
        "2023-10-05": {"App Crash": 30},
    }
    alerts = detect_anomalies(mock_data)
    assert len(alerts) > 0
    assert alerts[0].topic == "App Crash"
    assert alerts[0].severity == "CRITICAL"

def test_forecast_tomorrow():
    # Mock data simulating a clear upward trend
    mock_data = {
        "2023-10-01": {"Billing": 10},
        "2023-10-02": {"Billing": 12},
        "2023-10-03": {"Billing": 15},
        "2023-10-04": {"Billing": 18},
        "2023-10-05": {"Billing": 20},
        "2023-10-06": {"Billing": 22},
        "2023-10-07": {"Billing": 25},
    }
    forecast = forecast_tomorrow(mock_data, days_to_look_back=7)
    assert len(forecast) == 1
    assert forecast[0].topic == "Billing"
    assert forecast[0].predicted_volume > 25
    assert forecast[0].trend_slope > 0
