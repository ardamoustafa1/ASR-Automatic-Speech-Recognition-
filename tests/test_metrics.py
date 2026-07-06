from prometheus_client import Histogram

from asr_pro.observability.metrics import time_block, ws_active_connections


def _sample_value(hist: Histogram, name_suffix: str, **labels) -> float:
    for family in hist.collect():
        for sample in family.samples:
            if sample.name.endswith(name_suffix) and sample.labels.get("le") is None:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    raise AssertionError(f"no sample found for {name_suffix} with labels {labels}")


def test_time_block_observes_duration_with_labels():
    hist = Histogram("test_time_block_labeled_seconds", "test histogram", ["engine"])
    with time_block(hist, engine="unit_test"):
        pass
    assert _sample_value(hist, "_count", engine="unit_test") == 1


def test_time_block_observes_duration_without_labels():
    hist = Histogram("test_time_block_unlabeled_seconds", "test histogram")
    with time_block(hist):
        pass
    assert _sample_value(hist, "_count") == 1


def test_time_block_observes_even_on_exception():
    hist = Histogram("test_time_block_exception_seconds", "test histogram")
    try:
        with time_block(hist):
            raise ValueError("boom")
    except ValueError:
        pass
    assert _sample_value(hist, "_count") == 1


def test_metrics_endpoint_exposes_custom_metric_names(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "ws_active_connections" in body
    assert "ws_messages_total" in body
    assert "asr_transcribe_duration_seconds" in body
    assert "nlp_engine_duration_seconds" in body
    assert "audit_log_write_failures_total" in body


def test_metrics_endpoint_does_not_require_auth(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_ws_active_connections_gauge_inc_dec_roundtrip():
    before = ws_active_connections._value.get()
    ws_active_connections.inc()
    assert ws_active_connections._value.get() == before + 1
    ws_active_connections.dec()
    assert ws_active_connections._value.get() == before
