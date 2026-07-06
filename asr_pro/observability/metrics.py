# Custom Prometheus metrics for business-level observability (beyond generic HTTP telemetry).
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram

ws_active_connections = Gauge(
    "ws_active_connections",
    "Number of currently open /ws/live-asr WebSocket connections.",
)

ws_messages_total = Counter(
    "ws_messages_total",
    "Live ASR streaming protocol messages sent to clients, by type.",
    ["type"],
)

asr_transcribe_duration_seconds = Histogram(
    "asr_transcribe_duration_seconds",
    "Whisper inference duration in seconds.",
    ["mode"],
)

nlp_engine_duration_seconds = Histogram(
    "nlp_engine_duration_seconds",
    "NLP analysis engine duration in seconds, by engine.",
    ["engine"],
)

audit_log_write_failures_total = Counter(
    "audit_log_write_failures_total",
    "Number of times writing an audit log entry failed.",
)

churn_risk_score = Histogram(
    "churn_risk_score",
    "Distribution of churn risk scores (0-1) produced per analyzed conversation.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0),
)


@contextmanager
def time_block(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Observe the wall-clock duration of the wrapped block into `histogram`."""
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        (histogram.labels(**labels) if labels else histogram).observe(elapsed)
