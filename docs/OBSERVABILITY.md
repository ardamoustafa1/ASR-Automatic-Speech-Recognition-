# ASR-Pro Observability Guide

## What's exposed

The API always exposes Prometheus metrics at `GET /metrics` (excluded from auth and from
the request-latency histogram itself). Two families of metrics are available:

**Generic HTTP telemetry** (via `prometheus-fastapi-instrumentator`, always on):
- `http_requests_total{handler,method,status}`
- `http_request_duration_seconds_bucket{handler,method,le}`
- `http_request_size_bytes`, `http_response_size_bytes`

**ASR-Pro business metrics** (`asr_pro/observability/metrics.py`):
| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `ws_active_connections` | Gauge | — | Currently open `/ws/live-asr` WebSocket connections |
| `ws_messages_total` | Counter | `type` (partial/final/final_flush/error) | Live ASR streaming protocol messages sent |
| `asr_transcribe_duration_seconds` | Histogram | `mode` (batch/streaming) | Whisper inference wall-clock time |
| `nlp_engine_duration_seconds` | Histogram | `engine` (keyword/topic/empathy/churn) | Per-engine analysis duration |
| `audit_log_write_failures_total` | Counter | — | Failed audit log DB writes (previously silently swallowed) |
| `churn_risk_score` | Histogram | — | Distribution of churn risk scores (0-1) per analyzed conversation — watch for drift, see [CHURN_RISK_METHODOLOGY.md](CHURN_RISK_METHODOLOGY.md) |

## Running the observability stack locally

Prometheus, Grafana, and Alertmanager are opt-in via the `with-observability` Docker
Compose profile — they never start on a plain `docker compose up`:

```bash
docker compose --profile with-observability up -d db api prometheus grafana alertmanager
```

- **Prometheus**: http://localhost:9090 — scrapes `api:8000/metrics` every 15s
  (`deploy/prometheus/prometheus.yml`), loads alert rules from
  `deploy/prometheus/alerts.yml`.
- **Grafana**: http://localhost:3000 — login `admin` / `admin` (change on first login
  in any real deployment). The Prometheus datasource and the "ASR-Pro Overview"
  dashboard are auto-provisioned from `deploy/grafana/provisioning/` — nothing to
  click through manually.
- **Alertmanager**: http://localhost:9093 — receives firing alerts from Prometheus.

## Dashboard panels ("ASR-Pro Overview")

HTTP request rate by status, HTTP 5xx error ratio, HTTP p95 latency, active live-ASR
WebSocket connections, audit log write failures, WS message rate by type, ASR
transcribe p95 by mode (batch vs. streaming), and NLP engine p95 by engine. Defined in
`deploy/grafana/dashboards/asr-pro-overview.json`.

## Alert rules

Defined in `deploy/prometheus/alerts.yml`:
- **HighHttp5xxRate** — >5% of responses are 5xx over 5m.
- **HighRequestLatencyP95** — HTTP p95 latency above 1.5s for 5m.
- **WebSocketConnectionsNearLimit** — `ws_active_connections` above 40 (default
  `MAX_WS_CONNECTIONS` is 50, see `asr_pro/config.py`) for 2m — new connections may
  soon start being rejected with close code 1013.
- **ASRTranscribeSlow** — Whisper inference p95 above 5s for 5m, broken down by mode.

## Wiring a real notification channel

`deploy/alertmanager/alertmanager.yml` ships with a placeholder `default` receiver that
only logs to Alertmanager itself — no credentials are bundled. To page a real channel,
replace the receiver block, e.g. for Slack:

```yaml
receivers:
  - name: default
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T000/B000/XXXX"
        channel: "#asr-pro-alerts"
```

or a `pagerduty_configs` / `webhook_configs` block — see the [Alertmanager
configuration docs](https://prometheus.io/docs/alerting/latest/configuration/#receiver).

## Deliberately out of scope

- **Distributed tracing (OpenTelemetry)**: the existing `trace_id` (see
  `asr_pro/api/main.py`, propagated via the `X-Request-ID`/`X-Trace-ID` headers and
  attached to every log line) gives request correlation across log lines without
  adding a collector/exporter dependency that has no other consumer today.
- **WER/RTF benchmark results as Prometheus metrics**: `scripts/evaluate_wer.py`
  already enforces a regression threshold in CI (`--max-wer`); pushing those numbers
  to a Pushgateway for historical trending is a reasonable follow-up but a separate
  change.
