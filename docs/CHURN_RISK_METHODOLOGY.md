# Churn Risk Scoring Methodology

Engine: [`asr_pro/core/churn_engine.py`](../asr_pro/core/churn_engine.py) (`analyze_churn_risk`).
Used by both the Streamlit analysis tool and the FastAPI backend
(`asr_pro/services/conversation_service.py`) — there is a single scoring
implementation, not two.

## How a score is produced

1. **Chunking**: customer-only speech (agent lines are excluded via
   `customer_speaker_id`) is grouped into blocks of ~5 transcript segments to
   keep the NLP model call count low on long calls.
2. **Per-chunk model signal**: each chunk is classified with a zero-shot NLI
   model (`asr_pro/core/sentiment_engine.py`) against the labels `canceling
   service` / `switching to competitor` / `continue service` / `neutral
   complaint`. `base_risk` is the summed probability of the first two labels.
3. **Acoustic amplification**: `base_risk` is amplified if the chunk's
   speaking pace (words per minute) crosses a stress threshold — configurable
   via `ASR_CHURN_WPM_HIGH_THRESHOLD` / `ASR_CHURN_WPM_HIGH_MULTIPLIER` and
   the `_MID_` equivalents.
4. **Temporal weighting**: risk expressed later in a call is weighted higher
   than risk expressed early (a threat at the end of a call, after an agent
   has already tried to help, is a stronger signal than an opening remark).
5. **Per-chunk clipping**: the amplified value is clipped to `[0, 1]` *before*
   aggregation. This is the critical invariant — multipliers may not compound
   past a single unit of risk.
6. **Call-level aggregation**: the final score blends the single worst chunk
   (`max`) with the temporal-weighted mean across all chunks:
   ```
   risk = ASR_CHURN_MAX_WEIGHT * max(chunk_risks)
        + ASR_CHURN_MEAN_WEIGHT * weighted_mean(chunk_risks)
   ```
   Both weights are validated to sum to 1.0 at startup
   (`asr_pro/config.py::Settings._validate_churn_weights`). Because every
   input is already bounded to `[0, 1]`, the output is bounded by
   construction — the score cannot grow simply because a call has more
   chunks, which was the root cause of a prior miscalibration bug (a long
   call always saturating to 100%).
7. **Competitor bonus**: if a known competitor name is mentioned
   (`COMPETITORS` set) and there is already real risk signal (`> 0.3`), a
   small proportional bonus is added — `ASR_CHURN_COMPETITOR_BONUS_PER_MENTION`
   per unique competitor, capped at `ASR_CHURN_COMPETITOR_BONUS_CAP`. A
   mention nudges an existing signal; it does not override the model.
8. **Alarm threshold**: `is_high_risk` fires at `ASR_CHURN_ALARM_THRESHOLD`
   (default `0.75`).

## Configuration knobs

All tunables live in `Settings` (`asr_pro/config.py`) with the `ASR_` env
prefix, so a deployment can retune for a specific client's call patterns
without a code change or redeploy:

| Env var | Default | Meaning |
|---|---|---|
| `ASR_CHURN_WPM_HIGH_THRESHOLD` | 190 | WPM above which the high stress multiplier applies |
| `ASR_CHURN_WPM_HIGH_MULTIPLIER` | 1.4 | Multiplier applied above the high threshold |
| `ASR_CHURN_WPM_MID_THRESHOLD` | 160 | WPM above which the mid stress multiplier applies |
| `ASR_CHURN_WPM_MID_MULTIPLIER` | 1.2 | Multiplier applied above the mid threshold |
| `ASR_CHURN_MAX_WEIGHT` | 0.65 | Weight of the worst single chunk in the final blend |
| `ASR_CHURN_MEAN_WEIGHT` | 0.35 | Weight of the temporal-weighted mean (must sum to 1.0 with the above) |
| `ASR_CHURN_COMPETITOR_BONUS_PER_MENTION` | 0.1 | Bonus per unique competitor mentioned |
| `ASR_CHURN_COMPETITOR_BONUS_CAP` | 0.2 | Max total competitor bonus |
| `ASR_CHURN_ALARM_THRESHOLD` | 0.75 | Score at/above which `is_high_risk` fires |

## Explainability output

Every `ChurnResult` carries:
- `risk_breakdown`: `{"model_signal", "escalation_trend", "competitor_bonus"}`
  — the exact contribution of each term, so a reviewer can see why a score is
  what it is instead of trusting an opaque number.
- `confidence`: `"Düşük"` / `"Orta"` / `"Yüksek"`, based on how many chunks of
  customer speech actually backed the score — a single short outburst should
  not carry the same weight as several chunks agreeing across a whole call.
- `insights`: per-chunk breakdown (text, base model risk, WPM, multipliers)
  for every chunk that crossed the 0.3 relevance threshold.

Both the Streamlit tool ("Skor Detayı" expander) and the FastAPI response
(`conversation_service.py`'s `"churn"` dict, and `Conversation.metadata_json`)
surface this breakdown.

## Related limitation

Every score here depends on the customer/agent speaker split being correct.
See [DIARIZATION_LIMITATIONS.md](DIARIZATION_LIMITATIONS.md) for what that
split can and cannot guarantee, especially on real stereo call recordings.

## Testing / calibration evidence

- `tests/test_churn_engine.py` — unit tests for individual mechanics (WPM
  amplification, temporal weighting, competitor NER, speaker isolation, and a
  length-invariance regression test for the original saturation bug).
- `tests/test_churn_engine_calibration.py` — a golden-set suite of realistic
  Turkish call-center scenarios asserting each lands in the expected risk
  band (not exact values, since the underlying model is probabilistic), plus
  a monotonicity check (adding an explicit cancellation line to a neutral
  call must not lower the score).
- `churn_risk_score` Prometheus histogram (see
  [OBSERVABILITY.md](OBSERVABILITY.md)) — tracks the live production score
  distribution so a miscalibration is caught from metrics, not a customer
  screenshot.
