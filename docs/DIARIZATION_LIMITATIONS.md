# Speaker Diarization: Known Limitations

Every downstream engine (churn, empathy, compliance) isolates its analysis to
one speaker's segments (`agent_speaker_id` / `customer_speaker_id`), so a
speaker-attribution error corrupts that engine's output for the affected
line. This document states plainly what the current diarization approach can
and cannot guarantee.

## How stereo calls are diarized

Stereo recordings (`asr_pro/services/asr_service.py`, `transcribe`) have
their left/right channels transcribed **independently** by Whisper - each
channel gets its own VAD pass with no shared timeline reference. Segments are
tagged `SPEAKER_00`/`SPEAKER_01` directly from which physical channel
produced them, then merged and sorted by timestamp
(`asr_pro/services/diarization_service.py`, `_align_stereo_segments`).

## What this cannot fix

If the recording itself has **channel crosstalk/bleed** (one party's voice
audible, even faintly, on the other party's channel - common with analog
call-center recording equipment), Whisper can transcribe the bleed-through
audio as if the wrong party said it. This is a recording-quality limit, not a
software bug: no purely software-side heuristic can perfectly separate two
voices that are physically present on both channels. `diarization_energy_margin`
(`asr_pro/config.py`) provides some defense-in-depth for the fallback
energy-comparison path, but does not apply when a segment already carries a
channel tag from independent transcription (the common case).

Short, low-information exchanges (single-word greetings, backchannel acks)
are also Whisper's most hallucination-prone case regardless of diarization -
a known general ASR limitation, not specific to this pipeline.

## What is mitigated

Whisper's own per-segment confidence (`avg_logprob`) is surfaced directly in
the transcript UI (dimmed text + a warning icon below the configured
`transcript_low_confidence_logprob` threshold) so a human reviewer can spot
the lines most likely to be wrong instead of trusting the whole transcript
uniformly.

`assign_speakers_to_segments()` (`asr_pro/services/diarization_service.py`)
now reports which method actually produced the speaker split -
`"pyannote"` / `"stereo_physical"` (acoustically grounded) vs `"stereo_energy"`
/ `"text_heuristic"` (degraded, no real acoustic separation). The call-level
`speaker_separation_reliable` flag and `diarization_method` field
(`asr_pro/services/conversation_service.py`) only mark a call reliable when
an acoustically-grounded method produced two distinct speakers - two
different speaker IDs alone is not sufficient, since a degraded method can
also produce two labels with zero real acoustic signal behind them. A
production deployment (`ASR_ENV=prod`) additionally refuses to start at all
without a Hugging Face token (`asr_pro/config.py`), so pyannote can never
silently degrade to a heuristic without the operator knowing.

Crosstalk/interruption events (`extract_crosstalk_events`) are computed from
pyannote's true acoustic overlap regions (`DiarizationResult.overlap_regions`,
via the non-exclusive diarization annotation's `get_overlap()`) when
available, instead of guessing from Whisper segment-boundary timestamps.

Crosstalk windows are no longer transcribed as a single garbled/dropped
stream: `asr_pro/services/speech_separation_service.py` (speechbrain SepFormer,
trained on WHAMR! - noisy/reverberant conditions closer to telephony than
clean-speech separation models) splits the overlapping window into two
estimated per-speaker streams, and `crosstalk_resolution_service.py`
re-transcribes each independently, attaching `separated_transcripts` to the
event. Verified on a real 2-voice mixture: separated streams matched their
true source at 0.73/0.53 ECAPA-TDNN cosine similarity vs 0.27/0.15
cross-matches - real, correctly-directed (if imperfect) separation, not a
no-op. Controlled by `crosstalk_separation_enabled` /
`crosstalk_separation_min_duration_sec` (`asr_pro/config.py`) since it adds
per-event latency.

Diarization accuracy is measurable, not just asserted: `asr_pro/services/
diarization_eval.py` and `scripts/evaluate_diarization.py` compute
Diarization Error Rate (DER) against RTTM-format human-annotated reference
calls using `pyannote.metrics`, the same metric pyannote reports for its own
published benchmarks.

Agent voice identification (`asr_pro/services/biometric_service.py`) uses a
real learned speaker embedding (ECAPA-TDNN, speechbrain/spkrec-ecapa-voxceleb)
rather than a hand-rolled spectral vector; every voiceprint records which
model produced it (`embedding_model`) so mismatched-model comparisons never
happen even if the fallback embedding is used on a machine without the
pretrained model cached.

## Recommendation

Treat any single call's churn/empathy/compliance score as an **AI-assisted
signal for a human to review**, not an autonomous verdict - particularly for
short or noisy calls. Before relying on aggregate statistics across many
calls for a business decision, spot-check a sample of the underlying
transcripts against their audio.
