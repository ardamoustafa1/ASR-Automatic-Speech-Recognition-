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
uniformly. The call-level `speaker_separation_reliable` flag
(`asr_pro/services/conversation_service.py`) catches the coarser failure mode
where diarization can't distinguish two speakers at all.

## Recommendation

Treat any single call's churn/empathy/compliance score as an **AI-assisted
signal for a human to review**, not an autonomous verdict - particularly for
short or noisy calls. Before relying on aggregate statistics across many
calls for a business decision, spot-check a sample of the underlying
transcripts against their audio.
