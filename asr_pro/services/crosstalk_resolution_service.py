"""Resolve crosstalk (overlapping speech) events into per-speaker text.

`DiarizationService.extract_crosstalk_events()` reports *when* two speakers
talked over each other (real acoustic overlap regions when diarization used
pyannote). Whisper itself only ever produces one transcript for that window -
whichever voice it locks onto - silently losing the other party's words. This
module uses `speech_separation_service` to split the overlapping audio window
into two estimated streams and transcribes each independently, so both
parties' words in a crosstalk moment are captured instead of one.

This is deliberately scoped to crosstalk windows only (typically a few
hundred milliseconds to a couple of seconds per call), not the whole
recording - running source separation + a second Whisper pass on entire calls
would multiply compute cost for no benefit outside overlap regions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from asr_pro.config import settings

SAMPLE_RATE = 16000


def _extract_window(full_audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    start_idx = max(0, int(start_sec * SAMPLE_RATE))
    end_idx = min(len(full_audio), int(end_sec * SAMPLE_RATE))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return full_audio[start_idx:end_idx].astype(np.float32)


def resolve_crosstalk_events(
    audio_path: str,
    crosstalk_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich crosstalk events with separated per-speaker transcripts.

    Returns the same list of event dicts, each augmented in-place with a
    `separated_transcripts` key (list of {speaker, text} per estimated
    source) when separation + re-transcription succeeded. Events below
    `crosstalk_separation_min_duration_sec` or where separation/transcription
    fails are left untouched - callers should treat a missing
    `separated_transcripts` key as "not resolved", not an error.
    """
    if not settings.crosstalk_separation_enabled or not crosstalk_events:
        return crosstalk_events

    eligible = [
        e
        for e in crosstalk_events
        if e.get("duration", 0.0) >= settings.crosstalk_separation_min_duration_sec
    ]
    if not eligible:
        return crosstalk_events

    try:
        from faster_whisper import decode_audio

        full_audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        logger.warning(
            f"CrosstalkResolution: could not decode {audio_path} ({exc}). Skipping resolution."
        )
        return crosstalk_events

    from asr_pro.services.asr_service import ASRService
    from asr_pro.services.audio_conditioning import condition_telephony_audio
    from asr_pro.services.speech_separation_service import separate_two_speakers

    asr = ASRService.get_instance()
    resolved_count = 0

    for event in eligible:
        window = _extract_window(full_audio, event["start"], event["end"])
        if window.size < SAMPLE_RATE * 0.1:  # sub-100ms windows aren't worth separating
            continue
        window = condition_telephony_audio(window, sample_rate=SAMPLE_RATE)

        streams = separate_two_speakers(window, sample_rate=SAMPLE_RATE)
        if not streams:
            continue

        speakers = event.get("speakers") or []
        separated = []
        for idx, stream in enumerate(streams):
            try:
                segments, _duration = asr.transcribe_array(stream, language="tr")
                text = " ".join(s.text for s in segments).strip()
            except Exception as exc:
                logger.warning(
                    f"CrosstalkResolution: re-transcription failed for stream {idx} ({exc})."
                )
                text = ""
            if text:
                speaker_label = speakers[idx] if idx < len(speakers) else f"stream_{idx}"
                separated.append({"speaker": speaker_label, "text": text})

        if separated:
            event["separated_transcripts"] = separated
            resolved_count += 1

    if resolved_count:
        logger.info(
            f"CrosstalkResolution: resolved {resolved_count}/{len(eligible)} crosstalk event(s)."
        )

    return crosstalk_events
