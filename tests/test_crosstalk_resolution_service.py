"""Tests for crosstalk_resolution_service's graceful degradation behavior.

The real separation+re-transcription quality is covered by
tests/test_speech_separation_service.py; this file verifies the orchestration
logic (filtering, config gating, error handling) never raises and leaves
events untouched when separation isn't available - the path every call hits
in test mode / without the SepFormer model downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from asr_pro.config import settings
from asr_pro.services.crosstalk_resolution_service import (
    SAMPLE_RATE,
    _extract_window,
    resolve_crosstalk_events,
)


def test_resolve_crosstalk_events_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", False)
    events = [{"start": 1.0, "end": 2.0, "duration": 1.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    result = resolve_crosstalk_events("nonexistent.wav", events)
    assert result == events
    assert "separated_transcripts" not in result[0]


def test_resolve_crosstalk_events_noop_for_empty_events():
    result = resolve_crosstalk_events("nonexistent.wav", [])
    assert result == []


def test_resolve_crosstalk_events_skips_short_events(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", True)
    monkeypatch.setattr(settings, "crosstalk_separation_min_duration_sec", 0.5)
    events = [{"start": 1.0, "end": 1.2, "duration": 0.2, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    result = resolve_crosstalk_events("nonexistent.wav", events)
    assert "separated_transcripts" not in result[0]


def test_resolve_crosstalk_events_handles_missing_audio_gracefully(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", True)
    monkeypatch.setattr(settings, "crosstalk_separation_min_duration_sec", 0.1)
    events = [{"start": 1.0, "end": 2.0, "duration": 1.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    # Audio file doesn't exist - must not raise, must return events unresolved.
    result = resolve_crosstalk_events("/nonexistent/path/call.wav", events)
    assert "separated_transcripts" not in result[0]


def test_extract_window_returns_the_correct_slice():
    audio = np.arange(SAMPLE_RATE * 2, dtype=np.float32)  # 2 seconds of samples
    window = _extract_window(audio, 0.5, 1.0)
    assert len(window) == SAMPLE_RATE // 2
    assert window[0] == SAMPLE_RATE // 2


def test_extract_window_returns_empty_when_end_before_start():
    audio = np.arange(SAMPLE_RATE, dtype=np.float32)
    window = _extract_window(audio, 1.0, 0.5)
    assert window.size == 0


def test_resolve_crosstalk_events_full_success_path(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", True)
    monkeypatch.setattr(settings, "crosstalk_separation_min_duration_sec", 0.1)
    events = [{"start": 0.0, "end": 2.0, "duration": 2.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]

    fake_audio = np.ones(SAMPLE_RATE * 2, dtype=np.float32)
    stream_a = np.ones(SAMPLE_RATE, dtype=np.float32)
    stream_b = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.5

    mock_asr = MagicMock()
    mock_asr.transcribe_array.side_effect = [
        ([MagicMock(text="merhaba")], 1.0),
        ([MagicMock(text="nasılsın")], 1.0),
    ]

    with (
        patch("faster_whisper.decode_audio", return_value=fake_audio),
        patch("asr_pro.services.asr_service.ASRService.get_instance", return_value=mock_asr),
        patch(
            "asr_pro.services.audio_conditioning.condition_telephony_audio",
            side_effect=lambda w, sample_rate: w,
        ),
        patch(
            "asr_pro.services.speech_separation_service.separate_two_speakers",
            return_value=[stream_a, stream_b],
        ),
    ):
        result = resolve_crosstalk_events("fake.wav", events)

    assert result[0]["separated_transcripts"] == [
        {"speaker": "SPEAKER_00", "text": "merhaba"},
        {"speaker": "SPEAKER_01", "text": "nasılsın"},
    ]


def test_resolve_crosstalk_events_skips_when_separation_returns_nothing(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", True)
    monkeypatch.setattr(settings, "crosstalk_separation_min_duration_sec", 0.1)
    events = [{"start": 0.0, "end": 2.0, "duration": 2.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    fake_audio = np.ones(SAMPLE_RATE * 2, dtype=np.float32)

    with (
        patch("faster_whisper.decode_audio", return_value=fake_audio),
        patch("asr_pro.services.asr_service.ASRService.get_instance", return_value=MagicMock()),
        patch(
            "asr_pro.services.audio_conditioning.condition_telephony_audio",
            side_effect=lambda w, sample_rate: w,
        ),
        patch("asr_pro.services.speech_separation_service.separate_two_speakers", return_value=[]),
    ):
        result = resolve_crosstalk_events("fake.wav", events)

    assert "separated_transcripts" not in result[0]


def test_resolve_crosstalk_events_handles_transcription_failure(monkeypatch):
    monkeypatch.setattr(settings, "crosstalk_separation_enabled", True)
    monkeypatch.setattr(settings, "crosstalk_separation_min_duration_sec", 0.1)
    events = [{"start": 0.0, "end": 2.0, "duration": 2.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    fake_audio = np.ones(SAMPLE_RATE * 2, dtype=np.float32)
    stream_a = np.ones(SAMPLE_RATE, dtype=np.float32)

    mock_asr = MagicMock()
    mock_asr.transcribe_array.side_effect = RuntimeError("model unavailable")

    with (
        patch("faster_whisper.decode_audio", return_value=fake_audio),
        patch("asr_pro.services.asr_service.ASRService.get_instance", return_value=mock_asr),
        patch(
            "asr_pro.services.audio_conditioning.condition_telephony_audio",
            side_effect=lambda w, sample_rate: w,
        ),
        patch(
            "asr_pro.services.speech_separation_service.separate_two_speakers",
            return_value=[stream_a],
        ),
    ):
        result = resolve_crosstalk_events("fake.wav", events)

    # Transcription failed for the only stream -> nothing resolved, no crash.
    assert "separated_transcripts" not in result[0]
