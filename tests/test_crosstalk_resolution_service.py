"""Tests for crosstalk_resolution_service's graceful degradation behavior.

The real separation+re-transcription quality is covered by
tests/test_speech_separation_service.py; this file verifies the orchestration
logic (filtering, config gating, error handling) never raises and leaves
events untouched when separation isn't available - the path every call hits
in test mode / without the SepFormer model downloaded.
"""

from __future__ import annotations

from asr_pro.config import settings
from asr_pro.services.crosstalk_resolution_service import resolve_crosstalk_events


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
