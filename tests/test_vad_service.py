from __future__ import annotations

"""Unit tests for Silero VAD Service."""

from asr_pro.services.vad_service import DEFAULT_VAD_PARAMETERS, VADService


def test_vad_service_singleton():
    v1 = VADService.get_instance()
    v2 = VADService.get_instance()
    assert v1 is v2


def test_vad_silence_detection():
    vad = VADService.get_instance()
    silent_chunk = b"\x00" * 4096
    assert vad.is_speech(silent_chunk) is False


def test_vad_active_speech_detection():
    vad = VADService.get_instance()
    # High entropy active audio simulation
    active_chunk = bytes(x % 256 for x in range(4096))
    assert vad.is_speech(active_chunk) is True


def test_vad_tightened_parameters():
    vad = VADService.get_instance()
    params = vad.get_vad_parameters()
    assert params["threshold"] == 0.5
    assert params["min_speech_duration_ms"] == 250
    assert params["min_silence_duration_ms"] == 500
    assert DEFAULT_VAD_PARAMETERS["threshold"] == 0.5
