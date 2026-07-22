from __future__ import annotations

"""Unit tests for Silero VAD Service."""

from unittest.mock import MagicMock, patch

from asr_pro.services import vad_service
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


def test_vad_short_audio_is_never_speech():
    vad = VADService()
    assert vad.is_speech(b"") is False
    assert vad.is_speech(b"\x01" * 50) is False


def test_set_vad_parameters_updates_state():
    vad = VADService()
    vad.set_vad_parameters(threshold=0.7, min_speech_duration_ms=300, min_silence_duration_ms=600)
    params = vad.get_vad_parameters()
    assert params == {
        "threshold": 0.7,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 600,
    }


def test_load_model_success_sets_loaded_true():
    vad = VADService()
    mock_model = MagicMock()
    mock_utils = ["utils"]
    with patch.object(vad_service.torch.hub, "load", return_value=(mock_model, mock_utils)):
        vad._load_model()
    assert vad.loaded is True
    assert vad.model is mock_model
    assert vad.utils is mock_utils


def test_load_model_failure_keeps_loaded_false():
    vad = VADService()
    with patch.object(vad_service.torch.hub, "load", side_effect=RuntimeError("no network")):
        vad._load_model()
    assert vad.loaded is False
    assert vad.model is None


def test_is_speech_uses_neural_model_when_loaded():
    vad = VADService()
    vad.loaded = True
    vad.model = MagicMock(return_value=MagicMock(item=lambda: 0.9))
    audio_bytes = b"\x10\x00" * 1024  # >=512 int16 samples of non-zero PCM
    assert vad.is_speech(audio_bytes) is True

    vad.model = MagicMock(return_value=MagicMock(item=lambda: 0.1))
    assert vad.is_speech(audio_bytes) is False


def test_is_speech_falls_back_to_energy_ratio_on_model_exception():
    vad = VADService()
    vad.loaded = True
    vad.model = MagicMock(side_effect=RuntimeError("bad tensor"))
    active_chunk = bytes(x % 256 for x in range(4096))
    assert vad.is_speech(active_chunk) is True


def test_filter_speech_timestamps_returns_empty_when_not_loaded():
    vad = VADService()
    vad.loaded = False
    assert vad.filter_speech_timestamps(audio=object()) == []


def test_filter_speech_timestamps_delegates_to_silero_utils():
    vad = VADService()
    vad.loaded = True
    vad.model = MagicMock()
    expected = [{"start": 0, "end": 100}]
    get_speech_timestamps_fn = MagicMock(return_value=expected)
    vad.utils = [get_speech_timestamps_fn]

    result = vad.filter_speech_timestamps(audio="fake-audio", sampling_rate=16000)

    assert result == expected
    get_speech_timestamps_fn.assert_called_once()


def test_filter_speech_timestamps_returns_empty_on_exception():
    vad = VADService()
    vad.loaded = True
    vad.model = MagicMock()
    vad.utils = [MagicMock(side_effect=RuntimeError("boom"))]
    assert vad.filter_speech_timestamps(audio="fake-audio") == []
