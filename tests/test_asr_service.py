import os
from unittest.mock import MagicMock, patch

import pytest

from asr_pro.services.asr_service import ASRService


@pytest.mark.skipif(os.environ.get("ASR_TEST_NO_MODEL") == "1", reason="No model tests")
@patch("asr_pro.services.asr_service.platform.system")
@patch("asr_pro.services.asr_service.platform.machine")
def test_choose_device_and_compute(mock_machine, mock_system):
    # Test Apple Silicon
    mock_system.return_value = "Darwin"
    mock_machine.return_value = "arm64"

    svc = ASRService()
    # It should set to cpu for non-mlx paths or mps if we use default logic
    assert svc._choose_device() in ["cpu", "mps", "cuda"]


@pytest.mark.skipif(os.environ.get("ASR_TEST_NO_MODEL") == "1", reason="No model tests")
def test_check_cuda_failure():
    with patch("builtins.__import__", side_effect=ImportError("No torch")):
        svc = ASRService()
        assert svc._check_cuda() is False


@pytest.mark.skipif(os.environ.get("ASR_TEST_NO_MODEL") == "1", reason="No model tests")
@patch("asr_pro.services.asr_service.WhisperModel")
def test_transcribe(mock_whisper):
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "hello world"
    mock_info = MagicMock()
    mock_info.duration = 1.0
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    mock_whisper.return_value = mock_model

    svc = ASRService()
    svc._model = mock_model
    svc._is_mlx = False
    segments, duration = svc.transcribe("fake_path.wav")
    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert duration == 1.0


def test_sanitize_text():
    assert ASRService._sanitize_text("Efendim? Efendim?") == "Efendim?"
    assert ASRService._sanitize_text("evet evet evet") == "evet"
    assert ASRService._sanitize_text("normal cümle burada.") == "normal cümle burada."


@patch("asr_pro.services.asr_service.ASRService._is_stereo_file", return_value=True)
@patch("faster_whisper.decode_audio")
def test_transcribe_stereo(mock_decode, mock_is_stereo):
    import numpy as np

    from asr_pro.services.asr_service import TranscriptionSegment

    mock_decode.return_value = (np.ones(1600, dtype=np.float32), np.ones(1600, dtype=np.float32))

    svc = ASRService()
    svc._model = MagicMock()
    svc._is_mlx = False

    call_count = [0]

    def mock_single_ch(audio, language="tr", sector="telecom"):
        call_count[0] += 1
        if call_count[0] == 1:
            return [TranscriptionSegment(start=0.0, end=1.0, text="Merhaba agent")], 1.0
        return [TranscriptionSegment(start=0.5, end=1.5, text="Merhaba customer")], 1.5

    with patch.object(svc, "_transcribe_single_channel", side_effect=mock_single_ch):
        segments, duration = svc.transcribe("stereo.wav")
        assert len(segments) == 2
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[0].text == "Merhaba agent"
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[1].text == "Merhaba customer"
        assert duration == 1.5
