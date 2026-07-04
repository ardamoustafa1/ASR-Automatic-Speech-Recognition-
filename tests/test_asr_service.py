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
