"""Tests for GPU compute-capability detection and the CTranslate2 compute-type
fallback chain in ASRService - covers Blackwell-class (compute capability
>=10.x) detection/logging and the load_model() degrade-gracefully behavior
when a compute_type isn't supported by the installed CTranslate2 build.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from asr_pro.services import asr_service as asr_service_module
from asr_pro.services.asr_service import ASRService


def _make_service(device: str) -> ASRService:
    service = ASRService.__new__(ASRService)
    service._device = device
    service._compute_type = "float16"
    return service


def test_gpu_compute_capability_returns_zero_zero_when_torch_unavailable():
    service = _make_service("cuda")
    with patch("torch.cuda.get_device_capability", side_effect=RuntimeError("no GPU")):
        assert service._gpu_compute_capability() == (0, 0)


def test_gpu_compute_capability_returns_detected_value():
    service = _make_service("cuda")
    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        assert service._gpu_compute_capability() == (9, 0)


def test_choose_compute_type_cpu_uses_int8():
    service = _make_service("cpu")
    assert service._choose_compute_type() == "int8"


def test_choose_compute_type_mps_uses_float16():
    service = _make_service("mps")
    assert service._choose_compute_type() == "float16"


def test_choose_cuda_compute_type_defaults_to_float16_on_older_gpu():
    service = _make_service("cuda")
    with patch.object(ASRService, "_gpu_compute_capability", return_value=(8, 6)):
        assert service._choose_cuda_compute_type() == "float16"


def test_choose_cuda_compute_type_logs_blackwell_detection():
    service = _make_service("cuda")
    with (
        patch.object(ASRService, "_gpu_compute_capability", return_value=(10, 0)),
        patch.object(asr_service_module.logger, "info") as mock_log,
    ):
        result = service._choose_cuda_compute_type()

    assert result == "float16"
    assert any("Blackwell" in str(call.args[0]) for call in mock_log.call_args_list)


def test_choose_cuda_compute_type_detects_rtx_50_series_capability():
    service = _make_service("cuda")
    with (
        patch.object(ASRService, "_gpu_compute_capability", return_value=(12, 0)),
        patch.object(asr_service_module.logger, "info") as mock_log,
    ):
        result = service._choose_cuda_compute_type()

    assert result == "float16"
    assert any("Blackwell" in str(call.args[0]) for call in mock_log.call_args_list)


def test_compute_type_fallback_chain_on_cpu_is_just_the_selected_type():
    service = _make_service("cpu")
    service._compute_type = "int8"
    assert service._compute_type_fallback_chain() == ["int8"]


def test_compute_type_fallback_chain_on_cuda_tries_progressively_cheaper_types():
    service = _make_service("cuda")
    service._compute_type = "float16"
    chain = service._compute_type_fallback_chain()
    assert chain[0] == "float16"
    assert chain == ["float16", "int8_float16", "int8"]
    assert len(chain) == len(set(chain))  # no duplicates


def test_compute_type_fallback_chain_preserves_preferred_type_first():
    service = _make_service("cuda")
    service._compute_type = "int8_float16"
    chain = service._compute_type_fallback_chain()
    assert chain[0] == "int8_float16"
    assert chain.count("int8_float16") == 1


@patch("asr_pro.services.asr_service.platform.system", return_value="Linux")
@patch("asr_pro.services.asr_service.WhisperModel")
def test_load_model_falls_back_when_preferred_compute_type_unsupported(
    mock_whisper_cls, _mock_system
):
    """Simulates an installed CTranslate2 build that raises on float16 for a
    brand-new GPU architecture (e.g. Blackwell before kernel support lands) -
    load_model() must not crash, it must fall through to a working type.
    """
    service = ASRService.__new__(ASRService)
    service._device = "cuda"
    service._compute_type = "float16"
    service._model = None
    service._model_size = "large-v3"
    service._is_mlx = False

    mock_model_instance = MagicMock()
    mock_whisper_cls.side_effect = [
        RuntimeError("no kernel for this compute type/architecture combination"),
        mock_model_instance,
    ]

    result = service.load_model("large-v3")

    assert result is mock_model_instance
    assert service._compute_type == "int8_float16"
    assert mock_whisper_cls.call_count == 2


@patch("asr_pro.services.asr_service.platform.system", return_value="Linux")
@patch("asr_pro.services.asr_service.WhisperModel")
def test_load_model_raises_informatively_when_every_compute_type_fails(
    mock_whisper_cls, _mock_system
):
    service = ASRService.__new__(ASRService)
    service._device = "cuda"
    service._compute_type = "float16"
    service._model = None
    service._model_size = "large-v3"
    service._is_mlx = False

    mock_whisper_cls.side_effect = RuntimeError("totally unsupported GPU")

    with pytest.raises(RuntimeError, match="Could not load ASR model"):
        service.load_model("large-v3")

    assert mock_whisper_cls.call_count == 3  # float16, int8_float16, int8
