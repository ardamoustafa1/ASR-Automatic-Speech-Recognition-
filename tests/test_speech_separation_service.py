"""Tests for SepFormer-based crosstalk speech separation.

The full-quality functional test (separation actually isolates two distinct
voices) requires the real SepFormer checkpoint and is skipped under
ASR_TEST_NO_MODEL=1, matching this repo's convention for HuggingFace-model-
gated tests (see tests/test_churn_engine.py). The graceful-fallback behavior
(model unavailable -> None, never an exception) is always tested since that's
exactly the path production traffic hits without network/model access.
"""

from __future__ import annotations

import importlib.util
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

MODEL_AVAILABLE = (
    os.environ.get("ASR_TEST_NO_MODEL") != "1"
    and importlib.util.find_spec("speechbrain") is not None
)

from asr_pro.services.speech_separation_service import (
    _SepformerSeparator,
    separate_two_speakers,
)


@pytest.fixture(autouse=True)
def _reset_separator_singleton():
    """The lazy singleton's state is class-level and must not leak across tests."""
    original_instance = _SepformerSeparator._instance
    original_attempted = _SepformerSeparator._load_attempted
    _SepformerSeparator._instance = None
    _SepformerSeparator._load_attempted = False
    yield
    _SepformerSeparator._instance = original_instance
    _SepformerSeparator._load_attempted = original_attempted


def test_separate_two_speakers_empty_audio_returns_none():
    assert separate_two_speakers(np.array([], dtype=np.float32)) is None


def test_separate_two_speakers_rejects_wrong_sample_rate():
    audio = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError, match="16000Hz"):
        separate_two_speakers(audio, sample_rate=8000)


def test_separator_get_skips_loading_under_test_mode():
    assert _SepformerSeparator.get() is None
    # Second call must not re-attempt the load - the attempt flag latches.
    with patch("asr_pro.services.speech_separation_service._is_testing", False):
        assert _SepformerSeparator.get() is None


def test_separator_get_caches_instance_across_calls():
    sentinel = object()
    _SepformerSeparator._instance = sentinel
    _SepformerSeparator._load_attempted = True
    assert _SepformerSeparator.get() is sentinel


def test_separator_get_loads_model_successfully_outside_test_mode():
    mock_model = MagicMock()
    with (
        patch("asr_pro.services.speech_separation_service._is_testing", False),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "speechbrain.inference.separation.SepformerSeparation.from_hparams",
            return_value=mock_model,
        ),
    ):
        result = _SepformerSeparator.get()
    assert result is mock_model


def test_separator_get_returns_none_when_model_load_fails():
    with (
        patch("asr_pro.services.speech_separation_service._is_testing", False),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "speechbrain.inference.separation.SepformerSeparation.from_hparams",
            side_effect=RuntimeError("network unreachable"),
        ),
    ):
        result = _SepformerSeparator.get()
    assert result is None


def test_separate_two_speakers_returns_none_when_model_unavailable():
    audio = np.zeros(1600, dtype=np.float32)
    with patch.object(_SepformerSeparator, "get", return_value=None):
        assert separate_two_speakers(audio) is None


def test_separate_two_speakers_returns_streams_on_success():
    audio = np.random.rand(1600).astype(np.float32)
    mock_model = MagicMock()

    # Simulate SepFormer's (1, time, n_sources) output tensor using numpy,
    # since the real return type only needs to support the same slicing +
    # .detach().cpu().numpy() chain the production code calls.
    import torch as real_torch

    fake_estimated = real_torch.zeros((1, len(audio), 2))
    mock_model.separate_batch.return_value = fake_estimated

    with patch.object(_SepformerSeparator, "get", return_value=mock_model):
        streams = separate_two_speakers(audio)

    assert streams is not None
    assert len(streams) == 2
    assert all(isinstance(s, np.ndarray) for s in streams)


def test_separate_two_speakers_returns_none_on_inference_failure():
    audio = np.random.rand(1600).astype(np.float32)
    mock_model = MagicMock()
    mock_model.separate_batch.side_effect = RuntimeError("inference blew up")

    with patch.object(_SepformerSeparator, "get", return_value=mock_model):
        assert separate_two_speakers(audio) is None


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_separate_two_speakers_isolates_real_voices(monkeypatch):
    """Mix two acoustically-distinct real voices and verify separation
    actually pulls them apart, using ECAPA-TDNN cosine similarity as the
    correctness signal (same technique used to validate the synthetic
    diarization benchmark in docs/DIARIZATION_DER_WORKFLOW.md).

    _SepformerSeparator normally refuses to load real models under pytest
    (asr_pro.config._is_testing is True whenever "pytest" is in sys.modules,
    by design - production tests must never depend on network/model access
    at collection time). This test explicitly opts in by pre-populating the
    singleton with a real loaded model, bypassing that guard only for this
    one deliberately-gated test.
    """
    import soundfile as sf
    from scipy.signal import resample
    from speechbrain.inference.separation import SepformerSeparation

    from asr_pro.services.biometric_service import BiometricService

    real_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k", savedir="data/models/sepformer-whamr16k"
    )
    monkeypatch.setattr(_SepformerSeparator, "_instance", real_model)
    monkeypatch.setattr(_SepformerSeparator, "_load_attempted", True)

    def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
        factor = 2.0 ** (semitones / 12.0)
        n_new = max(1, int(len(audio) / factor))
        return resample(audio, n_new).astype(np.float32)

    a, sr_a = sf.read("benchmarks/audio/call_001_banking_kmh.wav", dtype="float32")
    b, sr_b = sf.read("benchmarks/audio/call_005_insurance_claim.wav", dtype="float32")
    assert sr_a == sr_b == 16000
    b = pitch_shift(b, -6.0)  # same narrator as `a` - shift for genuine acoustic separation

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mix = (a + b) * 0.5

    streams = separate_two_speakers(mix, sample_rate=16000)
    assert streams is not None
    assert len(streams) == 2

    emb_a, _ = BiometricService.extract_voiceprint(a, sample_rate=16000)
    emb_b, _ = BiometricService.extract_voiceprint(b, sample_rate=16000)
    embs = [BiometricService.extract_voiceprint(s, sample_rate=16000)[0] for s in streams]

    sims_to_a = [BiometricService.cosine_similarity(e, emb_a) for e in embs]
    sims_to_b = [BiometricService.cosine_similarity(e, emb_b) for e in embs]

    # At least one stream should clearly resemble A more than B, and the
    # other clearly resemble B more than A - i.e. separation pulled the two
    # voices apart into distinct streams rather than just returning two
    # copies of the mixture.
    best_a_idx = int(np.argmax(sims_to_a))
    best_b_idx = int(np.argmax(sims_to_b))
    assert best_a_idx != best_b_idx
