"""Tests for BiometricService voiceprint enrollment/matching and the /agents API contract.

In test mode (ASR_TEST_NO_MODEL=1) the ECAPA-TDNN encoder is not loaded, so
these exercise the legacy fft-legacy-v1 fallback path deterministically - the
same code path a production deployment falls back to if speechbrain/the
pretrained model is ever unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from asr_pro.services import biometric_service as biometric_module
from asr_pro.services.biometric_service import (
    ECAPA_MODEL_NAME,
    LEGACY_MODEL_NAME,
    BiometricService,
    _EcapaEncoder,
)


def _tone(freq_hz: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) + 0.5 * np.sin(2 * np.pi * freq_hz * 3.5 * t)).astype(
        np.float32
    )


def test_enroll_and_match_speaker(db_session):
    service = BiometricService(db_session=db_session)
    audio_agent_a = _tone(400.0)

    record = service.enroll_agent("AG-1001", "Ayşe Yılmaz", audio_agent_a)
    assert record is not None
    assert record.agent_code == "AG-1001"
    assert record.embedding_model == "fft-legacy-v1"
    assert len(record.embedding_json) == 128

    matched_code, score, matched_name = service.match_speaker(audio_agent_a, threshold=0.9)
    assert matched_code == "AG-1001"
    assert matched_name == "Ayşe Yılmaz"
    assert score > 0.9


def test_match_speaker_no_match_below_threshold(db_session):
    service = BiometricService(db_session=db_session)
    service.enroll_agent("AG-1002", "Mehmet Kaya", _tone(400.0))

    unrelated_audio = _tone(2000.0)
    matched_code, score, matched_name = service.match_speaker(unrelated_audio, threshold=0.9)
    assert matched_code is None
    assert matched_name is None


def test_list_voiceprints_returns_enrolled_agents(db_session):
    service = BiometricService(db_session=db_session)
    service.enroll_agent("AG-2001", "Test Agent", _tone(500.0))

    voiceprints = service.list_voiceprints()
    codes = [v.agent_code for v in voiceprints]
    assert "AG-2001" in codes


def _get_admin_token(client: TestClient) -> str:
    response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_agents_api_enroll_then_list(client: TestClient, tmp_path):
    import soundfile as sf

    token = _get_admin_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    wav_path = tmp_path / "agent_sample.wav"
    sf.write(str(wav_path), _tone(450.0), 16000)

    with open(wav_path, "rb") as f:
        response = client.post(
            "/api/v1/agents/voiceprints/enroll",
            params={"agent_code": "AG-API-01", "agent_name": "API Test Agent"},
            files={"file": ("agent_sample.wav", f, "audio/wav")},
            headers=headers,
        )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["embedding_dim"] == 128
    assert body["embedding_model"] == "fft-legacy-v1"

    list_response = client.get("/api/v1/agents/voiceprints", headers=headers)
    assert list_response.status_code == 200
    voiceprints = list_response.json()
    assert any(v["agent_code"] == "AG-API-01" for v in voiceprints)
    matched = next(v for v in voiceprints if v["agent_code"] == "AG-API-01")
    assert matched["embedding_dim"] == 128
    assert matched["embedding_model"] == "fft-legacy-v1"


@pytest.fixture(autouse=True)
def _reset_ecapa_singleton():
    original_instance = _EcapaEncoder._instance
    original_attempted = _EcapaEncoder._load_attempted
    _EcapaEncoder._instance = None
    _EcapaEncoder._load_attempted = False
    yield
    _EcapaEncoder._instance = original_instance
    _EcapaEncoder._load_attempted = original_attempted


def test_ecapa_encoder_skips_loading_under_test_mode():
    assert _EcapaEncoder.get() is None


def test_ecapa_encoder_caches_instance():
    sentinel = object()
    _EcapaEncoder._instance = sentinel
    _EcapaEncoder._load_attempted = True
    assert _EcapaEncoder.get() is sentinel


def test_ecapa_encoder_loads_successfully_outside_test_mode():
    mock_encoder = MagicMock()
    with (
        patch.object(biometric_module, "_is_testing", False),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "speechbrain.inference.speaker.EncoderClassifier.from_hparams",
            return_value=mock_encoder,
        ),
    ):
        assert _EcapaEncoder.get() is mock_encoder


def test_ecapa_encoder_falls_back_to_none_on_load_failure():
    with (
        patch.object(biometric_module, "_is_testing", False),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "speechbrain.inference.speaker.EncoderClassifier.from_hparams",
            side_effect=RuntimeError("offline"),
        ),
    ):
        assert _EcapaEncoder.get() is None


def test_decode_returns_empty_array_for_unsupported_type():
    result = BiometricService._decode(12345, sample_rate=16000)
    assert result.size == 0


def test_extract_voiceprint_returns_legacy_zero_vector_for_empty_audio():
    embedding, model = BiometricService.extract_voiceprint(np.array([], dtype=np.float32))
    assert embedding == [0.0] * 128
    assert model == LEGACY_MODEL_NAME


def test_extract_voiceprint_uses_ecapa_when_encoder_available():
    fake_embedding = MagicMock()
    fake_embedding.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones(
        192, dtype=np.float32
    )
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.return_value = fake_embedding

    with patch.object(_EcapaEncoder, "get", return_value=mock_encoder):
        embedding, model = BiometricService.extract_voiceprint(_tone(300.0))

    assert model == ECAPA_MODEL_NAME
    assert len(embedding) == 192


def test_extract_voiceprint_falls_back_to_legacy_on_ecapa_inference_failure():
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.side_effect = RuntimeError("bad tensor")

    with patch.object(_EcapaEncoder, "get", return_value=mock_encoder):
        embedding, model = BiometricService.extract_voiceprint(_tone(300.0))

    assert model == LEGACY_MODEL_NAME
    assert len(embedding) == 128


def test_legacy_fft_pads_short_audio():
    short_audio = np.ones(10, dtype=np.float32)
    embedding = BiometricService._extract_legacy_fft_voiceprint(short_audio)
    assert len(embedding) == 128


def test_legacy_fft_returns_zero_vector_for_silence():
    silence = np.zeros(16000, dtype=np.float32)
    embedding = BiometricService._extract_legacy_fft_voiceprint(silence)
    assert embedding == [0.0] * 128


def test_cosine_similarity_mismatched_dimensions_returns_zero():
    assert BiometricService.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert BiometricService.cosine_similarity([], []) == 0.0


def test_cosine_similarity_zero_vector_returns_zero():
    assert BiometricService.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_enroll_agent_returns_none_for_silent_audio(db_session):
    service = BiometricService(db_session=db_session)
    silence = np.zeros(16000, dtype=np.float32)
    assert service.enroll_agent("AG-SILENT", "Silent Agent", silence) is None


def test_enroll_agent_updates_existing_record(db_session):
    service = BiometricService(db_session=db_session)
    first = service.enroll_agent("AG-3001", "Original Name", _tone(350.0))
    assert first is not None

    second = service.enroll_agent("AG-3001", "Updated Name", _tone(600.0))
    assert second.id == first.id
    assert second.agent_name == "Updated Name"

    all_records = service.list_voiceprints()
    assert sum(1 for r in all_records if r.agent_code == "AG-3001") == 1


def test_enroll_agent_rolls_back_and_reraises_on_db_error(db_session):
    service = BiometricService(db_session=db_session)
    with patch.object(db_session, "commit", side_effect=RuntimeError("db exploded")):
        with pytest.raises(RuntimeError, match="db exploded"):
            service.enroll_agent("AG-4001", "Broken Agent", _tone(300.0))


def test_enroll_and_list_use_internal_session_when_none_provided(setup_db):
    # Uses its own internally-opened session (self._db is None), which commits
    # directly against the shared test DB rather than the per-test rollback
    # transaction `db_session` provides - clean up explicitly so this record
    # doesn't leak into other tests' match_speaker() candidate pool.
    service = BiometricService()
    try:
        record = service.enroll_agent("AG-5001", "Internal Session Agent", _tone(700.0))
        assert record is not None

        voiceprints = service.list_voiceprints()
        assert any(v.agent_code == "AG-5001" for v in voiceprints)
    finally:
        from asr_pro.db.models import AgentVoiceprint
        from asr_pro.db.session import SessionLocal

        cleanup = SessionLocal()
        cleanup.query(AgentVoiceprint).filter_by(agent_code="AG-5001").delete()
        cleanup.commit()
        cleanup.close()


def test_match_speaker_skips_mismatched_embedding_model(db_session):
    service = BiometricService(db_session=db_session)
    service.enroll_agent("AG-6001", "ECAPA Agent", _tone(400.0))
    # Force the enrolled record to look like it was produced by the ECAPA model.
    from asr_pro.db.models import AgentVoiceprint

    rec = db_session.query(AgentVoiceprint).filter_by(agent_code="AG-6001").first()
    rec.embedding_model = ECAPA_MODEL_NAME
    db_session.commit()

    # Query embedding is extracted via the legacy path (test mode) -> model mismatch, skipped.
    matched_code, score, matched_name = service.match_speaker(_tone(400.0), threshold=0.5)
    assert matched_code is None


def test_match_speaker_ignores_records_without_embedding(db_session):
    service = BiometricService(db_session=db_session)
    record = service.enroll_agent("AG-7001", "No Embedding Agent", _tone(400.0))
    record.embedding_json = None
    db_session.commit()

    matched_code, score, matched_name = service.match_speaker(_tone(400.0), threshold=0.1)
    assert matched_code is None


def test_match_speaker_uses_configured_default_threshold(db_session):
    service = BiometricService(db_session=db_session)
    service.enroll_agent("AG-8001", "Default Threshold Agent", _tone(400.0))
    with patch.object(biometric_module.settings, "biometric_match_threshold", 0.0):
        matched_code, score, matched_name = service.match_speaker(_tone(400.0))
    assert matched_code == "AG-8001"
