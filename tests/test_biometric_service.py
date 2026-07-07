"""Tests for BiometricService voiceprint enrollment/matching and the /agents API contract.

In test mode (ASR_TEST_NO_MODEL=1) the ECAPA-TDNN encoder is not loaded, so
these exercise the legacy fft-legacy-v1 fallback path deterministically - the
same code path a production deployment falls back to if speechbrain/the
pretrained model is ever unavailable.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from asr_pro.services.biometric_service import BiometricService


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
