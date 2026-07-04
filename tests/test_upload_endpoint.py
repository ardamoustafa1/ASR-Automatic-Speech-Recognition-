from __future__ import annotations

"""Unit test for REST API audio upload endpoint."""

import io


def test_upload_audio_file(client):
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "agent", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    dummy_audio = io.BytesIO(b"RIFFdummyWAVEfmt dummydata")
    dummy_audio.name = "test_call.wav"

    response = client.post(
        "/api/v1/conversations/upload?sector=omni",
        files={"file": ("test_call.wav", dummy_audio, "audio/wav")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "processing"
    assert data["filename"] == "test_call.wav"
