from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.websockets import WebSocketDisconnect


def test_websocket_missing_token(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-asr") as websocket:
            msg = websocket.receive_json()
            assert msg["type"] == "auth_required"
            # Send invalid auth
            websocket.send_json({"type": "auth", "token": "invalid"})
            websocket.receive_json()  # Should disconnect
    assert exc.value.code == 1008


def test_websocket_valid_token(client):
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    with client.websocket_connect("/ws/live-asr") as websocket:
        msg = websocket.receive_json()
        assert msg["type"] == "auth_required"

        websocket.send_json({"type": "auth", "token": token})
        ok_msg = websocket.receive_json()
        assert ok_msg["type"] == "auth_ok"


def _make_mock_session(push_audio_results):
    mock_session = MagicMock()
    mock_session.start = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.committed_offset_sec = 1.2
    mock_session.push_audio = AsyncMock(side_effect=push_audio_results)
    mock_session.flush_final = AsyncMock(return_value=None)
    return mock_session


@patch("asr_pro.api.routes.websocket.StreamingASRSession")
def test_websocket_partial_then_final_flow(mock_session_cls, client):
    """New streaming protocol: chunk-batch re-transcription is replaced by
    incremental partial/final messages driven by StreamingASRSession."""
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    mock_session_cls.return_value = _make_mock_session(
        [
            {"type": "partial", "text": "merhaba", "segments": []},
            {
                "type": "final",
                "text": "merhaba dünya",
                "segments": [{"start": 0.0, "end": 1.2, "text": "merhaba dünya", "speaker": None}],
                "transcript_so_far": "merhaba dünya",
            },
        ]
    )

    with client.websocket_connect("/ws/live-asr") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "auth", "token": token})
        websocket.receive_json()

        websocket.send_bytes(b"chunk-1")
        partial_msg = websocket.receive_json()
        assert partial_msg["type"] == "partial"
        assert partial_msg["text"] == "merhaba"
        assert "latency_ms" in partial_msg

        websocket.send_bytes(b"chunk-2")
        final_msg = websocket.receive_json()
        assert final_msg["type"] == "final"
        assert final_msg["transcript_so_far"] == "merhaba dünya"


@patch("asr_pro.api.routes.websocket.StreamingASRSession")
def test_websocket_decode_error_sent_to_client(mock_session_cls, client):
    from asr_pro.services.audio_stream_decoder import AudioDecodeError

    login_response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    mock_session = MagicMock()
    mock_session.start = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.committed_offset_sec = 0.0
    mock_session.push_audio = AsyncMock(side_effect=AudioDecodeError("boom"))
    mock_session.flush_final = AsyncMock(return_value=None)
    mock_session_cls.return_value = mock_session

    with client.websocket_connect("/ws/live-asr") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "auth", "token": token})
        websocket.receive_json()

        websocket.send_bytes(b"not-real-audio")
        err_msg = websocket.receive_json()
        assert err_msg["type"] == "error"


def test_websocket_connection_limit_rejected(client, monkeypatch):
    import asr_pro.api.routes.websocket as ws_module

    monkeypatch.setattr(ws_module, "MAX_WS_CONNECTIONS", 0)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-asr") as websocket:
            websocket.receive_json()
    assert exc.value.code == 1013
