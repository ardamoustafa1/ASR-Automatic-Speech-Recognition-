import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketDisconnect


def test_websocket_auth_failure(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-asr") as websocket:
            msg = websocket.receive_json()
            assert msg["type"] == "auth_required"
            websocket.send_json({"type": "auth", "token": "invalid_token"})
            websocket.receive_json()
    assert exc.value.code in (1000, 1008)


@patch("asr_pro.api.routes.websocket._validate_token")
@patch("asr_pro.api.routes.websocket.StreamingASRSession")
def test_websocket_latency_and_reconnect(mock_session_cls, mock_validate, client):
    mock_validate.return_value = {"sub": "user_123", "role": "admin"}

    def _make_session(*_args, **_kwargs):
        session = MagicMock()
        session.start = AsyncMock()
        session.close = AsyncMock()
        session.committed_offset_sec = 1.0
        session.push_audio = AsyncMock(
            return_value={
                "type": "final",
                "text": "Hello world",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": None}],
                "transcript_so_far": "Hello world",
            }
        )
        session.flush_final = AsyncMock(return_value=None)
        return session

    mock_session_cls.side_effect = _make_session

    for payload in (b"a" * 70 * 1024, b"b" * 70 * 1024):
        start_time = time.time()
        with client.websocket_connect("/ws/live-asr") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "auth", "token": "valid_token"})
            assert websocket.receive_json()["type"] == "auth_ok"
            websocket.send_bytes(payload)
            data = websocket.receive_json()
            latency = time.time() - start_time

            assert data["type"] == "final"
            assert data["text"] == "Hello world"
            assert data["transcript_so_far"] == "Hello world"
            assert latency < 2.0


@patch("asr_pro.api.routes.websocket._validate_token")
@patch("asr_pro.api.routes.websocket.StreamingASRSession")
def test_websocket_connection_limit_returns_1013(
    mock_session_cls, mock_validate, client, monkeypatch
):
    import asr_pro.api.routes.websocket as ws_module

    monkeypatch.setattr(ws_module, "MAX_WS_CONNECTIONS", 1)
    mock_validate.return_value = {"sub": "user_a", "role": "admin"}

    def _make_session(*_args, **_kwargs):
        session = MagicMock()
        session.start = AsyncMock()
        session.close = AsyncMock()
        session.committed_offset_sec = 0.0
        session.push_audio = AsyncMock(return_value=None)
        session.flush_final = AsyncMock(return_value=None)
        return session

    mock_session_cls.side_effect = _make_session

    # Hold the first connection open while a second one is attempted.
    with client.websocket_connect("/ws/live-asr") as first_ws:
        first_ws.receive_json()
        first_ws.send_json({"type": "auth", "token": "valid_token"})
        assert first_ws.receive_json()["type"] == "auth_ok"

        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws/live-asr") as second_ws:
                second_ws.receive_json()
        assert exc.value.code == 1013
