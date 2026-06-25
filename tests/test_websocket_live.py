import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from asr_pro.api.main import app
from starlette.websockets import WebSocketDisconnect
import time

client = TestClient(app)

def test_websocket_auth_failure():
    # Attempt to connect without valid token
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/v1/ws/live?token=invalid_token") as websocket:
            pass
    assert exc.value.code in (1000, 1008)

@patch('asr_pro.api.routes.websocket._validate_token')
@patch('asr_pro.api.routes.websocket.ASRService')
def test_websocket_latency_and_reconnect(mock_asr_service_cls, mock_validate):
    mock_validate.return_value = {"sub": "user_123", "role": "admin"}
    
    # Mock the instance
    mock_service_instance = MagicMock()
    mock_asr_service_cls.return_value = mock_service_instance
    
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "Hello world"
    mock_service_instance.transcribe_audio.return_value = ([mock_segment], "Hello world")
    
    start_time = time.time()
    
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/ws/live?token=valid_token") as websocket:
            websocket.send_bytes(b"dummy_audio_bytes")
            data = websocket.receive_json()
            latency = time.time() - start_time
            
            assert "text" in data or "transcript" in data
            assert latency < 2.0
        
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/ws/live?token=valid_token") as websocket2:
            websocket2.send_bytes(b"dummy_audio_bytes_2")
            data = websocket2.receive_json()
            assert "text" in data or "transcript" in data
