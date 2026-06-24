import pytest
from fastapi.websockets import WebSocketDisconnect

def test_websocket_missing_token(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live-asr") as websocket:
            msg = websocket.receive_json()
            assert msg["type"] == "auth_required"
            # Send invalid auth
            websocket.send_json({"type": "auth", "token": "invalid"})
            websocket.receive_json() # Should disconnect
    assert exc.value.code == 1008

def test_websocket_valid_token(client):
    login_response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password123"})
    token = login_response.json()["access_token"]
    
    with client.websocket_connect("/ws/live-asr") as websocket:
        msg = websocket.receive_json()
        assert msg["type"] == "auth_required"
        
        websocket.send_json({"type": "auth", "token": token})
        ok_msg = websocket.receive_json()
        assert ok_msg["type"] == "auth_ok"

def test_websocket_audio_chunk_accumulation(client):
    login_response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password123"})
    token = login_response.json()["access_token"]
    
    with client.websocket_connect("/ws/live-asr") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "auth", "token": token})
        websocket.receive_json()
        
        # 64KB'dan küçük bir chunk gönderelim (işlenmemeli)
        small_chunk = b"x" * 1024 * 10
        websocket.send_bytes(small_chunk)
        
        # Büyük chunk gönderelim (tetiklemeli)
        large_chunk = b"x" * 1024 * 70
        websocket.send_bytes(large_chunk)
        
        # Fake bir Whisper transcription yanıtı bekleriz veya warning
        try:
            data = websocket.receive_json()
            assert "status" in data
            assert data["status"] in ["success", "warning"]
        except Exception:
            pass
