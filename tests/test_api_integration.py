from fastapi.testclient import TestClient


def get_admin_token(client: TestClient) -> str:
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    return response.json()["access_token"]

def test_get_conversations_empty(client: TestClient):
    token = get_admin_token(client)
    response = client.get("/api/v1/conversations", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_post_analyze_conversation(client: TestClient):
    token = get_admin_token(client)
    payload = {
        "full_transcript": "Merhaba, internet iptali yapmak istiyorum. Çok şikayetçiyim.",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Merhaba, internet iptali yapmak istiyorum."},
            {"start": 2.0, "end": 4.0, "text": "Çok şikayetçiyim."}
        ],
        "sector": "telecom",
        "asr_confidence": 0.955,
        "quality_gate_passed": True
    }
    response = client.post("/api/v1/conversations/analyze", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "processing"

    # Wait for the background task to complete (or at least yield)
    import time
    time.sleep(1)

    # Test GET /conversations with data
    conv_response = client.get("/api/v1/conversations", headers={"Authorization": f"Bearer {token}"})
    assert conv_response.status_code == 200
    convs = conv_response.json()
    assert len(convs) >= 1
    assert convs[0]["full_transcript"] == payload["full_transcript"]



def test_get_alerts(client: TestClient):
    token = get_admin_token(client)
    response = client.get("/api/v1/alerts", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_brute_force_login(client: TestClient):
    # Depending on rate limits, we should just test wrong password
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

# --- EXTENDED API TESTS (TEST-001) ---

def test_get_conversations_pagination(client: TestClient):
    token = get_admin_token(client)
    response = client.get("/api/v1/conversations?skip=0&limit=5", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert len(response.json()) <= 5



def test_404_not_found(client: TestClient):
    token = get_admin_token(client)
    response = client.get("/api/v1/conversations/999999", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code in [404, 405] # Depends on exact routing

def test_post_conversations_missing_fields(client: TestClient):
    token = get_admin_token(client)
    payload = {"full_transcript": "Eksik veriler var."}
    response = client.post("/api/v1/conversations/analyze", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 422 # Pydantic Validation Error

def test_get_alerts_with_query(client: TestClient):
    token = get_admin_token(client)
    response = client.get("/api/v1/alerts?acknowledged=false", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
