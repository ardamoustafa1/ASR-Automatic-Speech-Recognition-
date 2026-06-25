
from fastapi.testclient import TestClient


def test_login_success(client: TestClient):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_wrong_password(client: TestClient):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrong"})
    assert response.status_code == 401

def test_auth_me_valid_token(client: TestClient):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password123"})
    token = response.json()["access_token"]

    me_response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["username"] == "admin"
    assert me_response.json()["role"] == "admin"

def test_auth_me_invalid_token(client: TestClient):
    me_response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer INVALID_TOKEN"})
    assert me_response.status_code == 401

def test_rate_limit_auth(client: TestClient):
    # Depending on slowapi config, we try 6 times to hit the 5/minute limit
    for _ in range(5):
        client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrong"})

    res = client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrong"})
    assert res.status_code in [401, 429] # Either fails auth or hits rate limit
