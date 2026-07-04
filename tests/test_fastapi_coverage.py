from fastapi.testclient import TestClient

from asr_pro.api.main import app

client = TestClient(app)


def get_token(client):
    response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    return response.json()["access_token"]


def test_alerts_routes(client):
    admin_token = get_token(client)
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Get active alerts
    response = client.get("/api/v1/alerts", headers=headers)
    assert response.status_code == 200

    # Get alert rules
    response = client.get("/api/v1/alerts/rules", headers=headers)
    assert response.status_code == 200

    # Create rule
    payload = {
        "name": "Test Rule",
        "target_type": "keyword",
        "target_id": "some_id",
        "condition": {"metric": "hit_count", "operator": "gte", "threshold": 5, "min_count": 5},
        "channels": ["in_app"],
        "cooldown_minutes": 60,
        "is_active": True,
    }
    response = client.post("/api/v1/alerts/rules", json=payload, headers=headers)
    assert response.status_code == 201
    rule_id = response.json().get("id")

    if rule_id:
        # Delete rule
        del_resp = client.delete(f"/api/v1/alerts/rules/{rule_id}", headers=headers)
        assert del_resp.status_code in [200, 204]


def test_keywords_routes(client):
    admin_token = get_token(client)
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Get rules
    response = client.get("/api/v1/keyword-rules", headers=headers)
    assert response.status_code == 200

    # Create rule
    payload = {
        "category": "comp_test",
        "name": "Test Keyword Rule",
        "keywords": ["bad", "terrible"],
        "speaker": "customer",
        "weight": 2.0,
    }
    response = client.post("/api/v1/keyword-rules", json=payload, headers=headers)
    assert response.status_code == 201
    rule_id = response.json().get("id")

    if rule_id:
        # Delete rule
        del_resp = client.delete(f"/api/v1/keyword-rules/{rule_id}", headers=headers)
        assert del_resp.status_code in [200, 204]
