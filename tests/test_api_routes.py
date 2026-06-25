def test_get_keywords(client):
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "agent", "password": "password123"}
    )
    token = login_response.json()["access_token"]
    response = client.get("/api/v1/keyword-rules", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_create_keyword_rule_unauthorized(client):
    response = client.post("/api/v1/keyword-rules", json={"name": "test", "keywords": ["test"]})
    assert response.status_code == 401


def test_create_keyword_rule_admin(client):
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "admin", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    response = client.post(
        "/api/v1/keyword-rules",
        json={"name": "test rule", "keywords": ["test"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201


def test_create_keyword_rule_agent(client):
    login_response = client.post(
        "/api/v1/auth/login", data={"username": "agent", "password": "password123"}
    )
    token = login_response.json()["access_token"]

    response = client.post(
        "/api/v1/keyword-rules",
        json={"name": "test rule 2", "keywords": ["test"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
