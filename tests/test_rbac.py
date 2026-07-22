from asr_pro.db.models import Conversation, User, new_uuid


def _login(client, username: str, password: str = "password123") -> str:
    resp = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_conversation(db_session, agent_id, transcript="merhaba dunya") -> str:
    conv = Conversation(
        id=new_uuid(),
        agent_id=agent_id,
        sector="omni",
        duration_sec=10.0,
        full_transcript=transcript,
        asr_confidence=0.9,
        quality_gate_passed=True,
    )
    db_session.add(conv)
    db_session.commit()
    return conv.id


def test_agent_sees_only_own_conversations(client, db_session):
    own_id = _make_conversation(db_session, agent_id="agent")
    other_id = _make_conversation(db_session, agent_id="someone_else")

    token = _login(client, "agent")
    resp = client.get("/api/v1/conversations", headers=_auth(token))
    assert resp.status_code == 200
    ids = {c["id"] for c in resp.json()}
    assert own_id in ids
    assert other_id not in ids

    assert client.get(f"/api/v1/conversations/{own_id}", headers=_auth(token)).status_code == 200
    assert client.get(f"/api/v1/conversations/{other_id}", headers=_auth(token)).status_code == 404


def test_team_lead_sees_team_conversations(client, db_session):
    # "agent" and "team_lead" are seeded onto the same team ("team_alpha").
    teammate_conv = _make_conversation(db_session, agent_id="agent")
    outsider_conv = _make_conversation(db_session, agent_id="qa")  # qa has no team

    token = _login(client, "team_lead")
    resp = client.get("/api/v1/conversations", headers=_auth(token))
    ids = {c["id"] for c in resp.json()}
    assert teammate_conv in ids
    assert outsider_conv not in ids


def test_qa_and_auditor_have_full_visibility(client, db_session):
    conv_id = _make_conversation(db_session, agent_id="someone_else_entirely")
    for role_username in ("qa", "auditor"):
        token = _login(client, role_username)
        resp = client.get("/api/v1/conversations", headers=_auth(token))
        ids = {c["id"] for c in resp.json()}
        assert conv_id in ids
        assert (
            client.get(f"/api/v1/conversations/{conv_id}", headers=_auth(token)).status_code == 200
        )


def test_delete_conversation_is_admin_only(client, db_session):
    for role_username in ("agent", "team_lead", "qa", "auditor"):
        conv_id = _make_conversation(db_session, agent_id=role_username)
        token = _login(client, role_username)
        resp = client.delete(f"/api/v1/conversations/{conv_id}", headers=_auth(token))
        assert resp.status_code == 403

    admin_conv_id = _make_conversation(db_session, agent_id="agent")
    admin_token = _login(client, "admin")
    resp = client.delete(f"/api/v1/conversations/{admin_conv_id}", headers=_auth(admin_token))
    assert resp.status_code == 200


def test_deactivated_user_rejected_immediately_even_with_valid_token(client, db_session):
    token = _login(client, "agent")
    assert client.get("/api/v1/auth/me", headers=_auth(token)).status_code == 200

    db_user = db_session.query(User).filter(User.username == "agent").first()
    db_user.is_active = False
    db_session.commit()

    # Same still-unexpired token must now be rejected — live DB check, not stale JWT claims.
    resp = client.get("/api/v1/auth/me", headers=_auth(token))
    assert resp.status_code == 401

    db_user.is_active = True
    db_session.commit()


def test_audit_log_endpoint_role_gating(client):
    for role_username in ("agent", "team_lead", "qa"):
        token = _login(client, role_username)
        resp = client.get("/api/v1/audit-logs", headers=_auth(token))
        assert resp.status_code == 403

    for role_username in ("admin", "auditor"):
        token = _login(client, role_username)
        resp = client.get("/api/v1/audit-logs", headers=_auth(token))
        assert resp.status_code == 200


def test_conversation_view_and_export_create_audit_entries(client, db_session):
    conv_id = _make_conversation(db_session, agent_id="agent")
    token = _login(client, "agent")

    assert client.get(f"/api/v1/conversations/{conv_id}", headers=_auth(token)).status_code == 200
    assert (
        client.get(f"/api/v1/conversations/{conv_id}/export", headers=_auth(token)).status_code
        == 200
    )

    admin_token = _login(client, "admin")
    resp = client.get("/api/v1/audit-logs?username=agent&action=VIEW", headers=_auth(admin_token))
    assert resp.status_code == 200
    entries = resp.json()
    assert any(conv_id in (e["target_resource"] or "") for e in entries)

    resp = client.get("/api/v1/audit-logs?username=agent&action=EXPORT", headers=_auth(admin_token))
    entries = resp.json()
    assert any(conv_id in (e["target_resource"] or "") for e in entries)
