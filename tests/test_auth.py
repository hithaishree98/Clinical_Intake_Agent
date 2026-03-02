import pytest
import jwt
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_client():
    # Patch graph.build_graph and LLM so the app starts without real connections
    with patch("app.graph.build_graph", return_value=MagicMock()), \
         patch("app.llm.get_gemini", return_value=MagicMock()):
        from app.api import app
        return TestClient(app)


@pytest.fixture
def client(tmp_db):
    return make_client()


def get_token(client, password="test-password"):
    resp = client.post("/clinician/token", data={"password": password})
    return resp


class TestTokenEndpoint:
    def test_valid_password_returns_token(self, client):
        resp = get_token(client)
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_wrong_password_returns_401(self, client):
        resp = get_token(client, password="wrong")
        assert resp.status_code == 401

    def test_token_is_valid_jwt(self, client):
        token = get_token(client).json()["access_token"]
        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["sub"] == "clinician"


class TestProtectedRoutes:
    def test_no_token_returns_401(self, client):
        resp = client.get("/clinician/pending")
        assert resp.status_code == 401

    def test_bad_token_returns_401(self, client):
        resp = client.get(
            "/clinician/pending",
            headers={"Authorization": "Bearer bad-token"}
        )
        assert resp.status_code == 401

    def test_valid_token_grants_access(self, client):
        token = get_token(client).json()["access_token"]
        resp = client.get(
            "/clinician/pending",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200

    def test_expired_token_returns_401(self, client):
        # Encode a token that expired 1 hour ago
        import time
        token = jwt.encode(
            {"sub": "clinician", "exp": time.time() - 3600},
            "test-secret",
            algorithm="HS256",
        )
        resp = client.get(
            "/clinician/pending",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401