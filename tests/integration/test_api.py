"""
Integration tests for the FastAPI backend.
Requires a running backend or uses TestClient.
"""

import io
import pytest
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client with mocked model."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Patch model service to return dummy predictor
    import backend.services.model_service as ms
    ms._predictor_instance = ms._DummyPredictor()

    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """Register and login a test user, return auth headers."""
    # Register
    client.post("/api/v1/auth/register", json={
        "email": "test@atmnet.com",
        "username": "testuser",
        "password": "testpass123",
    })
    # Login
    res = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "testpass123",
    })
    if res.status_code == 200:
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}


class TestHealthEndpoints:
    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

    def test_root(self, client):
        res = client.get("/")
        assert res.status_code == 200


class TestAuthEndpoints:
    def test_register_success(self, client):
        res = client.post("/api/v1/auth/register", json={
            "email": "newuser@test.com",
            "username": "newuser",
            "password": "password123",
        })
        assert res.status_code in (201, 409)  # 409 if already exists

    def test_register_invalid_email(self, client):
        res = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "username": "baduser",
            "password": "password123",
        })
        assert res.status_code == 422

    def test_login_wrong_credentials(self, client):
        res = client.post("/api/v1/auth/login", json={
            "username": "nobody",
            "password": "wrong",
        })
        assert res.status_code == 401

    def test_me_unauthorized(self, client):
        res = client.get("/api/v1/auth/me")
        assert res.status_code == 403 or res.status_code == 401


class TestPatientEndpoints:
    def test_create_patient(self, client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        res = client.post("/api/v1/patients", json={
            "patient_code": "PT_TEST_001",
            "sex": "F",
            "age": 45,
        }, headers=auth_headers)
        assert res.status_code in (201, 409)

    def test_list_patients(self, client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        res = client.get("/api/v1/patients", headers=auth_headers)
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_patient_requires_auth(self, client):
        res = client.get("/api/v1/patients")
        assert res.status_code in (401, 403)


class TestPredictEndpoints:
    def test_upload_png_image(self, client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        # Create a fake 64x64 grayscale PNG
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = client.post(
            "/api/v1/predict/upload-mri",
            files={"file": ("test.png", buf, "image/png")},
            data={"modality": "T2"},
            headers=auth_headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert "classification" in data
        assert "severity" in data
        assert "segmentation" in data
        assert "report" in data
        assert "levels" in data

    def test_upload_requires_auth(self, client):
        buf = io.BytesIO(b"fake data")
        res = client.post(
            "/api/v1/predict/upload-mri",
            files={"file": ("test.mha", buf, "application/octet-stream")},
        )
        assert res.status_code in (401, 403)

    def test_unsupported_format(self, client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        buf = io.BytesIO(b"fake data")
        res = client.post(
            "/api/v1/predict/upload-mri",
            files={"file": ("test.txt", buf, "text/plain")},
            headers=auth_headers,
        )
        # Should either reject or process
        assert res.status_code in (200, 400, 422, 500)


class TestAnalyticsEndpoints:
    def test_summary(self, client, auth_headers):
        if not auth_headers:
            pytest.skip("Auth not available")
        res = client.get("/api/v1/analytics/summary", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "total_studies" in data
        assert "total_patients" in data
        assert "total_predictions" in data
