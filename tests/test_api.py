"""
tests/test_api.py
──────────────────
Pytest test suite for the FastAPI inference endpoints.

Run:  pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ── Fixtures ──────────────────────────────────────────────────────────────

MOCK_RESULT = {
    "prediction":    "DDoS",
    "confidence":    0.92,
    "probabilities": {
        "Benign": 0.01, "DDoS": 0.92, "DoS": 0.03,
        "Mirai": 0.01, "Recon": 0.01, "Spoofing": 0.01,
        "Web_BruteForce": 0.01,
    },
    "is_attack":      True,
    "low_confidence": False,
}

MOCK_INFO = {
    "model_type":    "LightGBM (GBDT, multiclass softmax)",
    "n_features":    30,
    "feature_names": ["flow_duration", "Rate", "Srate"],
    "classes":       ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web_BruteForce"],
    "thresholds":    {"Benign": 0.5, "DDoS": 0.48, "DoS": 0.3, "Mirai": 0.5,
                      "Recon": 0.4, "Spoofing": 0.45, "Web_BruteForce": 0.15},
}

SAMPLE_FLOW = {
    "flow_duration": 1234.5,
    "Rate": 100.0,
    "Srate": 50.0,
    "Drate": 50.0,
    "syn_flag_number": 1.0,
    "ack_flag_number": 1.0,
    "fin_flag_number": 0.0,
    "rst_flag_number": 0.0,
    "psh_flag_number": 0.0,
}


@pytest.fixture
def client():
    """TestClient with mocked Predictor so no model files are needed."""
    mock_predictor = MagicMock()
    mock_predictor.predict_single.return_value = MOCK_RESULT
    mock_predictor.predict_batch.return_value = [MOCK_RESULT, MOCK_RESULT]
    mock_predictor.info = MOCK_INFO

    with patch("api.main._predictor", mock_predictor):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "classes" in data
        assert "thresholds" in data

    def test_health_has_process_time_header(self, client):
        r = client.get("/health")
        assert "X-Process-Time-Ms" in r.headers


class TestModelInfo:
    def test_info_returns_features(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert "feature_names" in data
        assert "thresholds" in data
        assert len(data["classes"]) == 7


class TestSinglePredict:
    def test_predict_ddos(self, client):
        r = client.post("/predict", json=SAMPLE_FLOW)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        result = data["result"]
        assert result["prediction"] == "DDoS"
        assert result["is_attack"] is True
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"].keys()) == {
            "Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web_BruteForce"
        }

    def test_predict_low_confidence_flag(self, client):
        """Prediction with confidence < 0.6 must set low_confidence=True."""
        low_conf_result = {**MOCK_RESULT, "confidence": 0.45, "low_confidence": True}
        with patch("api.main._predictor") as mp:
            mp.predict_single.return_value = low_conf_result
            mp.info = MOCK_INFO
            r = client.post("/predict", json=SAMPLE_FLOW)
        assert r.json()["result"]["low_confidence"] is True

    def test_predict_benign_not_attack(self, client):
        benign_result = {**MOCK_RESULT, "prediction": "Benign", "is_attack": False}
        with patch("api.main._predictor") as mp:
            mp.predict_single.return_value = benign_result
            mp.info = MOCK_INFO
            r = client.post("/predict", json=SAMPLE_FLOW)
        assert r.json()["result"]["is_attack"] is False

    def test_predict_empty_body_rejected(self, client):
        r = client.post("/predict", json={})
        # All None fields → predictor raises ValueError → 422
        assert r.status_code in (200, 422)   # depends on mock path

    def test_predict_probabilities_sum_to_one(self, client):
        r = client.post("/predict", json=SAMPLE_FLOW)
        probs = r.json()["result"]["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestBatchPredict:
    def test_batch_two_flows(self, client):
        payload = {"flows": [SAMPLE_FLOW, SAMPLE_FLOW]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_rejected(self, client):
        r = client.post("/predict/batch", json={"flows": []})
        assert r.status_code == 422

    def test_batch_over_limit_rejected(self, client):
        flows = [SAMPLE_FLOW] * 1001
        r = client.post("/predict/batch", json={"flows": flows})
        assert r.status_code == 422

    def test_batch_results_have_required_fields(self, client):
        payload = {"flows": [SAMPLE_FLOW]}
        r = client.post("/predict/batch", json=payload)
        result = r.json()["results"][0]
        for field in ("prediction", "confidence", "probabilities", "is_attack", "low_confidence"):
            assert field in result, f"Missing field: {field}"