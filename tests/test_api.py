
import json
import math

from fastapi.testclient import TestClient

from src.api import (
    app,
    explain_prediction,
    feature_columns,
    CLASSIFICATION_THRESHOLD,
)

client = TestClient(app)


def make_valid_payload():
    """Create a valid request structure for API validation tests."""
    return {
        "flow_id": "api-validation-test",
        "features": {
            feature: 0.0 for feature in feature_columns
        },
    }


# --------------------------------------------------
# Health and prediction history tests
# --------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predictions_endpoint_returns_list():
    response = client.get("/predictions?limit=10")

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, dict)
    assert "count" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert data["count"] == len(data["predictions"])


def test_predictions_endpoint_respects_limit():
    response = client.get("/predictions?limit=2")

    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) <= 2


def test_predictions_endpoint_rejects_invalid_limit():
    response = client.get("/predictions?limit=abc")

    assert response.status_code == 422


def test_predictions_endpoint_rejects_limit_above_maximum():
    response = client.get("/predictions?limit=201")

    assert response.status_code == 422


# --------------------------------------------------
# Prediction explanation tests
# --------------------------------------------------

def test_prediction_explanation_contains_top_features():
    values = [0.0] * len(feature_columns)

    explanation = explain_prediction(values, 0.95)

    assert "summary" in explanation
    assert "method" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) == min(
        5, len(feature_columns)
    )


def test_prediction_explanation_identifies_attack_threshold():
    values = [0.0] * len(feature_columns)

    explanation = explain_prediction(values, 0.95)

    assert "attack" in explanation["summary"].lower()
    assert f"{CLASSIFICATION_THRESHOLD:.2f}" in explanation["summary"]


def test_prediction_explanation_identifies_benign_threshold():
    values = [0.0] * len(feature_columns)

    explanation = explain_prediction(values, 0.10)

    assert "benign" in explanation["summary"].lower()
    assert f"{CLASSIFICATION_THRESHOLD:.2f}" in explanation["summary"]


def test_prediction_explanation_discloses_limitation():
    values = [0.0] * len(feature_columns)

    explanation = explain_prediction(values, 0.95)

    assert "not causal" in explanation["method"].lower()


# --------------------------------------------------
# Prediction request validation tests
# --------------------------------------------------

def test_prediction_rejects_missing_feature():
    payload = make_valid_payload()
    payload["features"].pop(feature_columns[0])

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_prediction_rejects_unexpected_feature():
    payload = make_valid_payload()
    payload["features"]["unexpected_feature"] = 1.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]


def test_prediction_rejects_empty_flow_id():
    payload = make_valid_payload()
    payload["flow_id"] = ""

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_non_finite_feature():
    payload = make_valid_payload()
    payload["features"][feature_columns[0]] = math.nan

    # Send raw JSON so NaN reaches the API for validation.
    response = client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
def test_prediction_saves_full_json_explanation(monkeypatch):
    import src.api as api_module

    saved_records = []

    def mock_save_prediction(
        flow_id,
        prediction,
        predicted_class,
        attack_probability,
        risk_level,
        model_name,
        created_at,
        explanation="",
    ):
        saved_records.append({
            "flow_id": flow_id,
            "explanation": explanation,
        })
        return 999

    monkeypatch.setattr(
        api_module,
        "save_prediction",
        mock_save_prediction,
    )

    payload = {
        "flow_id": "json-explanation-test",
        "features": {
            feature: 0.0
            for feature in feature_columns
        },
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert len(saved_records) == 1

    stored_explanation = json.loads(
        saved_records[0]["explanation"]
    )

    assert isinstance(stored_explanation, dict)
    assert "summary" in stored_explanation
    assert "method" in stored_explanation
    assert "top_features" in stored_explanation
    assert saved_records[0]["flow_id"] == "json-explanation-test"

def test_prediction_response_contains_full_explanation(monkeypatch):
    import src.api as api_module

    # Prevent changes to the real database.
    monkeypatch.setattr(
        api_module,
        "save_prediction",
        lambda **kwargs: 999,
    )

    payload = make_valid_payload()
    payload["flow_id"] = "response-explanation-test"

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "explanation" in result
    assert isinstance(result["explanation"], dict)
    assert "summary" in result["explanation"]
    assert "method" in result["explanation"]
    assert "top_features" in result["explanation"]