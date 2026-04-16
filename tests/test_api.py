import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def valid_transaction():
    return {
        "data": {
            "Time": 0.0, "V1": 0.1, "V2": -0.2, "V3": 0.3,
            "V4": -0.1, "V5": 0.05, "V6": -0.3, "V7": 0.2,
            "V8": -0.1, "V9": 0.4, "V10": -0.2, "V11": 0.1,
            "V12": -0.05, "V13": 0.2, "V14": -0.3, "V15": 0.1,
            "V16": -0.2, "V17": 0.3, "V18": -0.1, "V19": 0.05,
            "V20": 0.01, "V21": -0.02, "V22": 0.03, "V23": -0.04,
            "V24": 0.05, "V25": -0.06, "V26": 0.07, "V27": -0.08,
            "V28": 0.09, "Amount": 100.0
        }
    }


# --- infrastructure ---

def test_health():
    assert client.get("/health").status_code == 200

def test_ready():
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True

def test_features_returns_list():
    res = client.get("/features")
    assert res.status_code == 200
    assert isinstance(res.json()["feature_names"], list)
    assert res.json()["feature_count"] == 30

def test_policy_has_thresholds():
    res = client.get("/policy")
    assert res.status_code == 200
    data = res.json()
    assert "thresholds" in data
    assert data["thresholds"]["review_threshold"] < data["thresholds"]["block_threshold"]


# --- /score happy path ---

def test_score_returns_required_fields():
    res = client.post("/score", json=valid_transaction())
    assert res.status_code == 200
    data = res.json()
    for field in ["fraud_probability", "decision", "risk_tier",
                  "confidence_band", "reason", "decision_cost",
                  "thresholds", "business_impact"]:
        assert field in data, f"missing field: {field}"

def test_score_decision_is_valid():
    res = client.post("/score", json=valid_transaction())
    assert res.json()["decision"] in ("ALLOW", "REVIEW", "BLOCK")

def test_score_probability_in_range():
    res = client.post("/score", json=valid_transaction())
    prob = res.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0

def test_score_risk_tier_is_valid():
    res = client.post("/score", json=valid_transaction())
    assert res.json()["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


# --- /score failure cases ---

def test_score_rejects_empty_data():
    res = client.post("/score", json={"data": {}})
    assert res.status_code == 400

def test_score_rejects_missing_features():
    res = client.post("/score", json={"data": {"V1": 1.0, "Amount": 50.0}})
    assert res.status_code == 400

def test_score_rejects_extra_features():
    txn = valid_transaction()
    txn["data"]["FAKE_FEATURE"] = 99.9
    res = client.post("/score", json=txn)
    assert res.status_code == 400

def test_score_rejects_missing_body():
    res = client.post("/score", json={})
    assert res.status_code == 422


# --- /score-batch ---

def test_batch_correct_count():
    res = client.post("/score-batch", json={
        "transactions": [valid_transaction()["data"]] * 5
    })
    assert res.status_code == 200
    assert res.json()["count"] == 5

def test_batch_all_decisions_valid():
    res = client.post("/score-batch", json={
        "transactions": [valid_transaction()["data"]] * 3
    })
    for result in res.json()["results"]:
        assert result["decision"] in ("ALLOW", "REVIEW", "BLOCK")

def test_batch_rejects_wrong_feature_size():
    res = client.post("/score-batch", json={
        "transactions": [{"V1": 1.0, "Amount": 50.0}]
    })
    assert res.status_code == 400

def test_batch_rejects_empty_list():
    res = client.post("/score-batch", json={"transactions": []})
    # empty batch should either 400 or return count 0 — not 500
    assert res.status_code in (200, 400)
    if res.status_code == 200:
        assert res.json()["count"] == 0


# --- /simulate-policy ---

def test_simulate_policy_allow():
    res = client.post("/simulate-policy", json={"fraud_probability": 0.01})
    assert res.status_code == 200
    assert res.json()["decision"] == "ALLOW"

def test_simulate_policy_block():
    res = client.post("/simulate-policy", json={"fraud_probability": 0.99})
    assert res.status_code == 200
    assert res.json()["decision"] == "BLOCK"

def test_simulate_policy_custom_thresholds():
    res = client.post("/simulate-policy", json={
        "fraud_probability": 0.5,
        "review_threshold": 0.3,
        "block_threshold": 0.7
    })
    assert res.status_code == 200
    assert res.json()["decision"] == "REVIEW"

def test_simulate_policy_rejects_inverted_thresholds():
    res = client.post("/simulate-policy", json={
        "fraud_probability": 0.5,
        "review_threshold": 0.9,
        "block_threshold": 0.3
    })
    assert res.status_code == 400

def test_simulate_policy_rejects_probability_above_1():
    res = client.post("/simulate-policy", json={"fraud_probability": 1.5})
    assert res.status_code == 400

def test_simulate_policy_rejects_negative_probability():
    res = client.post("/simulate-policy", json={"fraud_probability": -0.1})
    assert res.status_code == 400

def test_simulate_policy_boundary_review_threshold():
    # exactly at review threshold should be REVIEW not ALLOW
    res = client.get("/policy")
    review_t = res.json()["thresholds"]["review_threshold"]
    sim = client.post("/simulate-policy", json={"fraud_probability": review_t})
    assert sim.json()["decision"] in ("REVIEW", "BLOCK")

def test_simulate_policy_returns_business_impact():
    res = client.post("/simulate-policy", json={"fraud_probability": 0.95})
    assert "business_impact" in res.json()
    assert "expected_action" in res.json()["business_impact"]


# --- /metrics ---

def test_metrics_increments_after_score():
    before = client.get("/metrics").json()["total_requests"]
    client.post("/score", json=valid_transaction())
    after = client.get("/metrics").json()["total_requests"]
    assert after == before + 1