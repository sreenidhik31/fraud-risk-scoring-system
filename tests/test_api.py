from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def valid_transaction():
    return {
        "data": {
            "Time": 0.0,
            "V1": 0.1,
            "V2": -0.2,
            "V3": 0.3,
            "V4": -0.1,
            "V5": 0.05,
            "V6": -0.3,
            "V7": 0.2,
            "V8": -0.1,
            "V9": 0.4,
            "V10": -0.2,
            "V11": 0.1,
            "V12": -0.05,
            "V13": 0.2,
            "V14": -0.3,
            "V15": 0.1,
            "V16": -0.2,
            "V17": 0.3,
            "V18": -0.1,
            "V19": 0.05,
            "V20": 0.01,
            "V21": -0.02,
            "V22": 0.03,
            "V23": -0.04,
            "V24": 0.05,
            "V25": -0.06,
            "V26": 0.07,
            "V27": -0.08,
            "V28": 0.09,
            "Amount": 100.0
        }
    }


def test_health():
    assert client.get("/health").status_code == 200


def test_ready():
    assert client.get("/ready").status_code == 200


def test_score():
    res = client.post("/score", json=valid_transaction())
    assert res.status_code == 200
    data = res.json()
    assert "decision" in data
    assert "risk_tier" in data


def test_score_batch():
    res = client.post("/score-batch", json={
        "transactions": [
            valid_transaction()["data"],
            valid_transaction()["data"]
        ]
    })
    assert res.status_code == 200
    assert len(res.json()["results"]) == 2


def test_policy():
    res = client.get("/policy")
    assert res.status_code == 200


def test_simulate_policy():
    res = client.post("/simulate-policy", json={
        "fraud_probability": 0.95
    })
    assert res.status_code == 200


def test_invalid_thresholds():
    res = client.post("/simulate-policy", json={
        "fraud_probability": 0.5,
        "review_threshold": 0.99,
        "block_threshold": 0.9
    })
    assert res.status_code == 400