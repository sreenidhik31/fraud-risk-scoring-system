import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")


def load_artifact(path=ARTIFACT_PATH):
    artifact = joblib.load(path)
    model = artifact["model"]
    block_threshold = artifact["block_threshold"]
    review_threshold = artifact["review_threshold"]
    feature_names = artifact["feature_names"]
    return model, block_threshold, review_threshold, feature_names


def score_one(transaction: dict):
    model, block_threshold, review_threshold, feature_names = load_artifact()

    X = pd.DataFrame([transaction])

    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = X[feature_names]

    prob = model.predict_proba(X)[0][1]

    if prob >= block_threshold:
        decision = "BLOCK"
    elif prob >= review_threshold:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    return {
        "fraud_probability": float(prob),
        "decision": decision
    }


def score_batch(transactions: list[dict]):
    model, block_threshold, review_threshold, feature_names = load_artifact()

    X = pd.DataFrame(transactions)

    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = X[feature_names]

    probs = model.predict_proba(X)[:, 1]

    results = []
    for prob in probs:
        if prob >= block_threshold:
            decision = "BLOCK"
        elif prob >= review_threshold:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        results.append({
            "fraud_probability": float(prob),
            "decision": decision
        })

    return results


if __name__ == "__main__":
    model, block_threshold, review_threshold, feature_names = load_artifact()

    dummy = {col: 0.0 for col in feature_names}
    if "Amount" in dummy:
        dummy["Amount"] = 100.0

    print("Single prediction:")
    print(score_one(dummy))

    print("\nBatch prediction:")
    print(score_batch([dummy, dummy]))