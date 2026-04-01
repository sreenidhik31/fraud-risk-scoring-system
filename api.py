from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from scoring import score_one, score_batch, load_artifact

app = FastAPI(title="Fraud Risk Scoring API")

# Load artifact once at startup
model, block_threshold, review_threshold, feature_names = load_artifact()


class Transaction(BaseModel):
    features: Dict[str, float]


class BatchTransactions(BaseModel):
    transactions: List[Dict[str, float]]


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Fraud Risk Scoring API is running",
        "block_threshold": block_threshold,
        "review_threshold": review_threshold
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/features")
def features():
    return {"feature_names": feature_names}


@app.post("/predict")
def predict(transaction: Transaction):
    result = score_one(transaction.features)
    return result


@app.post("/batch_predict")
def batch_predict(batch: BatchTransactions):
    results = score_batch(batch.transactions)
    return {"results": results}