from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import time

from scoring import score_one, score_batch, load_artifact

app = FastAPI(title="Fraud Risk Scoring API")

# Load artifact once at startup
model, block_threshold, review_threshold, feature_names = load_artifact()


# --------------------------
# Request Schemas
# --------------------------
class Transaction(BaseModel):
    data: Dict[str, float]


class BatchTransaction(BaseModel):
    transactions: List[Dict[str, float]]


# --------------------------
# Core Routes
# --------------------------
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Fraud Risk Scoring API is running",
        "block_threshold": block_threshold,
        "review_threshold": review_threshold,
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.get("/ready")
def ready():
    try:
        _, _, _, loaded_feature_names = load_artifact()
        if not loaded_feature_names:
            raise ValueError("Feature names not loaded")

        return {
            "status": "ready",
            "model_loaded": True,
            "feature_count": len(loaded_feature_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {str(e)}")


@app.get("/features")
def features():
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names)
    }


# --------------------------
# Scoring Routes
# --------------------------
@app.post("/score")
def score_transaction(txn: Transaction):
    try:
        if len(txn.data) != len(feature_names):
            raise HTTPException(status_code=400, detail="Invalid feature size")

        result = score_one(txn.data)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch")
def score_transactions(batch: BatchTransaction):
    try:
        for txn in batch.transactions:
            if len(txn) != len(feature_names):
                raise HTTPException(status_code=400, detail="Invalid feature size in batch")

        results = score_batch(batch.transactions)
        return {
            "results": results,
            "count": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))