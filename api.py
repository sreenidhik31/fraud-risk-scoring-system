from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import logging

from scoring import score_one, score_batch, load_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Risk Scoring API")

# Load artifact once at startup
model, block_threshold, review_threshold, feature_names = load_artifact()

# Monitoring / usage metrics
metrics = {
    "total_requests": 0,
    "single_requests": 0,
    "batch_requests": 0,
    "transactions_scored": 0,
    "allow_count": 0,
    "review_count": 0,
    "block_count": 0
}

# Policy metadata from notebook / business assumptions
ESTIMATED_TOTAL_COST = 2996
FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 150
REVIEW_COST = 3


# --------------------------
# Request Schemas
# --------------------------
class Transaction(BaseModel):
    data: Dict[str, float]


class BatchTransaction(BaseModel):
    transactions: List[Dict[str, float]]


class PolicySimulationRequest(BaseModel):
    fraud_probability: float
    review_threshold: Optional[float] = None
    block_threshold: Optional[float] = None


# --------------------------
# Helpers
# --------------------------
def update_decision_counts(results: List[Dict[str, object]]) -> None:
    for result in results:
        decision = result.get("decision")
        if decision == "ALLOW":
            metrics["allow_count"] += 1
        elif decision == "REVIEW":
            metrics["review_count"] += 1
        elif decision == "BLOCK":
            metrics["block_count"] += 1


def threshold_info() -> Dict[str, float]:
    return {
        "review_threshold": review_threshold,
        "block_threshold": block_threshold
    }


def policy_summary() -> Dict[str, object]:
    return {
        "objective": "Minimize total fraud + operational cost",
        "estimated_total_cost": ESTIMATED_TOTAL_COST
    }


def get_risk_tier(prob: float) -> str:
    if prob >= block_threshold:
        return "HIGH"
    elif prob >= review_threshold:
        return "MEDIUM"
    return "LOW"


def get_confidence_band(prob: float) -> str:
    if prob < 0.3:
        return "HIGH_CONFIDENCE_LEGIT"
    elif prob <= 0.7:
        return "UNCERTAIN"
    return "HIGH_CONFIDENCE_FRAUD"


def decision_reason(prob: float, decision: str) -> str:
    if decision == "BLOCK":
        return f"Probability {prob:.4f} is above block threshold {block_threshold:.4f}"
    elif decision == "REVIEW":
        return (
            f"Probability {prob:.4f} is above review threshold "
            f"{review_threshold:.4f} but below block threshold {block_threshold:.4f}"
        )
    return f"Probability {prob:.4f} is below review threshold {review_threshold:.4f}"


def get_decision_cost(decision: str) -> int:
    if decision == "REVIEW":
        return REVIEW_COST
    if decision == "BLOCK":
        return FALSE_POSITIVE_COST
    return 0


def get_business_impact(decision: str) -> Dict[str, str]:
    if decision == "BLOCK":
        return {
            "expected_action": "Block transaction immediately",
            "risk_if_ignored": "Potential fraud loss if fraudulent activity is allowed",
            "cost_note": "False positive cost may be incurred if a legitimate transaction is blocked"
        }
    elif decision == "REVIEW":
        return {
            "expected_action": "Send transaction to manual review queue",
            "risk_if_ignored": "Potential fraud may pass without analyst review",
            "cost_note": "Review cost applied instead of immediate block or unrestricted approval"
        }
    return {
        "expected_action": "Allow transaction",
        "risk_if_ignored": "Low estimated fraud risk under current policy",
        "cost_note": "No immediate intervention cost applied"
    }


def enrich_result(result: Dict[str, object]) -> Dict[str, object]:
    prob = float(result["fraud_probability"])
    decision = str(result["decision"])

    return {
        "fraud_probability": prob,
        "decision": decision,
        "risk_tier": get_risk_tier(prob),
        "confidence_band": get_confidence_band(prob),
        "reason": decision_reason(prob, decision),
        "decision_cost": get_decision_cost(decision),
        "thresholds": threshold_info(),
        "policy_summary": policy_summary(),
        "business_impact": get_business_impact(decision)
    }


def get_decision_from_thresholds(prob: float, review_t: float, block_t: float) -> str:
    if prob >= block_t:
        return "BLOCK"
    elif prob >= review_t:
        return "REVIEW"
    return "ALLOW"


def get_risk_tier_from_thresholds(prob: float, review_t: float, block_t: float) -> str:
    if prob >= block_t:
        return "HIGH"
    elif prob >= review_t:
        return "MEDIUM"
    return "LOW"


def get_confidence_band_from_thresholds(prob: float) -> str:
    if prob < 0.3:
        return "HIGH_CONFIDENCE_LEGIT"
    elif prob <= 0.7:
        return "UNCERTAIN"
    return "HIGH_CONFIDENCE_FRAUD"


def decision_reason_from_thresholds(prob: float, decision: str, review_t: float, block_t: float) -> str:
    if decision == "BLOCK":
        return f"Probability {prob:.4f} is above block threshold {block_t:.4f}"
    elif decision == "REVIEW":
        return (
            f"Probability {prob:.4f} is above review threshold "
            f"{review_t:.4f} but below block threshold {block_t:.4f}"
        )
    return f"Probability {prob:.4f} is below review threshold {review_t:.4f}"


# --------------------------
# Core Routes
# --------------------------
@app.get("/")
def home() -> Dict[str, object]:
    return {
        "status": "ok",
        "message": "Fraud Risk Scoring API is running",
        "block_threshold": block_threshold,
        "review_threshold": review_threshold,
        "version": "1.0"
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.get("/ready")
def ready() -> Dict[str, object]:
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
def features() -> Dict[str, object]:
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names)
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, int]:
    return metrics


@app.get("/model-info")
def model_info() -> Dict[str, str]:
    return {
        "model_type": str(type(model))
    }


@app.get("/policy")
def policy() -> Dict[str, object]:
    return {
        "decision_strategy": "3-tier cost-aware fraud decisioning",
        "thresholds": {
            "review_threshold": review_threshold,
            "block_threshold": block_threshold
        },
        "policy_summary": policy_summary(),
        "cost_assumptions": {
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "review_cost": REVIEW_COST
        },
        "model": {
            "type": str(type(model)),
            "feature_count": len(feature_names)
        }
    }


@app.post("/simulate-policy")
def simulate_policy(request: PolicySimulationRequest) -> Dict[str, object]:
    try:
        prob = float(request.fraud_probability)

        review_t = request.review_threshold if request.review_threshold is not None else review_threshold
        block_t = request.block_threshold if request.block_threshold is not None else block_threshold

        if not (0 <= prob <= 1):
            raise HTTPException(status_code=400, detail="fraud_probability must be between 0 and 1")

        if not (0 <= review_t <= 1 and 0 <= block_t <= 1):
            raise HTTPException(status_code=400, detail="thresholds must be between 0 and 1")

        if review_t >= block_t:
            raise HTTPException(
                status_code=400,
                detail="review_threshold must be less than block_threshold"
            )

        decision = get_decision_from_thresholds(prob, review_t, block_t)
        risk_tier = get_risk_tier_from_thresholds(prob, review_t, block_t)
        confidence_band = get_confidence_band_from_thresholds(prob)
        reason = decision_reason_from_thresholds(prob, decision, review_t, block_t)

        return {
            "fraud_probability": prob,
            "decision": decision,
            "risk_tier": risk_tier,
            "confidence_band": confidence_band,
            "reason": reason,
            "decision_cost": get_decision_cost(decision),
            "thresholds": {
                "review_threshold": review_t,
                "block_threshold": block_t
            },
            "policy_summary": policy_summary(),
            "business_impact": get_business_impact(decision)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# Scoring Routes
# --------------------------
@app.post("/score")
def score_transaction(txn: Transaction) -> Dict[str, object]:
    try:
        logger.info(f"Received transaction: {txn.data}")
        metrics["total_requests"] += 1
        metrics["single_requests"] += 1

        if len(txn.data) != len(feature_names):
            raise HTTPException(status_code=400, detail="Invalid feature size")

        result = score_one(txn.data)
        enriched = enrich_result(result)

        metrics["transactions_scored"] += 1
        update_decision_counts([result])

        logger.info(f"Prediction result: {enriched}")
        return enriched

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch")
def score_transactions(batch: BatchTransaction) -> Dict[str, object]:
    try:
        logger.info(f"Received batch of size: {len(batch.transactions)}")
        metrics["total_requests"] += 1
        metrics["batch_requests"] += 1

        for txn in batch.transactions:
            if len(txn) != len(feature_names):
                raise HTTPException(status_code=400, detail="Invalid feature size in batch")

        results = score_batch(batch.transactions)
        enriched_results = [enrich_result(r) for r in results]

        metrics["transactions_scored"] += len(results)
        update_decision_counts(results)

        logger.info("Batch scoring completed")

        return {
            "results": enriched_results,
            "count": len(enriched_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))