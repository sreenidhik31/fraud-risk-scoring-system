from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import logging
import os

from scoring import score_one, score_batch, load_artifact

# --------------------------
# Logging setup
# --------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Risk Scoring API")

# --------------------------
# Load artifact once at startup
# --------------------------
model, block_threshold, review_threshold, feature_names = load_artifact()

# --------------------------
# Monitoring / usage metrics
# --------------------------
metrics = {
    "total_requests": 0,
    "single_requests": 0,
    "batch_requests": 0,
    "transactions_scored": 0,
    "allow_count": 0,
    "review_count": 0,
    "block_count": 0
}

# --------------------------
# Policy metadata / business assumptions
# --------------------------
ESTIMATED_TOTAL_COST = 2996
FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 150
REVIEW_COST = 3

# --------------------------
# Governance metadata
# --------------------------
ARTIFACT_NAME = "fraud-risk-scoring-pipeline"
MODEL_VERSION = "1.0.0"
TRAINING_DATE = "2026-04-12"
FEATURE_SCHEMA_VERSION = "v1"
THRESHOLD_VERSION = "v1"
OWNER = "Sreenidhi reddy k"
DEPLOYMENT_STAGE = "development"

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
# Middleware
# --------------------------
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)

    logger.info(
        f"Request completed | method={request.method} | path={request.url.path} "
        f"| status_code={response.status_code} | duration_seconds={duration}"
    )
    return response


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


def threshold_info(review_t: Optional[float] = None, block_t: Optional[float] = None) -> Dict[str, float]:
    return {
        "review_threshold": review_threshold if review_t is None else review_t,
        "block_threshold": block_threshold if block_t is None else block_t
    }


def governance_info() -> Dict[str, object]:
    return {
        "artifact_name": ARTIFACT_NAME,
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "owner": OWNER,
        "deployment_stage": DEPLOYMENT_STAGE
    }


def policy_summary() -> Dict[str, object]:
    return {
        "objective": "Minimize total fraud + operational cost",
        "estimated_total_cost": ESTIMATED_TOTAL_COST
    }


def get_risk_tier(prob: float, review_t: float, block_t: float) -> str:
    if prob >= block_t:
        return "HIGH"
    if prob >= review_t:
        return "MEDIUM"
    return "LOW"


def get_confidence_band(prob: float) -> str:
    if prob < 0.3:
        return "HIGH_CONFIDENCE_LEGIT"
    if prob <= 0.7:
        return "UNCERTAIN"
    return "HIGH_CONFIDENCE_FRAUD"


def get_decision_from_thresholds(prob: float, review_t: float, block_t: float) -> str:
    if prob >= block_t:
        return "BLOCK"
    if prob >= review_t:
        return "REVIEW"
    return "ALLOW"


def decision_reason(prob: float, decision: str, review_t: float, block_t: float) -> str:
    if decision == "BLOCK":
        return f"Probability {prob:.4f} is above block threshold {block_t:.4f}"
    if decision == "REVIEW":
        return (
            f"Probability {prob:.4f} is above review threshold "
            f"{review_t:.4f} but below block threshold {block_t:.4f}"
        )
    return f"Probability {prob:.4f} is below review threshold {review_t:.4f}"


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
    if decision == "REVIEW":
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


def enrich_result(
    result: Dict[str, object],
    review_t: Optional[float] = None,
    block_t: Optional[float] = None
) -> Dict[str, object]:
    active_review = review_threshold if review_t is None else review_t
    active_block = block_threshold if block_t is None else block_t

    prob = float(result["fraud_probability"])
    decision = str(result["decision"])
    risk_tier = get_risk_tier(prob, active_review, active_block)

    return {
        "fraud_probability": prob,
        "decision": decision,
        "risk_tier": risk_tier,
        "confidence_band": get_confidence_band(prob),
        "reason": decision_reason(prob, decision, active_review, active_block),
        "decision_cost": get_decision_cost(decision),
        "thresholds": threshold_info(active_review, active_block),
        "policy_summary": policy_summary(),
        "business_impact": get_business_impact(decision),
        "governance": governance_info()
    }


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
        "version": MODEL_VERSION
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
            "feature_count": len(loaded_feature_names),
            "model_version": MODEL_VERSION
        }
    except Exception as e:
        logger.error(f"Ready check failed | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Model not ready: {str(e)}")


@app.get("/features")
def features() -> Dict[str, object]:
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "feature_schema_version": FEATURE_SCHEMA_VERSION
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, int]:
    return metrics


@app.get("/model-info")
def model_info() -> Dict[str, object]:
    return {
        "model_type": str(type(model)),
        "feature_count": len(feature_names),
        "governance": governance_info()
    }


@app.get("/policy")
def policy() -> Dict[str, object]:
    return {
        "decision_strategy": "3-tier cost-aware fraud decisioning",
        "thresholds": threshold_info(),
        "policy_summary": policy_summary(),
        "cost_assumptions": {
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "review_cost": REVIEW_COST
        },
        "model": {
            "type": str(type(model)),
            "feature_count": len(feature_names)
        },
        "governance": governance_info()
    }


@app.get("/log-summary")
def log_summary() -> Dict[str, object]:
    return {
        "logging_enabled": True,
        "log_file": "logs/api.log",
        "tracked_endpoints": [
            "/score",
            "/score-batch",
            "/simulate-policy",
            "/policy",
            "/metrics"
        ]
    }


# --------------------------
# Policy Simulation Route
# --------------------------
@app.post("/simulate-policy")
def simulate_policy(request: PolicySimulationRequest) -> Dict[str, object]:
    try:
        prob = float(request.fraud_probability)
        review_t = request.review_threshold if request.review_threshold is not None else review_threshold
        block_t = request.block_threshold if request.block_threshold is not None else block_threshold

        logger.info(
            f"Policy simulation request received | probability={prob:.4f} "
            f"| review_threshold={review_t:.4f} | block_threshold={block_t:.4f}"
        )

        if not (0 <= prob <= 1):
            raise HTTPException(status_code=400, detail="fraud_probability must be between 0 and 1")

        if not (0 <= review_t <= 1 and 0 <= block_t <= 1):
            raise HTTPException(status_code=400, detail="thresholds must be between 0 and 1")

        if review_t >= block_t:
            raise HTTPException(status_code=400, detail="review_threshold must be less than block_threshold")

        decision = get_decision_from_thresholds(prob, review_t, block_t)
        result = {
            "fraud_probability": prob,
            "decision": decision
        }
        enriched = enrich_result(result, review_t, block_t)

        logger.info(
            f"Policy simulation completed | decision={enriched['decision']} "
            f"| risk_tier={enriched['risk_tier']} | prob={enriched['fraud_probability']:.4f}"
        )

        return enriched

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy simulation error | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# Scoring Routes
# --------------------------
@app.post("/score")
def score_transaction(txn: Transaction) -> Dict[str, object]:
    try:
        logger.info(f"Score request received | features_count={len(txn.data)}")
        metrics["total_requests"] += 1
        metrics["single_requests"] += 1

        if len(txn.data) != len(feature_names):
            raise HTTPException(status_code=400, detail="Invalid feature size")

        result = score_one(txn.data)
        enriched = enrich_result(result)

        metrics["transactions_scored"] += 1
        update_decision_counts([result])

        logger.info(
            f"Single scoring completed | decision={enriched['decision']} | "
            f"risk_tier={enriched['risk_tier']} | prob={enriched['fraud_probability']:.4f}"
        )

        return enriched

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scoring | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch")
def score_transactions(batch: BatchTransaction) -> Dict[str, object]:
    try:
        logger.info(f"Batch scoring request received | batch_size={len(batch.transactions)}")
        metrics["total_requests"] += 1
        metrics["batch_requests"] += 1

        if not batch.transactions:
            raise HTTPException(status_code=400, detail="Batch cannot be empty")

        expected_keys = set(feature_names)

        for txn in batch.transactions:
            incoming_keys = set(txn.keys())
            if incoming_keys != expected_keys:
                missing = sorted(list(expected_keys - incoming_keys))
                extra = sorted(list(incoming_keys - expected_keys))
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Invalid feature schema in batch",
                        "missing_features": missing,
                        "extra_features": extra
                    }
                )

        results = score_batch(batch.transactions)
        enriched_results = [enrich_result(r) for r in results]

        metrics["transactions_scored"] += len(results)
        update_decision_counts(results)

        logger.info(f"Batch scoring completed | batch_size={len(enriched_results)}")

        return {
            "results": enriched_results,
            "count": len(enriched_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch scoring error | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))