from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import logging
import os

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

from scoring import score_one, score_batch, load_artifact

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

# Policy metadata / business assumptions
ESTIMATED_TOTAL_COST = 2996
FALSE_POSITIVE_COST = 10
FALSE_NEGATIVE_COST = 150
REVIEW_COST = 3

# Governance metadata
ARTIFACT_NAME = "fraud-risk-scoring-pipeline"
MODEL_VERSION = "1.0.0"
TRAINING_DATE = "2026-04-12"
FEATURE_SCHEMA_VERSION = "v1"
THRESHOLD_VERSION = "v1"
OWNER = "Poojitha Manchi"
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


class PolicyBatchEvaluationRequest(BaseModel):
    transactions: List[Dict[str, float]]
    review_threshold: Optional[float] = None
    block_threshold: Optional[float] = None

class CostImpactSimulationRequest(BaseModel):
    transactions: List[Dict[str, float]]
    review_threshold: Optional[float] = None
    block_threshold: Optional[float] = None
    baseline_review_threshold: float = 0.5
    baseline_block_threshold: float = 0.9

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


def threshold_info(
    review_t: Optional[float] = None,
    block_t: Optional[float] = None
) -> Dict[str, float]:
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


def get_risk_tier(
    prob: float,
    review_t: Optional[float] = None,
    block_t: Optional[float] = None
) -> str:
    review_val = review_threshold if review_t is None else review_t
    block_val = block_threshold if block_t is None else block_t

    if prob >= block_val:
        return "HIGH"
    elif prob >= review_val:
        return "MEDIUM"
    return "LOW"


def get_confidence_band(prob: float) -> str:
    if prob < 0.3:
        return "HIGH_CONFIDENCE_LEGIT"
    elif prob <= 0.7:
        return "UNCERTAIN"
    return "HIGH_CONFIDENCE_FRAUD"


def decision_reason(
    prob: float,
    decision: str,
    review_t: Optional[float] = None,
    block_t: Optional[float] = None
) -> str:
    review_val = review_threshold if review_t is None else review_t
    block_val = block_threshold if block_t is None else block_t

    if decision == "BLOCK":
        return f"Probability {prob:.4f} is above block threshold {block_val:.4f}"
    elif decision == "REVIEW":
        return (
            f"Probability {prob:.4f} is above review threshold "
            f"{review_val:.4f} but below block threshold {block_val:.4f}"
        )
    return f"Probability {prob:.4f} is below review threshold {review_val:.4f}"


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
        "business_impact": get_business_impact(decision),
        "governance": governance_info()
    }


def get_decision_from_thresholds(prob: float, review_t: float, block_t: float) -> str:
    if prob >= block_t:
        return "BLOCK"
    elif prob >= review_t:
        return "REVIEW"
    return "ALLOW"


def summarize_decisions(results: List[Dict[str, object]]) -> Dict[str, int]:
    summary = {"ALLOW": 0, "REVIEW": 0, "BLOCK": 0}
    for result in results:
        decision = str(result["decision"])
        if decision in summary:
            summary[decision] += 1
    return summary


def estimate_batch_decision_cost(results: List[Dict[str, object]]) -> int:
    total_cost = 0
    for result in results:
        total_cost += get_decision_cost(str(result["decision"]))
    return total_cost


def validate_transaction_schema(txn: Dict[str, float], context: str = "transaction") -> None:
    expected_keys = set(feature_names)
    incoming_keys = set(txn.keys())

    if incoming_keys != expected_keys:
        missing = sorted(list(expected_keys - incoming_keys))
        extra = sorted(list(incoming_keys - expected_keys))
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid feature schema in {context}",
                "missing_features": missing,
                "extra_features": extra
            }
        )
    
def validate_threshold_pair(review_t: float, block_t: float) -> None:
    if not (0 <= review_t <= 1 and 0 <= block_t <= 1):
        raise HTTPException(status_code=400, detail="thresholds must be between 0 and 1")

    if review_t >= block_t:
        raise HTTPException(
            status_code=400,
            detail="review_threshold must be less than block_threshold"
        )


def evaluate_results_with_thresholds(
    model_results: List[Dict[str, object]],
    review_t: float,
    block_t: float
) -> List[Dict[str, object]]:
    evaluated_results = []

    for result in model_results:
        prob = float(result["fraud_probability"])
        simulated_decision = get_decision_from_thresholds(prob, review_t, block_t)

        enriched = {
            "fraud_probability": prob,
            "decision": simulated_decision,
            "risk_tier": get_risk_tier(prob, review_t, block_t),
            "confidence_band": get_confidence_band(prob),
            "reason": decision_reason(prob, simulated_decision, review_t, block_t),
            "decision_cost": get_decision_cost(simulated_decision),
            "thresholds": threshold_info(review_t, block_t),
            "business_impact": get_business_impact(simulated_decision)
        }
        evaluated_results.append(enriched)

    return evaluated_results


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
            "/evaluate-policy-batch",
            "/policy",
            "/metrics"
        ]
    }


# --------------------------
# Policy Simulation Routes
# --------------------------
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

        return {
            "fraud_probability": prob,
            "decision": decision,
            "risk_tier": get_risk_tier(prob, review_t, block_t),
            "confidence_band": get_confidence_band(prob),
            "reason": decision_reason(prob, decision, review_t, block_t),
            "decision_cost": get_decision_cost(decision),
            "thresholds": threshold_info(review_t, block_t),
            "policy_summary": policy_summary(),
            "business_impact": get_business_impact(decision),
            "governance": governance_info()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy simulation error | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-policy-batch")
def evaluate_policy_batch(request: PolicyBatchEvaluationRequest) -> Dict[str, object]:
    try:
        logger.info(
            f"Policy batch evaluation request received | batch_size={len(request.transactions)}"
        )

        if not request.transactions:
            raise HTTPException(status_code=400, detail="Batch cannot be empty")

        review_t = request.review_threshold if request.review_threshold is not None else review_threshold
        block_t = request.block_threshold if request.block_threshold is not None else block_threshold

        if not (0 <= review_t <= 1 and 0 <= block_t <= 1):
            raise HTTPException(status_code=400, detail="thresholds must be between 0 and 1")

        if review_t >= block_t:
            raise HTTPException(
                status_code=400,
                detail="review_threshold must be less than block_threshold"
            )

        for txn in request.transactions:
            validate_transaction_schema(txn, context="policy batch evaluation")

        model_results = score_batch(request.transactions)

        evaluated_results = []
        for result in model_results:
            prob = float(result["fraud_probability"])
            simulated_decision = get_decision_from_thresholds(prob, review_t, block_t)

            enriched = {
                "fraud_probability": prob,
                "decision": simulated_decision,
                "risk_tier": get_risk_tier(prob, review_t, block_t),
                "confidence_band": get_confidence_band(prob),
                "reason": decision_reason(prob, simulated_decision, review_t, block_t),
                "decision_cost": get_decision_cost(simulated_decision),
                "thresholds": threshold_info(review_t, block_t),
                "business_impact": get_business_impact(simulated_decision)
            }
            evaluated_results.append(enriched)

        decision_summary = summarize_decisions(evaluated_results)
        average_probability = round(
            sum(float(r["fraud_probability"]) for r in evaluated_results) / len(evaluated_results),
            6
        )
        total_decision_cost = estimate_batch_decision_cost(evaluated_results)

        logger.info(
            f"Policy batch evaluation completed | batch_size={len(evaluated_results)} "
            f"| allow={decision_summary['ALLOW']} | review={decision_summary['REVIEW']} "
            f"| block={decision_summary['BLOCK']} | estimated_decision_cost={total_decision_cost}"
        )

        return {
            "batch_size": len(evaluated_results),
            "thresholds_used": threshold_info(review_t, block_t),
            "decision_summary": decision_summary,
            "average_fraud_probability": average_probability,
            "estimated_total_decision_cost": total_decision_cost,
            "policy_summary": {
                "objective": "Evaluate batch-level decision mix under active threshold policy"
            },
            "governance": governance_info(),
            "results": evaluated_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy batch evaluation error | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-cost-impact")
def simulate_cost_impact(request: CostImpactSimulationRequest) -> Dict[str, object]:
    try:
        logger.info(
            f"Cost impact simulation request received | batch_size={len(request.transactions)}"
        )

        if not request.transactions:
            raise HTTPException(status_code=400, detail="Batch cannot be empty")

        review_t = request.review_threshold if request.review_threshold is not None else review_threshold
        block_t = request.block_threshold if request.block_threshold is not None else block_threshold

        baseline_review_t = request.baseline_review_threshold
        baseline_block_t = request.baseline_block_threshold

        validate_threshold_pair(review_t, block_t)
        validate_threshold_pair(baseline_review_t, baseline_block_t)

        for txn in request.transactions:
            validate_transaction_schema(txn, context="cost impact simulation")

        model_results = score_batch(request.transactions)

        current_results = evaluate_results_with_thresholds(model_results, review_t, block_t)
        baseline_results = evaluate_results_with_thresholds(model_results, baseline_review_t, baseline_block_t)

        current_summary = summarize_decisions(current_results)
        baseline_summary = summarize_decisions(baseline_results)

        current_cost = estimate_batch_decision_cost(current_results)
        baseline_cost = estimate_batch_decision_cost(baseline_results)

        average_probability = round(
            sum(float(r["fraud_probability"]) for r in current_results) / len(current_results),
            6
        )

        cost_difference = baseline_cost - current_cost

        logger.info(
            f"Cost impact simulation completed | batch_size={len(current_results)} "
            f"| current_cost={current_cost} | baseline_cost={baseline_cost} "
            f"| cost_difference={cost_difference}"
        )

        return {
            "batch_size": len(current_results),
            "average_fraud_probability": average_probability,
            "current_policy": {
                "thresholds": threshold_info(review_t, block_t),
                "decision_summary": current_summary,
                "estimated_total_decision_cost": current_cost
            },
            "baseline_policy": {
                "thresholds": {
                    "review_threshold": baseline_review_t,
                    "block_threshold": baseline_block_t
                },
                "decision_summary": baseline_summary,
                "estimated_total_decision_cost": baseline_cost
            },
            "comparison": {
                "cost_difference_vs_baseline": cost_difference,
                "improvement": (
                    "lower_cost_than_baseline"
                    if current_cost < baseline_cost
                    else "higher_or_equal_cost_than_baseline"
                )
            },
            "policy_summary": {
                "objective": "Compare cost impact of active policy versus baseline policy"
            },
            "governance": governance_info()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost impact simulation error | error={str(e)}")
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

        validate_transaction_schema(txn.data)

        result = score_one(txn.data)
        enriched = enrich_result(result)

        metrics["transactions_scored"] += 1
        update_decision_counts([result])

        logger.info(
            f"Decision={enriched['decision']} | "
            f"Risk={enriched['risk_tier']} | "
            f"Prob={enriched['fraud_probability']:.4f}"
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

        for txn in batch.transactions:
            validate_transaction_schema(txn, context="batch")

        results = score_batch(batch.transactions)
        enriched_results = [enrich_result(r) for r in results]

        metrics["transactions_scored"] += len(results)
        update_decision_counts(results)

        logger.info(
            f"Batch scoring completed | batch_size={len(enriched_results)}"
        )

        return {
            "results": enriched_results,
            "count": len(enriched_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch scoring error | error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))