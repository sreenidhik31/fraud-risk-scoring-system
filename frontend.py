import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fraud Risk Decision System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Risk Decision System")
st.caption("Interactive interface for fraud scoring, policy simulation, and governance-aware decision review")

tab1, tab2, tab3 = st.tabs(["Single Score", "Policy Simulation", "Policy Info"])

FEATURE_NAMES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

DEFAULT_VALUES = {
    "Time": 0.0,
    "V1": 0.1, "V2": -0.2, "V3": 0.3, "V4": -0.1, "V5": 0.05,
    "V6": -0.3, "V7": 0.2, "V8": -0.1, "V9": 0.4, "V10": -0.2,
    "V11": 0.1, "V12": -0.05, "V13": 0.2, "V14": -0.3, "V15": 0.1,
    "V16": -0.2, "V17": 0.3, "V18": -0.1, "V19": 0.05, "V20": 0.01,
    "V21": -0.02, "V22": 0.03, "V23": -0.04, "V24": 0.05, "V25": -0.06,
    "V26": 0.07, "V27": -0.08, "V28": 0.09, "Amount": 100.0
}


def call_api(method: str, endpoint: str, payload=None):
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json(), None

        try:
            error_json = response.json()
            return None, f"{response.status_code}: {error_json}"
        except Exception:
            return None, f"{response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        return None, str(e)


def show_governance(governance: dict):
    st.markdown("### Governance Metadata")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Artifact", governance.get("artifact_name", "N/A"))
        st.metric("Model Version", governance.get("model_version", "N/A"))
        st.metric("Training Date", governance.get("training_date", "N/A"))

    with col2:
        st.metric("Feature Schema", governance.get("feature_schema_version", "N/A"))
        st.metric("Threshold Version", governance.get("threshold_version", "N/A"))
        st.metric("Stage", governance.get("deployment_stage", "N/A"))

    with col3:
        st.metric("Owner", governance.get("owner", "N/A"))


with tab1:
    st.subheader("Single Transaction Scoring")
    st.write("Submit one transaction to the scoring API and review the operational decision output.")

    with st.expander("Enter transaction features", expanded=True):
        cols = st.columns(3)
        transaction_data = {}

        for idx, feature in enumerate(FEATURE_NAMES):
            with cols[idx % 3]:
                transaction_data[feature] = st.number_input(
                    feature,
                    value=float(DEFAULT_VALUES[feature]),
                    format="%.6f"
                )

    if st.button("Score Transaction", use_container_width=True):
        payload = {"data": transaction_data}
        result, error = call_api("POST", "/score", payload)

        if error:
            st.error(error)
        else:
            st.success("Transaction scored successfully")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Decision", result.get("decision", "N/A"))
            c2.metric("Risk Tier", result.get("risk_tier", "N/A"))
            c3.metric("Fraud Probability", round(result.get("fraud_probability", 0.0), 4))
            c4.metric("Decision Cost", result.get("decision_cost", 0))

            st.markdown("### Decision Explanation")
            st.write(result.get("reason", "N/A"))

            st.markdown("### Business Impact")
            st.json(result.get("business_impact", {}))

            st.markdown("### Full API Response")
            st.json(result)


with tab2:
    st.subheader("Policy Simulation")
    st.write("Simulate policy outcomes under default or custom thresholds without scoring a real transaction.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fraud_probability = st.slider(
            "Fraud Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01
        )

        use_custom = st.checkbox("Use custom thresholds")

        if use_custom:
            review_threshold = st.slider("Review Threshold", 0.0, 1.0, 0.8, 0.01)
            block_threshold = st.slider("Block Threshold", 0.0, 1.0, 0.95, 0.01)
            sim_payload = {
                "fraud_probability": fraud_probability,
                "review_threshold": review_threshold,
                "block_threshold": block_threshold
            }
        else:
            st.info("Using default backend thresholds from the active policy.")
            sim_payload = {
                "fraud_probability": fraud_probability
            }

        if st.button("Run Policy Simulation", use_container_width=True):
            result, error = call_api("POST", "/simulate-policy", sim_payload)

            if error:
                st.error(error)
            else:
                st.success("Policy simulation completed")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Decision", result.get("decision", "N/A"))
                c2.metric("Risk Tier", result.get("risk_tier", "N/A"))
                c3.metric("Confidence Band", result.get("confidence_band", "N/A"))
                c4.metric("Decision Cost", result.get("decision_cost", 0))

                st.markdown("### Decision Explanation")
                st.write(result.get("reason", "N/A"))

                st.markdown("### Business Impact")
                st.json(result.get("business_impact", {}))

                st.markdown("### Full Simulation Response")
                st.json(result)

    with col_right:
        policy_data, policy_error = call_api("GET", "/policy")
        st.markdown("### Active Policy Snapshot")

        if policy_error:
            st.error(policy_error)
        elif policy_data:
            thresholds = policy_data.get("thresholds", {})
            st.metric("Review Threshold", thresholds.get("review_threshold", "N/A"))
            st.metric("Block Threshold", thresholds.get("block_threshold", "N/A"))
            st.metric(
                "Estimated Total Cost",
                policy_data.get("policy_summary", {}).get("estimated_total_cost", "N/A")
            )


with tab3:
    st.subheader("Policy & Governance Info")
    st.write("Review current policy, cost assumptions, model metadata, and governance versioning.")

    policy_data, policy_error = call_api("GET", "/policy")
    model_data, model_error = call_api("GET", "/model-info")

    if st.button("Refresh Policy Info", use_container_width=True):
        policy_data, policy_error = call_api("GET", "/policy")
        model_data, model_error = call_api("GET", "/model-info")

    if policy_error:
        st.error(policy_error)
    elif policy_data:
        st.markdown("### Current Policy")
        st.json(policy_data)

        cost_assumptions = policy_data.get("cost_assumptions", {})
        st.markdown("### Cost Assumptions")
        cost_df = pd.DataFrame([
            {"Metric": "False Positive Cost", "Value": cost_assumptions.get("false_positive_cost", "N/A")},
            {"Metric": "False Negative Cost", "Value": cost_assumptions.get("false_negative_cost", "N/A")},
            {"Metric": "Review Cost", "Value": cost_assumptions.get("review_cost", "N/A")}
        ])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        governance = policy_data.get("governance", {})
        if governance:
            show_governance(governance)

    if model_error:
        st.error(model_error)
    elif model_data:
        st.markdown("### Model Info")
        st.json(model_data)