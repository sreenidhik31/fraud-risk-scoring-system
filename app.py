import streamlit as st
import pandas as pd
from scoring import score_one, score_batch, load_artifact

st.set_page_config(page_title="Fraud Detection Risk Scoring System", layout="wide")

# -----------------------------
# Load model metadata
# -----------------------------
_, block_threshold, review_threshold, feature_names = load_artifact()

# -----------------------------
# Session state init
# -----------------------------
if "input_data" not in st.session_state:
    st.session_state.input_data = {feature: 0.0 for feature in feature_names}

for feature in feature_names:
    widget_key = f"input_{feature}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = float(st.session_state.input_data.get(feature, 0.0))

# -----------------------------
# Helper functions
# -----------------------------
def fill_example_data():
    example = {feature: 0.0 for feature in feature_names}
    if "Time" in example:
        example["Time"] = 10000.0
    if "Amount" in example:
        example["Amount"] = 150.0

    st.session_state.input_data = example
    for feature, value in example.items():
        st.session_state[f"input_{feature}"] = float(value)

def reset_inputs():
    zeros = {feature: 0.0 for feature in feature_names}
    st.session_state.input_data = zeros
    for feature, value in zeros.items():
        st.session_state[f"input_{feature}"] = float(value)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("System Summary")

st.sidebar.subheader("Decision Policy")
st.sidebar.write(f"**BLOCK** ≥ {block_threshold:.4f}")
st.sidebar.write(f"**REVIEW** ≥ {review_threshold:.4f}")
st.sidebar.write(f"**ALLOW** < {review_threshold:.4f}")

st.sidebar.subheader("Business Assumptions")
st.sidebar.write("Fraud loss = 200")
st.sidebar.write("Block cost = 5")
st.sidebar.write("Review cost = 2")
st.sidebar.write("Optimized 3-tier cost = 2996")

st.sidebar.subheader("Model Info")
st.sidebar.write(f"Features used: {len(feature_names)}")
st.sidebar.write("Model: Balanced Logistic Regression")

# -----------------------------
# Header
# -----------------------------
st.title("💳 Fraud Detection Risk Scoring System")
st.markdown(
    "A cost-optimized fraud scoring app with **ALLOW / REVIEW / BLOCK** decision tiers."
)

m1, m2, m3 = st.columns(3)
m1.metric("Block Threshold", f"{block_threshold:.4f}")
m2.metric("Review Threshold", f"{review_threshold:.4f}")
m3.metric("Optimized Cost", "2996")

st.divider()

# -----------------------------
# Single Transaction Scoring
# -----------------------------
st.header("Single Transaction Scoring")

c1, c2 = st.columns([3, 1])

with c2:
    st.write("")
    st.write("")
    if st.button("Fill Example Data", use_container_width=True):
        fill_example_data()
        st.rerun()

    if st.button("Reset Inputs", use_container_width=True):
        reset_inputs()
        st.rerun()

with c1:
    st.markdown("Enter transaction feature values below.")

left_col, right_col = st.columns(2)
half = len(feature_names) // 2

input_data = {}

for i, feature in enumerate(feature_names):
    target_col = left_col if i < half else right_col
    with target_col:
        input_data[feature] = st.number_input(
            feature,
            format="%.6f",
            key=f"input_{feature}"
        )

st.session_state.input_data = input_data

if st.button("Predict Single Transaction", type="primary"):
    try:
        result = score_one(input_data)

        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)
        prob = result["fraud_probability"]
        decision = result["decision"]

        r1.metric("Fraud Probability", f"{prob:.4f}")
        r2.metric("Decision", decision)

        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Risk score: {prob:.2%}")

        if decision == "BLOCK":
            st.error("🚫 HIGH RISK — BLOCK TRANSACTION")
            st.caption("High confidence fraud based on model probability exceeding block threshold.")
        elif decision == "REVIEW":
            st.warning("⚠️ MEDIUM RISK — SEND TO REVIEW")
            st.caption("Moderate risk — requires manual verification.")
        else:
            st.success("✅ LOW RISK — ALLOW TRANSACTION")
            st.caption("Low predicted fraud probability — safe to allow.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# -----------------------------
# Batch Transaction Scoring
# -----------------------------
st.header("Batch Transaction Scoring")
st.markdown("Upload a CSV file containing the required feature columns for batch fraud scoring.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        required_cols = set(feature_names)
        uploaded_cols = set(batch_df.columns)

        if not required_cols.issubset(uploaded_cols):
            missing_cols = sorted(list(required_cols - uploaded_cols))
            st.error(f"CSV is missing required columns: {missing_cols}")
        else:
            if st.button("Run Batch Scoring", type="primary"):
                results = score_batch(batch_df.to_dict(orient="records"))

                results_df = batch_df.copy()
                results_df["fraud_probability"] = [r["fraud_probability"] for r in results]
                results_df["decision"] = [r["decision"] for r in results]

                st.subheader("Batch Scoring Results")
                st.dataframe(results_df)

                decision_counts = results_df["decision"].value_counts().to_dict()
                s1, s2, s3 = st.columns(3)
                s1.metric("ALLOW", decision_counts.get("ALLOW", 0))
                s2.metric("REVIEW", decision_counts.get("REVIEW", 0))
                s3.metric("BLOCK", decision_counts.get("BLOCK", 0))

                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="fraud_scoring_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")