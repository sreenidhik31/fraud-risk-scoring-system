💳 Cost-Optimized Fraud Detection System with Risk-Tier Decision Engine

🚀 Overview

Traditional fraud detection models optimize for accuracy — but real-world systems optimize for cost.

This project implements a production-style fraud decision system that:
	•	predicts fraud probability using machine learning
	•	applies a 3-tier decision policy (ALLOW / REVIEW / BLOCK)
	•	minimizes total financial + operational cost
	•	exposes a FastAPI-based scoring service
	•	supports batch inference, policy simulation, logging, and governance metadata

⸻

🧠 Business Problem

Fraud detection operates under extreme class imbalance (~0.17% fraud rate):
	•	Missing fraud (false negatives) → direct financial loss
	•	Blocking legitimate users (false positives) → customer friction
	•	Manual review → operational cost

Most ML models ignore this.

👉 This system reframes fraud detection as a cost minimization problem, not just classification.

⸻

💡 Solution

A cost-aware fraud decision engine:
	1.	Model predicts fraud probability
	2.	Policy engine applies thresholds:
	•	ALLOW → low risk
	•	REVIEW → uncertain
	•	BLOCK → high risk
	3.	Outputs enriched decision with:
	•	risk tier
	•	reasoning
	•	business impact
	•	governance metadata

⸻

📉 Impact (Simulated)

Strategy	Total Cost
Naive threshold (0.5)	~4800
Cost-optimized thresholds	2996

🔥 Result:
	•	↓ ~37% reduction in total cost
	•	↓ unnecessary manual reviews
	•	↑ better balance between fraud risk & user experience

⸻

📊 Data Context
	•	Dataset: Credit Card Fraud Detection
	•	Fraud rate: ~0.17% (highly imbalanced)
	•	Used as a proxy for real-world financial transaction systems

Why this matters:
	•	Simulates rare event detection
	•	Enables cost-sensitive optimization
	•	Reflects real production challenges

⸻

🏗️ System Architecture

[ Input Transaction ]
        ↓
[ ML Model → Fraud Probability ]
        ↓
[ Decision Engine ]
   - review_threshold
   - block_threshold
        ↓
[ Risk Tier Assignment ]
        ↓
[ Business Output ]
   - decision
   - cost
   - explanation
        ↓
[ FastAPI Service ]
        ↓
[ Logging + Metrics + Governance ]


⸻

⚙️ API Endpoints

🔹 Single Transaction

POST /score

Returns:
	•	fraud probability
	•	decision (ALLOW / REVIEW / BLOCK)
	•	risk tier
	•	confidence band
	•	decision reasoning
	•	business impact

⸻

🔹 Batch Scoring

POST /score-batch
	•	Processes multiple transactions
	•	Returns enriched results for each

⸻

🔹 Policy Simulation

POST /simulate-policy
	•	Test custom thresholds without retraining
	•	Evaluate decision outcomes instantly

⸻

🔹 System Monitoring
	•	/metrics → usage stats
	•	/log-summary → logging configuration
	•	/policy → active decision logic
	•	/model-info → model + governance

⸻

🧾 Example Output

{
  "fraud_probability": 0.1052,
  "decision": "ALLOW",
  "risk_tier": "LOW",
  "confidence_band": "HIGH_CONFIDENCE_LEGIT",
  "reason": "Probability below review threshold",
  "decision_cost": 0
}


⸻

🔍 Key Features
	•	✅ Cost-sensitive decision optimization
	•	✅ 3-tier risk policy (production-style)
	•	✅ Batch + real-time scoring
	•	✅ Policy simulation engine
	•	✅ Structured logging (logs/api.log)
	•	✅ Governance metadata (versioning, ownership)
	•	✅ CI/CD with GitHub Actions

⸻

🧱 Tech Stack
	•	Python
	•	FastAPI
	•	Scikit-learn
	•	Pandas / NumPy
	•	Uvicorn
	•	GitHub Actions (CI)

⸻

🧪 How to Run

pip install -r requirements.txt
uvicorn api:app --reload

Visit:

http://127.0.0.1:8000/docs


⸻

🧠 Key Insight

Fraud detection is not a classification problem —
it is a decision optimization problem under asymmetric cost.

⸻

📌 Future Improvements
	•	Threshold optimization using expected cost curves
	•	Real-time streaming scoring (Kafka / AWS)
	•	SHAP-based explainability
	•	Adaptive thresholds based on transaction patterns

⸻

👤 Author

Sreenidhi k 
M.S. Data Science & AI




“Give resume bullet + elevator pitch” 🎯
