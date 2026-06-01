# 🌱 Green-SLA: Multi-Objective Workload Optimization Engine

## 📊 Live Interactive Dashboard
👉 **[[Click Here to Interact with the Live App]](https://green-sla-mfync3qcngchyenvgmlczu.streamlit.app/)** 

---

## 🔬 Abstract
Data center energy consumption represents a critical sustainability challenge for global cloud infrastructure. This research implements an end-to-end data-driven scheduling engine designed to minimize carbon expenditure without violating industrial Service Level Agreements (SLAs). 

Utilizing classical machine learning models trained on historical electricity transmission logs, we forecast grid load volatility 24 hours in advance. A heuristic optimization framework then evaluates incoming distributed system tasks, shifting execution schedules to peak renewable windows based on priority constraints.

## 🛠️ Core Project Architecture
The project is built symmetrically across two operational phases:
1. **Predictive Analytics Pipeline (`src/train_model.py`):** Implements a `GradientBoostingRegressor` via Scikit-learn, generating hourly grid demand forecasts. It utilizes temporal feature engineering and historical autoregressive lags ($t-24, t-168$) to map seasonal grid dynamics.
2. **Operations Optimization Engine (`src/optimizer.py`):** Runs an evaluation heuristic over workload queues. It maps high-priority tasks to tighter windows and low-priority tasks to more flexible windows to maximize carbon-free energy utilization.

## 💾 Datasets Utilized
To ensure academic and industrial validity, this engine is tested against real production log dimensions:
* **Grid Data:** Hourly PJM Regional Transmission electricity grid generation logs.
* **Cluster Workloads:** Production logs from **Google Cluster Traces**, utilizing `collection_id`, `instance_index`, `average_usage`, and `priority` columns.

## 📈 Empirical Evaluation & Results
The engine was validated by processing randomized production batches of real cluster tasks. The optimization framework yielded the following baseline results:

| Optimization Strategy | Average Workload Delay | SLA Violations | Carbon Saved vs. Immediate |
| :--- | :--- | :--- | :--- |
| Baseline (Immediate Execution) | 0.00 Hours | 0.00% | 0 Units (Reference) |
| **Green-SLA Engine (Ours)** | **4.32 Hours** | **0.00%** | **~240,363.52 Units** |

*Conclusion: The Green-SLA engine successfully achieves a massive reduction in operational carbon footprints while maintaining flawless (100%) compliance with strict workload latency constraints.*
