# 🔍 AI Fraud Detection System (FinTech / Banking)

An end-to-end machine learning project detecting fraudulent credit card transactions in a severely imbalanced dataset (0.17% fraud rate) — comparing unsupervised anomaly detection against supervised, imbalance-aware classification, with a real-time fraud alert simulation dashboard.

---

## 📌 Overview

Fraud detection is a classic high-stakes, highly imbalanced classification problem — fraudulent transactions make up a tiny fraction of all activity, which makes naive accuracy a misleading metric. This project builds and compares three different approaches to catching fraud, evaluates them using metrics that actually matter in this context, and presents results through an interactive dashboard simulating live transaction monitoring.

---

## 🎯 Objectives

- Understand and properly handle severe class imbalance (492 frauds out of 284,807 transactions)
- Build and compare an unsupervised anomaly detection model against supervised, imbalance-aware models
- Evaluate models using precision, recall, F1, and ROC-AUC — not accuracy
- Simulate a real-time fraud monitoring system with live alerts

---

## 🗂️ Dataset

[Credit Card Fraud Detection dataset (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European cardholder transactions over 2 days, with 492 (0.17%) labeled as fraud. Features `V1`–`V28` are PCA-anonymized for privacy; `Time` and `Amount` are the only raw features.

*Note: dataset not included in this repo due to file size — download it directly from the Kaggle link above to run locally.*

---

## 🧠 Models Built

| Model | Type | Approach |
|---|---|---|
| Isolation Forest | Unsupervised | Learns what "normal" looks like, flags anything that deviates — no fraud labels used in training |
| Logistic Regression | Supervised | Class-weighted to penalize missed fraud more heavily |
| Random Forest + SMOTE | Supervised | Trained on a synthetically rebalanced dataset (SMOTE oversampling) |

---

## 📊 Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.29 | 0.27 | 0.28 | — |
| Logistic Regression | 0.07 | 0.88 | 0.12 | 0.968 |
| **Random Forest + SMOTE** | **0.85** | **0.78** | **0.81** | **0.949** |

*(A naive model predicting "no fraud" for every transaction scores 99.83% accuracy while catching 0 of 148 frauds — this is why accuracy is not used as the primary metric here.)*

---

## 💡 Key Insights

- **Random Forest + SMOTE delivered the best overall balance**, catching 115 of 148 test-set frauds (77.7% recall) with only 20 false alarms — the strongest precision-recall trade-off of the three models.
- **Logistic Regression maximized recall (87.8%)** but at a steep cost: 1,810 false alarms to catch 130 real frauds, meaning ~14 legitimate transactions flagged for every actual fraud caught — a real business trade-off between catching more fraud and over-flagging genuine customers.
- **Isolation Forest performed weakest numerically (F1 = 0.28)**, but uniquely, it never saw a single fraud label during training — making it the only model of the three capable of flagging genuinely novel fraud patterns it has never seen before.
- **Recall matters more than precision in fraud detection**: missing real fraud is typically costlier than a false alarm, which is why Logistic Regression's high recall is still a meaningful result despite its low precision.

---

## 🖥️ Dashboard Features

- **Live transaction simulation** — transactions stream in one by one with real-time fraud alerts
- **Model selector** — switch between Isolation Forest, Logistic Regression, and Random Forest live
- **Adjustable simulation speed and volume**
- **Running accuracy and fraud-flag counters**
- **Model performance comparison table**

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | scikit-learn, imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Persistence | joblib |

---

## 📁 Project Structure
fraud-detection-system/
│
├── 01_explore.py # EDA, class imbalance analysis
├── 02_modeling.py # Model training (Isolation Forest, Logistic Regression, RF+SMOTE)
├── 03_evaluation.py # Precision/recall/F1/ROC-AUC evaluation, confusion matrices
├── 04_dashboard.py # Streamlit real-time simulation dashboard
├── model_results.csv # Saved test-set predictions
├── requirements.txt
└── README.md
---

## ▶️ How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt

# Download creditcard.csv from Kaggle and place it in this folder

python 01_explore.py
python 02_modeling.py
python 03_evaluation.py
streamlit run 04_dashboard.py
```

---

## 🚀 Future Improvements

- Deploy with the dataset hosted externally (e.g. cloud storage) for a live public demo
- Add SHAP-based explainability to show *why* a transaction was flagged
- Experiment with XGBoost and threshold tuning to further optimize the precision-recall trade-off
- Build a true streaming pipeline (Kafka-style) instead of a simulated loop

---

## 👩‍💻 Author

**Abhinaya M A**