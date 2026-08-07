import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
import plotly.express as px

st.set_page_config(page_title="Fraud Detection System", layout="wide")


@st.cache_resource
def load_models():
    iso_forest = joblib.load("model_isolation_forest.pkl")
    log_reg = joblib.load("model_logistic_regression.pkl")
    rf_model = joblib.load("model_random_forest_smote.pkl")
    scaler = joblib.load("scaler.pkl")
    return iso_forest, log_reg, rf_model, scaler


@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")


iso_forest, log_reg, rf_model, scaler = load_models()
df = load_data()

st.title("🔍 Real-Time Fraud Detection Simulation")
st.markdown(
    "Simulates transactions streaming in live, flagged by a trained fraud detection model."
)

st.sidebar.header("Simulation Controls")

model_choice = st.sidebar.selectbox(
    "Choose Detection Model",
    ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"],
)

n_transactions = st.sidebar.slider("Number of transactions to simulate", 10, 200, 50)
speed = st.sidebar.slider("Simulation speed (seconds between transactions)", 0.05, 1.0, 0.2)
fraud_boost = st.sidebar.checkbox("Boost fraud likelihood in sample (for demo purposes)", value=True)
start_button = st.sidebar.button("▶ Start Simulation")


def prepare_sample(df, n, boost_fraud):
    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    if boost_fraud:
        n_fraud = max(3, int(n * 0.15))
        sample_fraud = fraud.sample(n=min(n_fraud, len(fraud)), random_state=None)
        sample_normal = normal.sample(n=n - len(sample_fraud), random_state=None)
        sample = pd.concat([sample_fraud, sample_normal])
    else:
        sample = df.sample(n=n, random_state=None)

    return sample.sample(frac=1).reset_index(drop=True)


def predict_transaction(row, model_choice, scaler, iso_forest, log_reg, rf_model):
    features = row.drop(labels=["Class"]).copy()
    features["Amount_scaled"] = scaler.transform([[row["Amount"]]])[0][0]
    features["Time_scaled"] = scaler.transform([[row["Time"]]])[0][0]
    features = features.drop(labels=["Amount", "Time"])
    X = features.values.reshape(1, -1)

    if model_choice == "Isolation Forest":
        pred = iso_forest.predict(X)[0]
        is_fraud = pred == -1
        confidence = None
    elif model_choice == "Logistic Regression":
        proba = log_reg.predict_proba(X)[0][1]
        is_fraud = proba > 0.5
        confidence = proba
    else:
        proba = rf_model.predict_proba(X)[0][1]
        is_fraud = proba > 0.5
        confidence = proba

    return is_fraud, confidence


if start_button:
    sample = prepare_sample(df, n_transactions, fraud_boost)

    col1, col2, col3 = st.columns(3)
    total_placeholder = col1.empty()
    flagged_placeholder = col2.empty()
    accuracy_placeholder = col3.empty()

    log_placeholder = st.empty()
    alert_placeholder = st.empty()

    total_seen = 0
    total_flagged = 0
    correct = 0
    log_rows = []

    for _, row in sample.iterrows():
        is_fraud_pred, confidence = predict_transaction(
            row, model_choice, scaler, iso_forest, log_reg, rf_model
        )
        actual_fraud = row["Class"] == 1

        total_seen += 1
        if is_fraud_pred:
            total_flagged += 1
        if is_fraud_pred == actual_fraud:
            correct += 1

        log_rows.append({
            "Transaction #": total_seen,
            "Amount": f"₹{row['Amount']:.2f}",
            "Model Flagged Fraud": "🚨 YES" if is_fraud_pred else "✅ No",
            "Confidence": f"{confidence:.2%}" if confidence is not None else "N/A",
            "Actually Fraud": "Yes" if actual_fraud else "No",
        })

        total_placeholder.metric("Transactions Processed", total_seen)
        flagged_placeholder.metric("Flagged as Fraud", total_flagged)
        accuracy_placeholder.metric("Running Accuracy", f"{correct / total_seen:.1%}")

        if is_fraud_pred:
            alert_placeholder.error(
                f"🚨 FRAUD ALERT — Transaction #{total_seen}, Amount ₹{row['Amount']:.2f}"
            )
        else:
            alert_placeholder.empty()

        log_df = pd.DataFrame(log_rows[::-1])
        log_placeholder.dataframe(log_df, use_container_width=True, height=350)

        time.sleep(speed)

    st.success(
        f"Simulation complete — processed {total_seen} transactions, flagged {total_flagged} as fraud."
    )


st.markdown("---")
st.header("📊 Model Performance Overview")

perf_data = {
    "Model": ["Isolation Forest", "Logistic Regression", "Random Forest + SMOTE"],
    "Precision": [0.2899, 0.0670, 0.8519],
    "Recall": [0.2703, 0.8784, 0.7770],
    "F1": [0.2797, 0.1245, 0.8127],
}
perf_df = pd.DataFrame(perf_data)
st.dataframe(perf_df, use_container_width=True)

st.caption(
    "Random Forest + SMOTE gives the best overall balance (F1 = 0.81). "
    "Logistic Regression catches the most fraud (Recall = 0.88) but with far more false alarms "
    "(Precision = 0.07). Isolation Forest is fully unsupervised — no fraud labels used in training — "
    "trading some accuracy for the ability to flag novel, previously unseen fraud patterns."
)