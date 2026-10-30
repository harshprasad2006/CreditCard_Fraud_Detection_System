import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fraud-alert {
        background-color: #ffebee;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #d32f2f;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Resource Loaders (Cached)
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    iso_forest = joblib.load("model_isolation_forest.pkl") if os.path.exists("model_isolation_forest.pkl") else None
    log_reg = joblib.load("model_logistic_regression.pkl") if os.path.exists("model_logistic_regression.pkl") else None
    rf_model = joblib.load("model_random_forest_smote.pkl") if os.path.exists("model_random_forest_smote.pkl") else None
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return iso_forest, log_reg, rf_model, scaler

@st.cache_data
def load_evaluation_data():
    if os.path.exists("model_results.csv"):
        return pd.read_csv("model_results.csv")
    return None

@st.cache_data
def load_raw_dataset():
    if os.path.exists("creditcard.csv"):
        return pd.read_csv("creditcard.csv")
    return None


iso_forest, log_reg, rf_model, scaler = load_models()
results_df = load_evaluation_data()
raw_df = load_raw_dataset()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("Navigation")
st.sidebar.caption("End-to-End Fraud Detection Pipeline")

page = st.sidebar.radio(
    "Go to Module:",
    [
        "1. 📌 Project Overview",
        "2. 📊 Exploratory Data Analysis (EDA)",
        "3. 🧠 Model Comparison & Evaluation",
        "4. 🔴 Live Fraud Simulation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project Status:**\n"
    f"- Models Loaded: {'✅ Ready' if rf_model else '❌ Missing'}\n"
    f"- Eval Results: {'✅ Ready' if results_df is not None else '❌ Missing'}\n"
    f"- Raw Dataset (`creditcard.csv`): {'✅ Loaded' if raw_df is not None else '⚠️ Not Found (Using Cached Results)'}"
)


# -------------------------------------------------------------
# 1. Project Overview Module
# -------------------------------------------------------------
if page == "1. 📌 Project Overview":
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("#### High-Stakes Classification on Severely Imbalanced Financial Data")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Transactions Analyzed", value="284,807")
    with col2:
        st.metric(label="Total Fraud Cases", value="492 (0.17%)")
    with col3:
        st.metric(label="Class Ratio", value="578 : 1 (Normal vs Fraud)")

    st.markdown("### 🎯 Problem Statement")
    st.info(
        """
        - **Extreme Class Imbalance:** Frauds represent only **0.17%** of transactions.
        - **Accuracy Trap:** A naive model that blindly predicts *"No Fraud"* every time gets **99.83% accuracy**, but catches **0 frauds**!
        - **Objective:** Evaluate models on **Precision, Recall, F1-Score, and ROC-AUC** instead of accuracy, comparing:
          1. **Unsupervised Anomaly Detection:** Isolation Forest (trained on normal patterns).
          2. **Cost-Sensitive Supervised:** Logistic Regression (with balanced class weights).
          3. **Oversampling Supervised:** Random Forest with **SMOTE** (Synthetic Minority Over-sampling Technique).
        """
    )

    st.markdown("### 🏗️ Pipeline Architecture")
    st.code(
        """
Raw Data (creditcard.csv)
       │
       ├──> Data Scaling (StandardScaler on Time & Amount)
       ├──> Stratified Train/Test Split (70% Train, 30% Test)
       │
       ├──> [Model 1] Isolation Forest (Unsupervised Anomaly Detection)
       ├──> [Model 2] Logistic Regression (Cost-sensitive class_weight='balanced')
       └──> [Model 3] SMOTE Oversampling ──> Random Forest Classifier
                     │
                     └──> Real-time Simulation Engine & Alerts
        """,
        language="text"
    )


# -------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA) Module (from 01_explore.py)
# -------------------------------------------------------------
elif page == "2. 📊 Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Exploring transaction distributions, time patterns, and class imbalances.")
    st.markdown("---")

    if raw_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Class Distribution (Log Scale)")
            counts = raw_df["Class"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            counts["Class_Label"] = counts["Class"].map({0: "Normal (0)", 1: "Fraud (1)"})
            fig_bar = px.bar(
                counts, x="Class_Label", y="Count", text="Count",
                color="Class_Label", color_discrete_map={"Normal (0)": "#1f77b4", "Fraud (1)": "#d62728"},
                log_y=True, title="Severe Imbalance (Log Scale)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("2. Transaction Amount vs Fraud")
            fig_box = px.box(
                raw_df[raw_df["Amount"] <= 500],
                x="Class", y="Amount", color="Class",
                color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                title="Amount Comparison (Capped at $500 for visibility)"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("3. Transactions by Hour of Day")
        raw_df_copy = raw_df.copy()
        raw_df_copy["Hour"] = (raw_df_copy["Time"] // 3600) % 24
        fraud_hour = raw_df_copy[raw_df_copy["Class"] == 1].groupby("Hour").size().reset_index(name="Fraud Count")
        normal_hour = raw_df_copy[raw_df_copy["Class"] == 0].groupby("Hour").size().reset_index(name="Normal Count")

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(x=normal_hour["Hour"], y=normal_hour["Normal Count"], name="Normal Transactions", line=dict(color="#1f77b4")))
        fig_hour.add_trace(go.Scatter(x=fraud_hour["Hour"], y=fraud_hour["Fraud Count"], name="Fraud Transactions (Right Axis)", yaxis="y2", line=dict(color="#d62728", width=3)))

        fig_hour.update_layout(
            title="Hourly Volume Distribution (Normal vs Fraud)",
            xaxis_title="Hour of Day (0 - 23)",
            yaxis_title="Normal Transaction Count",
            yaxis2=dict(title="Fraud Count", overlaying="y", side="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    else:
        st.info("💡 Real dataset `creditcard.csv` not detected in root folder. Displaying pre-saved EDA artifacts generated during project training.")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("chart_class_imbalance.png"):
                st.image("chart_class_imbalance.png", caption="Class Imbalance Distribution")
        with c2:
            if os.path.exists("chart_amount_by_class.png"):
                st.image("chart_amount_by_class.png", caption="Transaction Amount Distribution")
            else:
                st.image("chart_confusion_matrices.png", caption="Confusion Matrix Summary")


# -------------------------------------------------------------
# 3. Model Comparison & Evaluation Module (from 02 & 03)
# -------------------------------------------------------------
elif page == "3. 🧠 Model Comparison & Evaluation":
    st.title("🧠 Model Comparison & Evaluation")
    st.markdown("Benchmarking **Isolation Forest**, **Logistic Regression (Weighted)**, and **Random Forest (SMOTE)**.")
    st.markdown("---")

    if results_df is not None:
        y_true = results_df["actual"]
        
        models = {
            "Isolation Forest": (results_df["pred_isoforest"], None),
            "Logistic Regression (Balanced)": (results_df["pred_logreg"], results_df["pred_logreg_proba"]),
            "Random Forest (SMOTE)": (results_df["pred_rf_smote"], results_df["pred_rf_smote_proba"]),
        }

        perf = []
        cms = {}
        for name, (pred, proba) in models.items():
            acc = accuracy_score(y_true, pred)
            prec = precision_score(y_true, pred)
            rec = recall_score(y_true, pred)
            f1 = f1_score(y_true, pred)
            auc = roc_auc_score(y_true, proba) if proba is not None else np.nan
            perf.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "N/A"
            })
            cms[name] = confusion_matrix(y_true, pred)

        perf_table = pd.DataFrame(perf)
        st.subheader("📋 Performance Comparison Table")
        st.dataframe(perf_table, use_container_width=True)

        st.subheader("📊 Metric Comparison")
        fig_metrics = px.bar(
            perf_table, x="Model", y=["Precision", "Recall", "F1-Score"],
            barmode="group", title="Precision vs Recall vs F1-Score across Models"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.subheader("🧩 Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, (name, cm) in enumerate(cms.items()):
            with cm_cols[idx]:
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                    title=f"{name}"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("📈 ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        for name, (pred, proba) in models.items():
            if proba is not None:
                fpr, tpr, _ = roc_curve(y_true, proba)
                auc_val = roc_auc_score(y_true, proba)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC = {auc_val:.3f})", mode='lines'))

        fig_roc.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=800, height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.error("Missing `model_results.csv`. Please make sure the project files are present.")


# -------------------------------------------------------------
# 4. Live Fraud Simulation Module (from 04_dashboard.py)
# -------------------------------------------------------------
elif page == "4. 🔴 Live Fraud Simulation":
    st.title("🔴 Real-Time Fraud Detection Simulation")
    st.markdown("Live transaction monitor simulating real-time payment ingestion and instant model inference.")
    st.markdown("---")

    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Simulation Controls")
        model_choice = st.selectbox(
            "Select Machine Learning Model",
            ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"]
        )
        n_transactions = st.slider("Transaction Count", min_value=10, max_value=150, value=30, step=5)
        speed = st.slider("Simulation Latency (seconds/txn)", min_value=0.05, max_value=0.8, value=0.15)
        fraud_boost = st.checkbox("Boost Fraud Ratio (for live demonstration)", value=True)
        start_btn = st.button("▶ Start Real-Time Simulation", type="primary")

    with col_display:
        st.subheader("Stream Monitor")
        status_box = st.empty()
        m_col1, m_col2, m_col3 = st.columns(3)
        stat_processed = m_col1.empty()
        stat_frauds = m_col2.empty()
        stat_amount = m_col3.empty()
        log_container = st.empty()

    if start_btn:
        if raw_df is not None:
            normal = raw_df[raw_df["Class"] == 0]
            fraud = raw_df[raw_df["Class"] == 1]
            if fraud_boost:
                n_fraud = max(2, int(n_transactions * 0.25))
                n_normal = n_transactions - n_fraud
                sample = pd.concat([fraud.sample(n_fraud), normal.sample(n_normal)]).sample(frac=1).reset_index(drop=True)
            else:
                sample = raw_df.sample(n_transactions).reset_index(drop=True)
        else:
            # Fallback synthetic sample matching 30 feature structure
            np.random.seed(42)
            n_fraud = int(n_transactions * 0.25) if fraud_boost else max(1, int(n_transactions * 0.02))
            n_normal = n_transactions - n_fraud
            synth_normal = np.random.normal(0, 1, (n_normal, 28))
            synth_fraud = np.random.normal(2.5, 1.5, (n_fraud, 28))
            all_feats = np.vstack([synth_normal, synth_fraud])
            all_labels = np.array([0]*n_normal + [1]*n_fraud)
            
            sample = pd.DataFrame(all_feats, columns=[f"V{i}" for i in range(1, 29)])
            sample["Time"] = np.random.randint(0, 172800, n_transactions)
            sample["Amount"] = np.random.uniform(5, 500, n_transactions)
            sample["Class"] = all_labels
            sample = sample.sample(frac=1).reset_index(drop=True)

        selected_model = (
            rf_model if model_choice == "Random Forest (SMOTE)"
            else log_reg if model_choice == "Logistic Regression"
            else iso_forest
        )

        frauds_caught = 0
        total_amount_flagged = 0.0
        stream_logs = []

        status_box.info("🚀 Transaction stream initiated...")

        for i, row in sample.iterrows():
            time.sleep(speed)
            
            # Prepare feature vector (V1-V28, Amount_scaled, Time_scaled)
            features = row[[f"V{k}" for k in range(1, 29)]].to_dict()
            if scaler is not None:
                amt_scaled = (row["Amount"] - scaler.mean_[0]) / scaler.scale_[0] if hasattr(scaler, "mean_") else 0
                time_scaled = row["Time"] / 100000.0
            else:
                amt_scaled = row["Amount"] / 100.0
                time_scaled = 0.0
            
            features["Amount_scaled"] = amt_scaled
            features["Time_scaled"] = time_scaled
            
            X_input = pd.DataFrame([features])
            
            # Prediction
            pred = selected_model.predict(X_input)[0]
            is_fraud = (pred == 1) if model_choice != "Isolation Forest" else (pred == -1)

            if is_fraud:
                frauds_caught += 1
                total_amount_flagged += row["Amount"]
                badge = "🚨 **FRAUD DETECTED**"
                msg_color = "red"
            else:
                badge = "✅ Approved"
                msg_color = "green"

            stream_logs.insert(0, {
                "Txn ID": f"TXN-{1000 + i}",
                "Amount ($)": f"${row['Amount']:.2f}",
                "Status": badge,
                "Actual Label": "Fraud" if row["Class"] == 1 else "Normal"
            })

            # Update Metrics
            stat_processed.metric("Processed", f"{i+1} / {n_transactions}")
            stat_frauds.metric("🚨 Frauds Flagged", f"{frauds_caught}")
            stat_amount.metric("💰 Amount Intercepted", f"${total_amount_flagged:,.2f}")

            log_container.dataframe(pd.DataFrame(stream_logs), use_container_width=True)

        status_box.success("✅ Real-Time Simulation Finished Successfully!")
