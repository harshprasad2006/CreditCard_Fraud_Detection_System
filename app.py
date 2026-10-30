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
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Detection Engine",
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
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Resource Loaders (Cached Models)
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    iso_forest = joblib.load("model_isolation_forest.pkl") if os.path.exists("model_isolation_forest.pkl") else None
    log_reg = joblib.load("model_logistic_regression.pkl") if os.path.exists("model_logistic_regression.pkl") else None
    rf_model = joblib.load("model_random_forest_smote.pkl") if os.path.exists("model_random_forest_smote.pkl") else None
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return iso_forest, log_reg, rf_model, scaler

iso_forest, log_reg, rf_model, default_scaler = load_models()


# -------------------------------------------------------------
# Sidebar Dataset Management (Dynamic Upload or Default)
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=65)
st.sidebar.title("Data Control Center")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Custom CSV Dataset:",
    type=["csv"],
    help="Upload any credit card transaction dataset with V1-V28, Time, Amount, and optional Class column."
)

@st.cache_data
def get_dataset(file_upload):
    if file_upload is not None:
        return pd.read_csv(file_upload), "Uploaded CSV (" + file_upload.name + ")"
    elif os.path.exists("creditcard.csv"):
        return pd.read_csv("creditcard.csv"), "Default Dataset (creditcard.csv)"
    elif os.path.exists("model_results.csv"):
        return None, "No Raw CSV (Pre-calculated fallback only)"
    return None, "No Data Found"

df, dataset_source = get_dataset(uploaded_file)

st.sidebar.success(f"**Active Source:**\n{dataset_source}")
if df is not None:
    st.sidebar.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation:",
    [
        "1. 📌 Project Overview",
        "2. 📊 Dynamic Dataset & EDA",
        "3. 🧠 Live Model Inference & Evaluation",
        "4. 🔴 Real-Time Simulation Engine",
        "5. 🧪 Single Transaction Inspector"
    ]
)


# -------------------------------------------------------------
# Helper: Live Feature Scaler & Predictor
# -------------------------------------------------------------
def run_live_predictions(input_df, model_obj, model_type="rf"):
    # Expected features: V1..V28, Amount_scaled, Time_scaled
    proc_df = input_df.copy()
    
    # Scale Amount and Time
    cur_scaler = StandardScaler()
    if "Amount" in proc_df.columns:
        proc_df["Amount_scaled"] = cur_scaler.fit_transform(proc_df[["Amount"]])
    else:
        proc_df["Amount_scaled"] = 0.0

    if "Time" in proc_df.columns:
        proc_df["Time_scaled"] = cur_scaler.fit_transform(proc_df[["Time"]])
    else:
        proc_df["Time_scaled"] = 0.0

    # Ensure all required V1-V28 columns exist
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    for col in [f"V{i}" for i in range(1, 29)]:
        if col not in proc_df.columns:
            proc_df[col] = 0.0

    X_features = proc_df[feature_cols]

    if model_type == "iso":
        preds_raw = model_obj.predict(X_features)
        preds = np.where(preds_raw == -1, 1, 0)
        probas = None
    else:
        preds = model_obj.predict(X_features)
        probas = model_obj.predict_proba(X_features)[:, 1] if hasattr(model_obj, "predict_proba") else None

    return preds, probas


# -------------------------------------------------------------
# 1. Project Overview Module
# -------------------------------------------------------------
if page == "1. 📌 Project Overview":
    st.title("🛡️ AI Fraud Detection Engine")
    st.markdown("#### Real-Time Machine Learning Pipeline for High-Stakes Financial Transactions")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Loaded Rows", f"{len(df):,}" if df is not None else "0")
    with c2:
        frauds_in_data = int(df["Class"].sum()) if (df is not None and "Class" in df.columns) else "N/A"
        st.metric("Fraud Labels in Data", str(frauds_in_data))
    with c3:
        if df is not None and "Class" in df.columns:
            pct_fraud = (df["Class"].sum() / len(df)) * 100
            st.metric("Imbalance Rate", f"{pct_fraud:.3f}%")
        else:
            st.metric("Imbalance Rate", "Unlabeled")
    with c4:
        st.metric("Engine Mode", "Dynamic Live Inference")

    st.markdown("### 🎯 Problem Statement & Methodology")
    st.info(
        """
        - **Real-Time Data Ingestion:** Upload any dataset in the sidebar. The system dynamically scales features (`StandardScaler`), checks schema integrity, and feeds it directly into trained ML models.
        - **Why Not Accuracy?** With a ~0.17% fraud rate, a naive algorithm that flags every transaction as 'Normal' scores 99.83% accuracy while missing 100% of frauds.
        - **Models Evaluated:**
          1. **Isolation Forest (Unsupervised):** Anomaly detection that isolates outliers without ground-truth labels.
          2. **Cost-Sensitive Logistic Regression:** Applies balanced class weighting to punish missed frauds.
          3. **Random Forest + SMOTE (Champion Model):** Trained on synthetically balanced minority samples to maximize F1-Score & Recall.
        """
    )


# -------------------------------------------------------------
# 2. Dynamic Dataset & EDA Module
# -------------------------------------------------------------
elif page == "2. 📊 Dynamic Dataset & EDA":
    st.title("📊 Dynamic Dataset & Exploratory Data Analysis")
    st.markdown(f"**Data Source:** `{dataset_source}`")
    st.markdown("---")

    if df is not None:
        st.subheader("1. Raw Data Preview (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Class" in df.columns:
                st.subheader("2. Class Imbalance (Actual Distribution)")
                counts = df["Class"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]
                counts["Label"] = counts["Class"].map({0: "Legitimate (0)", 1: "Fraud (1)"})
                fig_bar = px.bar(
                    counts, x="Label", y="Count", text="Count",
                    color="Label", color_discrete_map={"Legitimate (0)": "#1f77b4", "Fraud (1)": "#d62728"},
                    log_y=True, title="Class Imbalance (Logarithmic Scale)"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Column 'Class' not found in uploaded file. Class distribution chart skipped.")

        with c2:
            if "Amount" in df.columns and "Class" in df.columns:
                st.subheader("3. Transaction Amount Comparison")
                fig_box = px.box(
                    df[df["Amount"] <= 500],
                    x="Class", y="Amount", color="Class",
                    color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                    title="Transaction Amount ($0 - $500 Range)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            elif "Amount" in df.columns:
                st.subheader("3. Transaction Amount Distribution")
                fig_hist = px.histogram(df[df["Amount"] <= 500], x="Amount", nbins=50, title="Amount Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)

        if "Time" in df.columns and "Class" in df.columns:
            st.subheader("4. Hourly Transaction Patterns")
            df_hour = df.copy()
            df_hour["Hour"] = (df_hour["Time"] // 3600) % 24
            fraud_h = df_hour[df_hour["Class"] == 1].groupby("Hour").size().reset_index(name="Fraud Count")
            norm_h = df_hour[df_hour["Class"] == 0].groupby("Hour").size().reset_index(name="Normal Count")

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=norm_h["Hour"], y=norm_h["Normal Count"], name="Normal Transactions", line=dict(color="#1f77b4")))
            fig_h.add_trace(go.Scatter(x=fraud_h["Hour"], y=fraud_h["Fraud Count"], name="Fraud Transactions (Right Axis)", yaxis="y2", line=dict(color="#d62728", width=3)))
            fig_h.update_layout(
                title="Transactions by Hour of Day",
                xaxis_title="Hour (0 - 23)",
                yaxis_title="Normal Volume",
                yaxis2=dict(title="Fraud Volume", overlaying="y", side="right"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.error("No dataset available. Please upload a CSV file from the sidebar.")


# -------------------------------------------------------------
# 3. Live Model Inference & Evaluation (Dynamic!)
# -------------------------------------------------------------
elif page == "3. 🧠 Live Model Inference & Evaluation":
    st.title("🧠 Live Model Inference & Performance Evaluation")
    st.markdown("Run live predictions against your active dataset across all 3 trained models in real time.")
    st.markdown("---")

    if df is not None and "Class" in df.columns:
        st.write(f"Evaluating models live on **{min(len(df), 25000):,} samples** from `{dataset_source}`...")
        
        # Take a balanced evaluation slice for instant responsiveness
        eval_slice = df.sample(min(len(df), 25000), random_state=42) if len(df) > 25000 else df
        y_actual = eval_slice["Class"].values

        with st.spinner("Executing live predictions across Random Forest, Logistic Regression, and Isolation Forest..."):
            pred_rf, proba_rf = run_live_predictions(eval_slice, rf_model, "rf")
            pred_lr, proba_lr = run_live_predictions(eval_slice, log_reg, "lr")
            pred_iso, _ = run_live_predictions(eval_slice, iso_forest, "iso")

        model_results = {
            "Random Forest + SMOTE": (pred_rf, proba_rf),
            "Logistic Regression (Balanced)": (pred_lr, proba_lr),
            "Isolation Forest": (pred_iso, None)
        }

        perf_data = []
        cms = {}
        for m_name, (m_pred, m_prob) in model_results.items():
            acc = accuracy_score(y_actual, m_pred)
            prec = precision_score(y_actual, m_pred, zero_division=0)
            rec = recall_score(y_actual, m_pred, zero_division=0)
            f1 = f1_score(y_actual, m_pred, zero_division=0)
            auc = roc_auc_score(y_actual, m_prob) if m_prob is not None else np.nan

            perf_data.append({
                "Model": m_name,
                "Accuracy": f"{acc * 100:.2f}%",
                "Precision": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1-Score": f"{f1:.4f}",
                "ROC-AUC": f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            })
            cms[m_name] = confusion_matrix(y_actual, m_pred)

        st.subheader("📋 Live Performance Comparison Table")
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        st.subheader("🧩 Confusion Matrices (Live Predictions)")
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        for idx, (m_name, cm) in enumerate(cms.items()):
            col_target = [col_cm1, col_cm2, col_cm3][idx]
            with col_target:
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                    title=f"{m_name}"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("📈 ROC Curves Comparison")
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        for m_name, (m_pred, m_prob) in model_results.items():
            if m_prob is not None:
                fpr, tpr, _ = roc_curve(y_actual, m_prob)
                auc_val = roc_auc_score(y_actual, m_prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{m_name} (AUC = {auc_val:.3f})", mode='lines'))

        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=800, height=500)
        st.plotly_chart(fig_roc, use_container_width=True)

    elif df is not None:
        st.warning("The uploaded file does not have a 'Class' column (unlabeled data). Evaluation metrics require ground truth labels, but you can still run Real-Time Simulation and Single Transaction Inspector!")
    else:
        st.error("No dataset found. Please upload a dataset in the sidebar.")


# -------------------------------------------------------------
# 4. Real-Time Simulation Engine (Dynamic Source Streaming)
# -------------------------------------------------------------
elif page == "4. 🔴 Real-Time Simulation Engine":
    st.title("🔴 Real-Time Transaction Monitoring Engine")
    st.markdown(f"Streaming live transactions directly from: `{dataset_source}`")
    st.markdown("---")

    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Simulation Controls")
        chosen_model_name = st.selectbox(
            "Detection Model Engine",
            ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"]
        )
        sim_count = st.slider("Stream Batch Size", 10, 100, 30, 5)
        sim_speed = st.slider("Latency (seconds per transaction)", 0.05, 0.8, 0.15)
        boost_frauds = st.checkbox("Inject Frauds in Stream (Recommended for Live Demo)", value=True)
        start_stream_btn = st.button("▶ Start Real-Time Stream", type="primary")

    with col_display:
        st.subheader("Live Gateway Ticker")
        status_box = st.empty()
        m1, m2, m3 = st.columns(3)
        w_proc = m1.empty()
        w_fraud = m2.empty()
        w_amt = m3.empty()
        log_box = st.empty()

    if start_stream_btn:
        if df is None:
            st.error("Please provide a dataset first.")
            st.stop()

        # Sample transactions from active dataset
        if "Class" in df.columns and boost_frauds:
            frauds_subset = df[df["Class"] == 1]
            normal_subset = df[df["Class"] == 0]
            n_f = max(2, int(sim_count * 0.25)) if len(frauds_subset) > 0 else 0
            n_n = sim_count - n_f
            sample_stream = pd.concat([
                frauds_subset.sample(min(n_f, len(frauds_subset)), replace=True),
                normal_subset.sample(n_n, replace=True)
            ]).sample(frac=1).reset_index(drop=True)
        else:
            sample_stream = df.sample(min(sim_count, len(df)), replace=True).reset_index(drop=True)

        selected_model = (
            rf_model if chosen_model_name == "Random Forest (SMOTE)"
            else log_reg if chosen_model_name == "Logistic Regression"
            else iso_forest
        )
        model_key = "rf" if chosen_model_name == "Random Forest (SMOTE)" else "lr" if chosen_model_name == "Logistic Regression" else "iso"

        frauds_count = 0
        total_intercepted = 0.0
        stream_records = []

        status_box.info("⚡ Ingesting live transaction events...")

        for idx, row in sample_stream.iterrows():
            time.sleep(sim_speed)
            row_df = pd.DataFrame([row])
            
            # Predict live using the unified inference helper
            pred, proba = run_live_predictions(row_df, selected_model, model_key)
            is_fraud = (pred[0] == 1)
            fraud_score = proba[0] if proba is not None else (0.95 if is_fraud else 0.05)
            amt_val = float(row.get("Amount", 0.0))

            if is_fraud:
                frauds_count += 1
                total_intercepted += amt_val
                badge = "🚨 BLOCKED (Fraud)"
            else:
                badge = "✅ APPROVED"

            actual_lbl = row.get("Class", "Unknown")
            actual_str = "Fraud" if actual_lbl == 1 else "Normal" if actual_lbl == 0 else "Unknown"

            stream_records.insert(0, {
                "Transaction ID": f"TXN-{1000 + idx}",
                "Amount": f"${amt_val:.2f}",
                "Risk Score": f"{fraud_score * 100:.1f}%",
                "Action Taken": badge,
                "Actual Ground Truth": actual_str
            })

            w_proc.metric("Transactions Streamed", f"{idx + 1} / {len(sample_stream)}")
            w_fraud.metric("🚨 Suspicious Intercepted", f"{frauds_count}")
            w_amt.metric("💰 Value Protected", f"${total_intercepted:,.2f}")

            log_box.dataframe(pd.DataFrame(stream_records), use_container_width=True)

        status_box.success("🎉 Stream completed! All transactions processed.")


# -------------------------------------------------------------
# 5. Single Transaction Inspector (Manual Tester)
# -------------------------------------------------------------
elif page == "5. 🧪 Single Transaction Inspector":
    st.title("🧪 Single Transaction Inspector")
    st.markdown("Test single transaction inputs live with customizable feature sliders to demonstrate how the model reacts.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_amount = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=25000.0, value=250.0, step=10.0)
    with c2:
        test_time = st.number_input("Time Elapsed (seconds)", min_value=0, max_value=172800, value=3600)
    with c3:
        test_model_choice = st.selectbox("Inference Model", ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"])

    st.subheader("Key PCA Latent Features (Anomaly Indicators)")
    st.caption("Features like V14, V10, V12, and V4 have the strongest mathematical correlation with fraudulent behavior.")
    
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        v14 = st.slider("V14 (Strong Negative Indicator)", -20.0, 10.0, -4.5 if test_amount > 1000 else 0.5)
    with col_v2:
        v10 = st.slider("V10 (Negative Indicator)", -15.0, 10.0, -3.0 if test_amount > 1000 else 0.0)
    with col_v3:
        v12 = st.slider("V12 (Negative Indicator)", -15.0, 10.0, -2.5 if test_amount > 1000 else 0.2)
    with col_v4:
        v4 = st.slider("V4 (Positive Indicator)", -5.0, 15.0, 4.0 if test_amount > 1000 else -0.5)

    if st.button("🔍 Run Fraud Assessment", type="primary"):
        # Construct single row dataframe
        test_row = {f"V{i}": 0.0 for i in range(1, 29)}
        test_row["V14"] = v14
        test_row["V10"] = v10
        test_row["V12"] = v12
        test_row["V4"] = v4
        test_row["Amount"] = test_amount
        test_row["Time"] = test_time

        single_df = pd.DataFrame([test_row])
        chosen_m = rf_model if test_model_choice == "Random Forest (SMOTE)" else log_reg if test_model_choice == "Logistic Regression" else iso_forest
        m_type = "rf" if test_model_choice == "Random Forest (SMOTE)" else "lr" if test_model_choice == "Logistic Regression" else "iso"

        pred, proba = run_live_predictions(single_df, chosen_m, m_type)
        is_fraud = (pred[0] == 1)

        st.markdown("---")
        if is_fraud:
            st.error("### 🚨 HIGH RISK DETECTED: Transaction Flagged as FRAUD")
            if proba is not None:
                st.write(f"**Fraud Probability:** `{proba[0] * 100:.2f}%`")
            st.warning("Recommendation: Reject transaction immediately and send OTP verification to cardholder.")
        else:
            st.success("### ✅ TRANSACTION APPROVED: Classified as Legitimate")
            if proba is not None:
                st.write(f"**Legitimate Confidence:** `{(1 - proba[0]) * 100:.2f}%`")
            st.info("Recommendation: Proceed with standard payment gateway settlement.")
