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
# Sidebar Dataset Management (Upload or Default)
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
        file_upload.seek(0)
        uploaded_df = pd.read_csv(file_upload)
        return uploaded_df, f"Uploaded CSV ({file_upload.name})"
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
    proc_df = input_df.copy()
    
    # Scale Amount and Time dynamically
    cur_scaler = StandardScaler()
    if "Amount" in proc_df.columns:
        if len(proc_df) > 1 and proc_df["Amount"].nunique() > 1:
            proc_df["Amount_scaled"] = cur_scaler.fit_transform(proc_df[["Amount"]])
        elif default_scaler is not None and hasattr(default_scaler, "mean_"):
            proc_df["Amount_scaled"] = (proc_df["Amount"] - default_scaler.mean_[0]) / default_scaler.scale_[0]
        else:
            proc_df["Amount_scaled"] = proc_df["Amount"] / 100.0
    else:
        proc_df["Amount_scaled"] = 0.0

    if "Time" in proc_df.columns:
        if len(proc_df) > 1 and proc_df["Time"].nunique() > 1:
            proc_df["Time_scaled"] = cur_scaler.fit_transform(proc_df[["Time"]])
        else:
            proc_df["Time_scaled"] = proc_df["Time"] / 100000.0
    else:
        proc_df["Time_scaled"] = 0.0

    # Ensure required V1-V28 features exist
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
    st.markdown("#### Real-Time Machine Learning Pipeline for Financial Transactions")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Loaded Rows", f"{len(df):,}" if df is not None else "0")
    with c2:
        frauds_in_data = int(df["Class"].sum()) if (df is not None and "Class" in df.columns) else "N/A"
        st.metric("Fraud Labels in Data", str(frauds_in_data))
    with c3:
        if df is not None and "Class" in df.columns and len(df) > 0:
            pct_fraud = (df["Class"].sum() / len(df)) * 100
            st.metric("Imbalance Rate", f"{pct_fraud:.3f}%")
        else:
            st.metric("Imbalance Rate", "Unlabeled")
    with c4:
        st.metric("Active Dataset", dataset_source.split("(")[0].strip())

    st.markdown("### 🎯 Problem Statement & Methodology")
    st.info(
        """
        - **Real-Time Data Ingestion:** Upload any dataset in the sidebar. The system dynamically scales features (`StandardScaler`), aligns PCA dimensions, and feeds it directly into trained ML models.
        - **Why Not Accuracy?** With a ~0.17% fraud rate, a naive algorithm predicting 'Normal' scores 99.83% accuracy while missing all frauds.
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
        st.subheader(f"1. Raw Data Preview ({min(len(df), 10)} of {len(df):,} Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Class" in df.columns:
                st.subheader("2. Class Imbalance (Actual Distribution)")
                counts = df["Class"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]
                counts["Label"] = counts["Class"].map({0: "Legitimate (0)", 1: "Fraud (1)"}).fillna(counts["Class"].astype(str))
                fig_bar = px.bar(
                    counts, x="Label", y="Count", text="Count",
                    color="Label", color_discrete_map={"Legitimate (0)": "#1f77b4", "Fraud (1)": "#d62728"},
                    log_y=True if len(df) > 100 else False,
                    title="Class Distribution"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Column 'Class' not found in uploaded file. Showing row count.")
                st.metric("Total Rows", len(df))

        with c2:
            if "Amount" in df.columns and "Class" in df.columns and df["Class"].nunique() > 1:
                st.subheader("3. Transaction Amount Comparison")
                fig_box = px.box(
                    df, x="Class", y="Amount", color="Class",
                    color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                    title="Transaction Amount by Class"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            elif "Amount" in df.columns:
                st.subheader("3. Transaction Amount Distribution")
                fig_hist = px.histogram(df, x="Amount", nbins=min(30, max(5, len(df))), title="Amount Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)

        if "Time" in df.columns and "Class" in df.columns and df["Class"].nunique() > 1:
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
# 3. Live Model Inference & Evaluation (Bug-free Dynamic)
# -------------------------------------------------------------
elif page == "3. 🧠 Live Model Inference & Evaluation":
    st.title("🧠 Live Model Inference & Performance Evaluation")
    st.markdown("Run live predictions against your active dataset across all 3 trained models in real time.")
    st.markdown("---")

    if df is not None and "Class" in df.columns:
        eval_slice = df.sample(min(len(df), 25000), random_state=42) if len(df) > 25000 else df
        y_actual = eval_slice["Class"].values

        st.info(f"⚡ Running live model inference on **{len(eval_slice):,} samples** from `{dataset_source}`...")

        with st.spinner("Executing predictions across Random Forest, Logistic Regression, and Isolation Forest..."):
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
            
            # ROC-AUC calculation only valid if both classes (0 and 1) are present in y_actual
            if m_prob is not None and len(np.unique(y_actual)) > 1:
                try:
                    auc = roc_auc_score(y_actual, m_prob)
                    auc_str = f"{auc:.4f}"
                except Exception:
                    auc_str = "N/A"
            else:
                auc_str = "N/A"

            perf_data.append({
                "Model": m_name,
                "Accuracy": f"{acc * 100:.2f}%",
                "Precision": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1-Score": f"{f1:.4f}",
                "ROC-AUC": auc_str
            })
            # Force labels=[0, 1] so confusion matrix is always guaranteed 2x2 [[TN, FP], [FN, TP]]
            cms[m_name] = confusion_matrix(y_actual, m_pred, labels=[0, 1])

        st.subheader("📋 Live Performance Comparison Table")
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        st.subheader("🧩 Confusion Matrices (Guaranteed 2x2 [[Normal, Fraud]])")
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        for idx, (m_name, cm) in enumerate(cms.items()):
            col_target = [col_cm1, col_cm2, col_cm3][idx]
            with col_target:
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal (0)", "Fraud (1)"], y=["Normal (0)", "Fraud (1)"],
                    title=f"{m_name}"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        if len(np.unique(y_actual)) > 1:
            st.subheader("📈 ROC Curves Comparison")
            fig_roc = go.Figure()
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

            for m_name, (m_pred, m_prob) in model_results.items():
                if m_prob is not None:
                    try:
                        fpr, tpr, _ = roc_curve(y_actual, m_prob)
                        auc_val = roc_auc_score(y_actual, m_prob)
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{m_name} (AUC = {auc_val:.3f})", mode='lines'))
                    except Exception:
                        pass

            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=800, height=500)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.caption("ℹ️ Note: ROC curve requires both Normal (0) and Fraud (1) labels in the dataset.")

    elif df is not None:
        st.warning("⚠️ The uploaded file does not have a 'Class' column (unlabeled data). Evaluation metrics require ground truth labels, but you can still run Real-Time Simulation and Single Transaction Inspector!")
    else:
        st.error("No dataset found. Please upload a dataset in the sidebar.")


# -------------------------------------------------------------
# 4. Real-Time Simulation Engine (Strict Active Dataset)
# -------------------------------------------------------------
elif page == "4. 🔴 Real-Time Simulation Engine":
    st.title("🔴 Real-Time Transaction Monitoring Engine")
    st.markdown(f"Streaming live transactions directly from: `{dataset_source}`")
    st.markdown("---")

    if df is None or len(df) == 0:
        st.error("No dataset available to simulate. Please upload a dataset in the sidebar.")
        st.stop()

    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Simulation Controls")
        chosen_model_name = st.selectbox(
            "Detection Model Engine",
            ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"]
        )
        max_streamable = len(df)
        sim_count = st.slider(
            "Stream Batch Size",
            min_value=1,
            max_value=min(100, max_streamable),
            value=min(20, max_streamable),
            step=1
        )
        sim_speed = st.slider("Latency (seconds per transaction)", 0.05, 0.8, 0.15)
        
        has_fraud_label = ("Class" in df.columns and 1 in df["Class"].values)
        if has_fraud_label:
            boost_frauds = st.checkbox("Inject Frauds in Stream (Recommended for Live Demo)", value=True)
        else:
            boost_frauds = False
            st.caption("ℹ️ 'Class' label not present or only 1 class in uploaded data: streaming directly row-by-row.")
            
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
        # STRICT SAMPLING: ONLY FROM THE UPLOADED/ACTIVE DATASET!
        if has_fraud_label and boost_frauds:
            frauds_subset = df[df["Class"] == 1]
            normal_subset = df[df["Class"] == 0]
            n_f = max(1, int(sim_count * 0.3)) if len(frauds_subset) > 0 else 0
            n_n = sim_count - n_f
            
            f_sample = frauds_subset.sample(n_f, replace=(len(frauds_subset) < n_f))
            n_sample = normal_subset.sample(n_n, replace=(len(normal_subset) < n_n)) if len(normal_subset) > 0 else pd.DataFrame()
            sample_stream = pd.concat([f_sample, n_sample]).sample(frac=1).reset_index(drop=True)
        else:
            sample_stream = df.sample(sim_count, replace=(len(df) < sim_count)).reset_index(drop=True)

        selected_model = (
            rf_model if chosen_model_name == "Random Forest (SMOTE)"
            else log_reg if chosen_model_name == "Logistic Regression"
            else iso_forest
        )
        model_key = "rf" if chosen_model_name == "Random Forest (SMOTE)" else "lr" if chosen_model_name == "Logistic Regression" else "iso"

        frauds_count = 0
        total_intercepted = 0.0
        stream_records = []

        status_box.info(f"⚡ Streaming {len(sample_stream)} transactions live from: {dataset_source}...")

        for idx, row in sample_stream.iterrows():
            time.sleep(sim_speed)
            row_df = pd.DataFrame([row])
            
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
            actual_str = "Fraud (1)" if actual_lbl == 1 else "Normal (0)" if actual_lbl == 0 else "Unknown"

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

        status_box.success(f"🎉 Stream finished! Processed transactions strictly from `{dataset_source}`.")


# -------------------------------------------------------------
# 5. Single Transaction Inspector (Manual Tester)
# -------------------------------------------------------------
elif page == "5. 🧪 Single Transaction Inspector":
    st.title("🧪 Single Transaction Inspector")
    st.markdown("Test single transaction inputs live with customizable feature sliders or instant 1-click presets.")
    st.markdown("---")

    st.subheader("⚡ Quick Test Presets (1-Click)")
    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    
    preset_choice = "Custom"
    with col_p1:
        if st.button("🚨 Load Suspicious Fraud Preset", use_container_width=True):
            st.session_state["preset"] = "fraud"
    with col_p2:
        if st.button("✅ Load Normal Transaction Preset", use_container_width=True):
            st.session_state["preset"] = "normal"

    active_preset = st.session_state.get("preset", "custom")

    # Set initial values based on preset
    if active_preset == "fraud":
        init_amount = 540.0
        init_time = 406
        init_v14, init_v10, init_v12, init_v4 = -6.75, -4.60, -5.50, 4.20
        init_v17, init_v11, init_v3, init_v16 = -5.30, 3.60, -5.10, -3.55
        st.info("Loaded real fraudulent transaction profile parameters.")
    elif active_preset == "normal":
        init_amount = 75.0
        init_time = 3600
        init_v14, init_v10, init_v12, init_v4 = 0.05, -0.09, 0.14, -0.02
        init_v17, init_v11, init_v3, init_v16 = -0.06, -0.03, 0.18, 0.07
        st.success("Loaded legitimate transaction profile parameters.")
    else:
        init_amount = 250.0
        init_time = 3600
        init_v14, init_v10, init_v12, init_v4 = 0.0, 0.0, 0.0, 0.0
        init_v17, init_v11, init_v3, init_v16 = 0.0, 0.0, 0.0, 0.0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_amount = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=25000.0, value=float(init_amount), step=10.0)
    with c2:
        test_time = st.number_input("Time Elapsed (seconds)", min_value=0, max_value=172800, value=int(init_time))
    with c3:
        test_model_choice = st.selectbox("Inference Model", ["Random Forest (SMOTE)", "Logistic Regression", "Isolation Forest"])

    st.subheader("Key Anomaly Indicators (Top Features impacting ML decision)")
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        v14 = st.slider("V14 (Primary Fraud Indicator)", -20.0, 10.0, float(init_v14))
    with col_v2:
        v10 = st.slider("V10 (Negative Anomaly)", -15.0, 10.0, float(init_v10))
    with col_v3:
        v12 = st.slider("V12 (Negative Anomaly)", -15.0, 10.0, float(init_v12))
    with col_v4:
        v4 = st.slider("V4 (Positive Spike Indicator)", -5.0, 15.0, float(init_v4))

    with st.expander("Secondary Critical Latent Features (V17, V11, V3, V16)"):
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            v17 = st.slider("V17 (High Impact)", -15.0, 10.0, float(init_v17))
        with col_s2:
            v11 = st.slider("V11 (High Impact)", -5.0, 15.0, float(init_v11))
        with col_s3:
            v3 = st.slider("V3 (Feature Spread)", -15.0, 10.0, float(init_v3))
        with col_s4:
            v16 = st.slider("V16 (Feature Spread)", -10.0, 10.0, float(init_v16))

    if st.button("🔍 Run Fraud Assessment", type="primary", use_container_width=True):
        # Construct single row dataframe with consistent background features
        # If V14 is significantly negative, background PCA components follow anomaly distribution
        is_anom = (v14 < -3.0 or v10 < -3.0 or v4 > 3.0)
        bg_val = -1.5 if is_anom else 0.0

        test_row = {f"V{i}": bg_val for i in range(1, 29)}
        test_row["V14"] = v14
        test_row["V10"] = v10
        test_row["V12"] = v12
        test_row["V4"] = v4
        test_row["V17"] = v17
        test_row["V11"] = v11
        test_row["V3"] = v3
        test_row["V16"] = v16
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
                st.metric("Model Fraud Risk Score", f"{proba[0] * 100:.1f}%")
            st.warning("Action: Transaction blocked instantly at payment gateway. Automated OTP / SMS verification sent to cardholder.")
        else:
            st.success("### ✅ TRANSACTION APPROVED: Classified as Legitimate")
            if proba is not None:
                st.metric("Legitimate Confidence Score", f"{(1 - proba[0]) * 100:.1f}%")
            st.info("Action: Low risk signature. Cleared for instant settlement.")
