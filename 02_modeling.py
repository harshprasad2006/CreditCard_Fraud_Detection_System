import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

# Scale 'Amount' and 'Time' — V1-V28 are already scaled (PCA output), these two aren't
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(columns=["Amount", "Time"])

X = df.drop(columns=["Class"])
y = df["Class"]

# stratify=y keeps the same fraud/normal ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Fraud cases:", y_train.sum())
print("Test shape:", X_test.shape, "Fraud cases:", y_test.sum())
from sklearn.ensemble import IsolationForest

# contamination = expected proportion of anomalies (~0.17% here)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.0017,
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train)

# Isolation Forest outputs -1 for anomaly, 1 for normal — convert to match our labels (1=fraud, 0=normal)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

print("Isolation Forest — predicted frauds:", y_pred_iso.sum())
print("Actual frauds in test set:", y_test.sum())
from sklearn.linear_model import LogisticRegression

# class_weight='balanced' automatically penalizes mistakes on the minority class (fraud) more heavily
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]  # probability of fraud

print("Logistic Regression — predicted frauds:", y_pred_lr.sum())
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts().to_dict())

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
import joblib

joblib.dump(iso_forest, "model_isolation_forest.pkl")
joblib.dump(log_reg, "model_logistic_regression.pkl")
joblib.dump(rf_model, "model_random_forest_smote.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save test set + predictions for evaluation in the next script
results = pd.DataFrame({
    "actual": y_test.values,
    "pred_isoforest": y_pred_iso,
    "pred_logreg": y_pred_lr,
    "pred_logreg_proba": y_pred_proba_lr,
    "pred_rf_smote": y_pred_rf,
    "pred_rf_smote_proba": y_pred_proba_rf,
})
results.to_csv("model_results.csv", index=False)
print("\n✅ Models and results saved.")