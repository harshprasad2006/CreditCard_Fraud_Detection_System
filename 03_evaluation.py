import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("model_results.csv")

# The trap: a model that predicts "normal" for everything
naive_preds = [0] * len(results)
naive_accuracy = accuracy_score(results["actual"], naive_preds)
print(f"Naive 'always predict normal' accuracy: {naive_accuracy:.4f}")
print(f"But it catches 0 out of {results['actual'].sum()} frauds.\n")
def evaluate_model(name, y_true, y_pred, y_proba=None):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return cm

cm_iso = evaluate_model("Isolation Forest", results["actual"], results["pred_isoforest"])
cm_lr = evaluate_model("Logistic Regression", results["actual"], results["pred_logreg"], results["pred_logreg_proba"])
cm_rf = evaluate_model("Random Forest + SMOTE", results["actual"], results["pred_rf_smote"], results["pred_rf_smote_proba"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
matrices = [(cm_iso, "Isolation Forest"), (cm_lr, "Logistic Regression"), (cm_rf, "Random Forest + SMOTE")]

for ax, (cm, title) in zip(axes, matrices):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("chart_confusion_matrices.png", dpi=150)
plt.show()
plt.figure(figsize=(8, 6))

for col_pred, col_proba, label in [
    ("pred_logreg", "pred_logreg_proba", "Logistic Regression"),
    ("pred_rf_smote", "pred_rf_smote_proba", "Random Forest + SMOTE"),
]:
    fpr, tpr, _ = roc_curve(results["actual"], results[col_proba])
    auc = roc_auc_score(results["actual"], results[col_proba])
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("chart_roc_curve.png", dpi=150)
plt.show()