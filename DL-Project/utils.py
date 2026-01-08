import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_distribution(
    df: pd.DataFrame,
    columns: Iterable[str],
    out_dir: str = "plots",
    bins: int = 40,
) -> None:
    """Histogram distributions for selected columns."""
    ensure_dir(out_dir)
    for col in columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), bins=bins, kde=True)
        plt.title(f"Distribution - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{col}.png"))
        plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    out_dir: str = "plots",
    filename: str = "correlation_heatmap.png",
    max_features: int = 40,
) -> None:
    """Correlation heatmap of numeric columns (optionally limited for readability)."""
    ensure_dir(out_dir)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        return

    # Limit features for readability (take top columns by variance)
    if num_df.shape[1] > max_features:
        vars_ = num_df.var(numeric_only=True).sort_values(ascending=False)
        cols = vars_.head(max_features).index.tolist()
        num_df = num_df[cols]

    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    out_dir: str = "plots",
) -> None:
    ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


def plot_roc_pr(
    y_true,
    y_score,
    model_name: str,
    out_dir: str = "plots",
) -> dict:
    """
    Plot ROC + PR curves. Requires probability scores for positive class.
    Returns dict with auc and avg_precision.
    """
    ensure_dir(out_dir)
    auc_score = roc_auc_score(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_roc_curve.png"))
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR (AP={avg_precision:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_pr_curve.png"))
    plt.close()

    return {"auc": float(auc_score), "avg_precision": float(avg_precision)}


def evaluate_model(model, X_test, y_test, model_name: str, out_dir: str = "plots") -> dict:
    """
    Evaluate a sklearn model/pipeline and save plots:
    - confusion matrix
    - ROC / PR (if predict_proba is available)
    """
    ensure_dir(out_dir)
    y_pred = model.predict(X_test)

    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    plot_confusion_matrix(y_test, y_pred, model_name=model_name, out_dir=out_dir)

    auc_score = None
    avg_precision = None
    if y_score is not None:
        try:
            curves = plot_roc_pr(y_test, y_score, model_name=model_name, out_dir=out_dir)
            auc_score = curves["auc"]
            avg_precision = curves["avg_precision"]
        except Exception:
            pass

    precision_val = float(report.get("1", {}).get("precision", 0.0))
    recall_val = float(report.get("1", {}).get("recall", 0.0))
    f1_val = float(report.get("1", {}).get("f1-score", 0.0))

    print(f"\n=== {model_name} Performance Metrics ===")
    print(f"Accuracy:           {report['accuracy']:.4f}")
    print(f"Precision (Attack): {precision_val:.4f}")
    print(f"Recall (Attack):    {recall_val:.4f}")
    print(f"F1-Score (Attack):  {f1_val:.4f}")
    if auc_score is not None:
        print(f"AUC-ROC:            {auc_score:.4f}")
    if avg_precision is not None:
        print(f"Avg Precision:      {avg_precision:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": float(report["accuracy"]),
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "auc": auc_score,
        "avg_precision": avg_precision,
    }


