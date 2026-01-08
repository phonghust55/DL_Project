"""
Main Training Pipeline for UNSW-NB15 Intrusion Detection.

This pipeline now includes:
1. Traditional ML: Logistic Regression (baseline)
2. Shallow Neural Network: MLPClassifier from sklearn
3. Deep Learning: Deep Neural Network with TensorFlow/Keras
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import preprocess
import utils
from logistic_regression_model import create_logistic_regression, param_grid as lr_param_grid
from neural_network import create_fnn, param_grid as fnn_param_grid
from deep_neural_network import DNNClassifier, get_model_configs

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def load_for_training(resplit_if_one_class: bool = True):
    """
    Load raw train/test from UNSW files.
    If train has only 1 class and resplit_if_one_class=True, combine and stratified split.
    """
    train_df, test_df = preprocess.load_raw_data()

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Không tìm thấy cột `label` trong dữ liệu UNSW_NB15.")

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    drop_cols = ["label"]
    if "attack_cat" in train_df.columns:
        drop_cols.append("attack_cat")

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    if resplit_if_one_class and y_train.nunique() < 2:
        print(
            "\n[WARN] File train chỉ có 1 class -> không train được. "
            "Sẽ gộp train+test và chia lại (stratified)."
        )
        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=y_all if y_all.nunique() > 1 else None,
        )

    print(f"Train shape: {X_train.shape} | label dist: {np.bincount(y_train)}")
    print(f"Test  shape: {X_test.shape}  | label dist: {np.bincount(y_test)}")
    return X_train, y_train, X_test, y_test


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def tune_and_eval(model_name: str, pipe: Pipeline, param_grid: dict, X_train, y_train, X_test, y_test):
    """Tune sklearn model with GridSearchCV and evaluate."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="recall",  # ưu tiên bắt được attack
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print("\nBest params:", gs.best_params_)
    best_model = gs.best_estimator_

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    metrics = utils.evaluate_model(best_model, X_test, y_test, safe_name, out_dir=PLOTS_DIR)

    model_path = os.path.join(SCRIPT_DIR, f"{safe_name}_best.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved: {model_path}")
    return metrics, gs.best_params_


def train_deep_learning(
    model_name: str,
    X_train_processed: np.ndarray,
    y_train: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> dict:
    """
    Train a Deep Neural Network using TensorFlow/Keras.
    
    Args:
        model_name: Name for the model
        X_train_processed: Preprocessed training features
        y_train: Training labels
        X_test_processed: Preprocessed test features
        y_test: Test labels
        config: Model configuration dictionary
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Training Deep Learning Model: {model_name}")
    print(f"{'='*80}")
    print(f"Config: {config}")
    
    # Create and train model
    dnn = DNNClassifier(
        input_dim=X_train_processed.shape[1],
        hidden_layers=config.get("hidden_layers", (512, 256, 128, 64, 32)),
        dropout_rate=config.get("dropout_rate", 0.3),
        l2_reg=config.get("l2_reg", 1e-4),
        learning_rate=config.get("learning_rate", 1e-3),
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 256),
        validation_split=0.1,
        use_residual=config.get("use_residual", False),
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    # Fit the model
    dnn.fit(X_train_processed, y_train.values if hasattr(y_train, 'values') else y_train)
    
    # Save model
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(SCRIPT_DIR, f"{safe_name}_best.keras")
    dnn.save(model_path)
    print(f"Saved: {model_path}")
    
    # Evaluate using sklearn-compatible interface
    y_pred = dnn.predict(X_test_processed)
    y_score = dnn.predict_proba(X_test_processed)[:, 1]
    
    # Use utils for evaluation and plotting
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Plot confusion matrix
    utils.plot_confusion_matrix(y_test_np, y_pred, model_name=safe_name, out_dir=PLOTS_DIR)
    
    # Plot ROC and PR curves
    try:
        curves = utils.plot_roc_pr(y_test_np, y_score, model_name=safe_name, out_dir=PLOTS_DIR)
        auc_score = curves["auc"]
        avg_precision = curves["avg_precision"]
    except Exception as e:
        print(f"Warning: Could not plot ROC/PR curves: {e}")
        auc_score = None
        avg_precision = None
    
    # Calculate metrics
    from sklearn.metrics import classification_report, accuracy_score
    
    report = classification_report(y_test_np, y_pred, output_dict=True, zero_division=0)
    
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
    print(classification_report(y_test_np, y_pred, digits=4, zero_division=0))
    
    # Plot training history
    if dnn.history is not None:
        plot_training_history(dnn.history, safe_name)
    
    return {
        "accuracy": float(report["accuracy"]),
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "auc": auc_score,
        "avg_precision": avg_precision,
    }


def plot_training_history(history, model_name: str) -> None:
    """Plot training history (loss and metrics over epochs)."""
    import matplotlib.pyplot as plt
    
    utils.ensure_dir(PLOTS_DIR)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[0, 0].plot(history.history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        axes[0, 1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0, 1].set_title("Accuracy over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if "auc" in history.history:
        axes[1, 0].plot(history.history["auc"], label="Train AUC")
        if "val_auc" in history.history:
            axes[1, 0].plot(history.history["val_auc"], label="Val AUC")
        axes[1, 0].set_title("AUC over Epochs")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if "recall" in history.history:
        axes[1, 1].plot(history.history["recall"], label="Train Recall")
        if "val_recall" in history.history:
            axes[1, 1].plot(history.history["val_recall"], label="Val Recall")
        axes[1, 1].set_title("Recall over Epochs")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Training History - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_training_history.png"), dpi=150)
    plt.close()
    print(f"Saved: {model_name}_training_history.png")


def main():
    utils.ensure_dir(PLOTS_DIR)

    # Load data
    X_train, y_train, X_test, y_test = load_for_training(resplit_if_one_class=True)

    # Preprocessor (missing value + encoding + scaling) is part of the pipeline
    preprocessor = preprocess.make_preprocessor(X_train)

    # Optional: some basic plots on raw data (numeric correlation)
    utils.plot_correlation_heatmap(pd.concat([X_train, y_train.rename("label")], axis=1), out_dir=PLOTS_DIR)

    results = []

    # =========================================================================
    # PART 1: Traditional Machine Learning (Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: TRADITIONAL MACHINE LEARNING")
    print("=" * 80)

    # 1) Linear Logistic Regression
    lr = create_logistic_regression(random_state=RANDOM_STATE)
    lr_pipe = build_pipeline(preprocessor, lr)
    lr_metrics, lr_best = tune_and_eval(
        "Linear Logistic Regression",
        lr_pipe,
        lr_param_grid(),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append({"Model": "Linear Logistic Regression", "Type": "Traditional ML", **lr_metrics, "best_params": str(lr_best)})

    # =========================================================================
    # PART 2: Shallow Neural Network (sklearn MLP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: SHALLOW NEURAL NETWORK (sklearn)")
    print("=" * 80)

    # 2) FNN using sklearn MLPClassifier
    fnn = create_fnn(random_state=RANDOM_STATE)
    fnn_pipe = build_pipeline(preprocessor, fnn)
    fnn_metrics, fnn_best = tune_and_eval(
        "FNN (MLPClassifier)",
        fnn_pipe,
        fnn_param_grid(),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    results.append({"Model": "FNN (MLPClassifier)", "Type": "Shallow NN", **fnn_metrics, "best_params": str(fnn_best)})

    # =========================================================================
    # PART 3: DEEP LEARNING (TensorFlow/Keras)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: DEEP LEARNING (TensorFlow/Keras)")
    print("=" * 80)

    # Preprocess data for deep learning (Keras models don't use sklearn pipeline)
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert sparse matrix to dense if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()
    
    print(f"\nPreprocessed data shape: Train={X_train_processed.shape}, Test={X_test_processed.shape}")

    # Get model configurations
    model_configs = get_model_configs()

    # 3) Deep Neural Network - Medium (Default)
    dnn_config = model_configs["DNN_Medium"]
    dnn_config["epochs"] = 50  # Adjust for faster training
    dnn_config["batch_size"] = 256
    
    dnn_metrics = train_deep_learning(
        "Deep_Neural_Network",
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        dnn_config,
    )
    results.append({
        "Model": "Deep Neural Network (Keras)",
        "Type": "Deep Learning",
        **dnn_metrics,
        "best_params": str(dnn_config),
    })

    # 4) Deep Neural Network with Residual Connections
    residual_config = model_configs["DNN_Residual"]
    residual_config["epochs"] = 50
    residual_config["batch_size"] = 256
    
    residual_metrics = train_deep_learning(
        "Residual_DNN",
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        residual_config,
    )
    results.append({
        "Model": "Residual DNN (Keras)",
        "Type": "Deep Learning",
        **residual_metrics,
        "best_params": str(residual_config),
    })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 80)
    
    # Display summary with key metrics
    display_cols = ["Model", "Type", "accuracy", "precision", "recall", "f1", "auc"]
    print(summary[display_cols].to_string(index=False))
    
    summary_path = os.path.join(SCRIPT_DIR, "model_results_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    
    # Find best model
    best_idx = summary["f1"].idxmax()
    best_model = summary.loc[best_idx, "Model"]
    best_f1 = summary.loc[best_idx, "f1"]
    print(f"\n🏆 Best Model (by F1-score): {best_model} with F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
