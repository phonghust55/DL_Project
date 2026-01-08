import os

import numpy as np
import pandas as pd

import preprocess
import utils


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def main() -> None:
    utils.ensure_dir(PLOTS_DIR)

    # Load raw data and combine train + test for full data understanding
    # (consistent with main_training_pipeline.py)
    train_df, test_df = preprocess.load_raw_data()
    df = pd.concat([train_df, test_df], ignore_index=True)
    source = "raw UNSW (combined train + test)"

    print("=" * 80)
    print("STEP 1: DATA UNDERSTANDING (UNSW-NB15)")
    print("=" * 80)

    print(f"\nSource: {source}")
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nMissing values (top 20):")
    miss = df.isnull().sum().sort_values(ascending=False)
    print(miss.head(20))

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False))

    # Basic summary for numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 0:
        print("\nSummary statistics (numeric):")
        print(num_df.describe().T.head(30))

    # Correlation heatmap (numeric columns, limited)
    utils.plot_correlation_heatmap(df, out_dir=PLOTS_DIR, filename="correlation_heatmap.png", max_features=40)

    # Distribution plots: pick top numeric features by variance (avoid plotting dozens of columns)
    if num_df.shape[1] > 0:
        variances = num_df.var(numeric_only=True).sort_values(ascending=False)
        # Exclude label itself from distribution list
        cols = [c for c in variances.index.tolist() if c != "label"][:12]
        utils.plot_distribution(df, columns=cols, out_dir=PLOTS_DIR, bins=40)
        print(f"\nSaved distribution plots for: {cols} -> {PLOTS_DIR}/dist_<col>.png")

    print("\nDone look at:", PLOTS_DIR)


if __name__ == "__main__":
    main()


