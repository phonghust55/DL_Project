import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data files are in the same directory as this script
TRAIN_CSV = os.path.join(SCRIPT_DIR, "UNSW_NB15_training-set.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "UNSW_NB15_testing-set.csv")
TRAIN_PARQUET = os.path.join(SCRIPT_DIR, "Compressed", "UNSW_NB15_training-set.parquet")
TEST_PARQUET = os.path.join(SCRIPT_DIR, "Compressed", "UNSW_NB15_testing-set.parquet")

OUT_TRAIN_CSV = os.path.join(SCRIPT_DIR, "UNSW_NB15_train_preprocessed.csv")
OUT_TEST_CSV = os.path.join(SCRIPT_DIR, "UNSW_NB15_test_preprocessed.csv")
OUT_PREPROCESSOR = os.path.join(SCRIPT_DIR, "unsw_preprocessor.joblib")


def _load_unsw(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 train/test. Prefer parquet if possible; fallback to CSV."""
    if train_path.lower().endswith(".parquet"):
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        return train_df, test_df

    # Try different separators (comma first, then semicolon)
    for sep in [",", ";"]:
        try:
            train_df = pd.read_csv(train_path, sep=sep)
            if train_df.shape[1] > 5:  # Valid if more than 5 columns
                test_df = pd.read_csv(test_path, sep=sep)
                return train_df, test_df
        except Exception:
            continue
    
    # Fallback to semicolon
    train_df = pd.read_csv(train_path, sep=";")
    test_df = pd.read_csv(test_path, sep=";")
    return train_df, test_df


def _normalize_multi_dot_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some CSV exports contain numbers like '7.408.748.626.708.980' (many dots).
    Heuristic: keep first dot (decimal), remove all subsequent dots:
      '7.408.748.626.708.980' -> '7.408748626708980'

    If conversion to numeric works for >=90% of non-null values, keep the column numeric.
    """
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()

    def fix_cell(x):
        if not isinstance(x, str):
            return x
        s = x.strip()
        if s in {"", "-"}:
            return s
        if s.count(".") <= 1:
            return s
        digits_only = s.replace(".", "").replace("-", "")
        if not digits_only.isdigit():
            return s
        first = s.find(".")
        return s[: first + 1] + s[first + 1 :].replace(".", "")

    for c in obj_cols:
        sample = out[c].dropna().astype(str).head(200)
        if sample.empty:
            continue
        if (sample.str.count(r"\.") > 1).any():
            out[c] = out[c].map(fix_cell)
            converted = pd.to_numeric(out[c], errors="coerce")
            non_na = out[c].notna().sum()
            if non_na > 0 and (converted.notna().sum() / non_na) >= 0.9:
                out[c] = converted

    return out


def load_raw_data(auto_resplit: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw UNSW train/test.
    
    Args:
        auto_resplit: If True and train set has only 1 class, merge and stratified resplit.
    
    Returns:
        train_df, test_df with both classes in train set.
    """
    try:
        if os.path.exists(TRAIN_PARQUET) and os.path.exists(TEST_PARQUET):
            train_df, test_df = _load_unsw(TRAIN_PARQUET, TEST_PARQUET)
        else:
            train_df, test_df = _load_unsw(TRAIN_CSV, TEST_CSV)
    except Exception:
        train_df, test_df = _load_unsw(TRAIN_CSV, TEST_CSV)

    train_df = _normalize_multi_dot_numbers(train_df)
    test_df = _normalize_multi_dot_numbers(test_df)
    
    # Check if train set has both classes
    if auto_resplit and "label" in train_df.columns:
        train_labels = train_df["label"].astype(int)
        test_labels = test_df["label"].astype(int)
        
        # If train has only 1 class, merge and resplit
        if train_labels.nunique() < 2:
            print("\n" + "=" * 60)
            print("[WARNING] Train set chỉ có 1 class!")
            print(f"  - Train: {train_labels.value_counts().to_dict()}")
            print(f"  - Test:  {test_labels.value_counts().to_dict()}")
            print("=> Gộp train+test và chia lại (stratified 70/30)")
            print("=" * 60)
            
            # Merge both datasets
            all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            y_all = all_df["label"].astype(int)
            
            # Stratified split
            train_df, test_df = train_test_split(
                all_df,
                test_size=0.3,
                random_state=RANDOM_STATE,
                stratify=y_all,
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            print(f"\nSau khi chia lại:")
            print(f"  - Train: {train_df.shape[0]} rows, labels: {train_df['label'].value_counts().to_dict()}")
            print(f"  - Test:  {test_df.shape[0]} rows, labels: {test_df['label'].value_counts().to_dict()}")
    
    return train_df, test_df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # MLP + export CSV => prefer dense
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def _to_feature_df(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    Xt = preprocessor.transform(X)
    try:
        cols = preprocessor.get_feature_names_out()
        return pd.DataFrame(Xt, columns=cols)
    except Exception:
        return pd.DataFrame(Xt)


def preprocess_and_save() -> None:
    train_df, test_df = load_raw_data()

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Không tìm thấy cột target `label` trong UNSW_NB15.")

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    drop_cols = ["label"]
    if "attack_cat" in train_df.columns:
        drop_cols.append("attack_cat")

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    preprocessor = make_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_p = _to_feature_df(preprocessor, X_train)
    X_test_p = _to_feature_df(preprocessor, X_test)

    train_out = X_train_p.copy()
    train_out["label"] = y_train.to_numpy()

    test_out = X_test_p.copy()
    test_out["label"] = y_test.to_numpy()

    train_out.to_csv(OUT_TRAIN_CSV, index=False)
    test_out.to_csv(OUT_TEST_CSV, index=False)
    joblib.dump(preprocessor, OUT_PREPROCESSOR)

    print(f"Saved: {OUT_TRAIN_CSV} (shape={train_out.shape}) | label={np.bincount(y_train)}")
    print(f"Saved: {OUT_TEST_CSV} (shape={test_out.shape})  | label={np.bincount(y_test)}")
    print(f"Saved: {OUT_PREPROCESSOR}")


if __name__ == "__main__":
    preprocess_and_save()