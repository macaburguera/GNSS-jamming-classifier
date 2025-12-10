#!/usr/bin/env python3
"""
Feature importance analysis for a trained model with 78 features.

Computes:
- Pearson correlation |r|
- Mutual information (MI)
- MICC = 0.5 * (normalized |r| + normalized MI)
- Permutation importance
- Mean |SHAP| values (if shap is installed)

Usage:
- Edit the CONFIG section (paths, target column, etc.)
- Run:  python feature_importance_analysis.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils import check_random_state
import joblib

# Try to import shap (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ========================
# CONFIG: EDIT THIS BLOCK
# ========================
MODEL_PATH = r"..\artifacts\jammertest_sim\xgb_run_20251209_131721\xgb_20251209_131741\xgb_trainval.joblib"
DATA_PATH = r"..\artifacts\jammertest_sim\prep_run_20251209_131531\test_features.npz"
TARGET_COLUMN = "target"          # Only used for CSV data
RANDOM_STATE = 42
N_SHAP_SAMPLES = 500              # Limit for shap sampling
N_PERM_REPEATS = 10               # Repeats for permutation importance
TOP_K_PRINT = 30                  # How many top features to print
# ========================


def load_data(path, target_col):
    """
    Load X, y and feature names from CSV or NPZ.

    CSV:
        - assumes one column named `target_col` contains y
        - all other columns are features

    NPZ:
        - expects arrays 'X' (N x n_features) and 'y' (N,)
        - if 'feature_names' exists, use it.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = df.drop(columns=[target_col]).columns.to_list()
    elif ext == ".npz":
        # allow_pickle=True because feature_names/class_names are dtype=object
        data = np.load(path, allow_pickle=True)
        if "X" not in data or "y" not in data:
            raise ValueError("NPZ must contain 'X' and 'y' arrays.")
        X = data["X"]
        y = data["y"]
        n_features = X.shape[1]

        if "feature_names" in data:
            raw_names = data["feature_names"]
            # raw_names is usually a 1D array of dtype=object
            feature_names = [str(x) for x in raw_names.tolist()]
        else:
            feature_names = [f"f{i}" for i in range(n_features)]
    else:
        raise ValueError("Unsupported data format. Use .csv or .npz")

    return X, y, feature_names


def detect_problem_type(y):
    """
    Heuristic: if <= 10 unique values -> classification, else regression.
    """
    unique_vals = np.unique(y)
    if len(unique_vals) <= 10:
        return "classification"
    return "regression"


def compute_correlations(X, y, problem_type):
    """
    Pearson/point-biserial-like correlations.

    Returns:
        corr_abs: array of |correlation| per feature (nan replaced by 0).
    """
    y_proc = y.copy()

    # For classification, encode labels as 0,1,2,...
    if problem_type == "classification" and not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y_proc = le.fit_transform(y)
    elif problem_type == "classification":
        # Map unique labels starting from 0
        _, inv = np.unique(y, return_inverse=True)
        y_proc = inv

    corr_abs = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        if np.all(xj == xj[0]):
            corr_abs.append(0.0)
            continue
        r = np.corrcoef(xj, y_proc)[0, 1]
        if np.isnan(r):
            corr_abs.append(0.0)
        else:
            corr_abs.append(abs(r))
    return np.asarray(corr_abs)


def compute_mutual_information(X, y, problem_type, random_state):
    """
    Mutual information per feature.
    """
    if problem_type == "classification":
        # Ensure labels are integers for MI
        if not np.issubdtype(y.dtype, np.integer):
            le = LabelEncoder()
            y_proc = le.fit_transform(y)
        else:
            y_proc = y
        mi = mutual_info_classif(X, y_proc, random_state=random_state)
    else:
        mi = mutual_info_regression(X, y, random_state=random_state)

    # Replace NaN with 0
    mi = np.nan_to_num(mi, nan=0.0)
    return mi


def compute_micc(corr_abs, mi):
    """
    Compute MICC as average of normalized |corr| and normalized MI.
    """
    corr_norm = corr_abs / corr_abs.max() if corr_abs.max() > 0 else corr_abs
    mi_norm = mi / mi.max() if mi.max() > 0 else mi
    micc = 0.5 * (corr_norm + mi_norm)
    return micc, corr_norm, mi_norm


def compute_permutation_importance_metric(model, X, y, problem_type, random_state, n_repeats=10):
    """
    Computes permutation importance using accuracy (classification) or R^2 (regression).
    """
    scoring = "accuracy" if problem_type == "classification" else "r2"
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    # result.importances_mean is usually shape (n_features,)
    return result.importances_mean


def compute_shap_importance(model, X, problem_type, n_features, random_state, max_samples=500):
    """
    Compute mean |SHAP| value per feature.

    Tries TreeExplainer first; falls back to KernelExplainer if needed.
    Returns a 1D array of length n_features.
    """
    if not SHAP_AVAILABLE:
        return None

    rng = check_random_state(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Try TreeExplainer first
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Generic black-box fallback (can be slower)
        if hasattr(model, "predict_proba") and problem_type == "classification":
            f = lambda data: model.predict_proba(data)
        else:
            f = lambda data: model.predict(data)
        explainer = shap.KernelExplainer(f, shap.sample(X, min(100, X.shape[0])))
        shap_values = explainer.shap_values(X_sample)

    # ---- Debug info (just to understand shapes) ----
    if isinstance(shap_values, list):
        print("SHAP: list of arrays with shapes:", [sv.shape for sv in shap_values])
    else:
        print("SHAP: array with shape:", np.array(shap_values).shape)
    # ------------------------------------------------

    # We want: one importance value per feature = mean |SHAP| over samples (and classes if multi-output).

    if isinstance(shap_values, list):
        # shape: list[n_outputs] of (n_samples, n_features)
        abs_vals = [np.abs(sv) for sv in shap_values]
        # mean over samples -> (n_outputs, n_features)
        mean_abs_per_output = np.stack([sv.mean(axis=0) for sv in abs_vals], axis=0)
        # mean over outputs -> (n_features,)
        mean_abs = mean_abs_per_output.mean(axis=0)
        return mean_abs

    # If it's a single ndarray
    vals = np.abs(np.array(shap_values))

    if vals.ndim == 2:
        # Typical: (n_samples, n_features)
        if vals.shape[1] == n_features:
            return vals.mean(axis=0)
        # Less typical: (n_outputs, n_features)
        if vals.shape[-1] == n_features:
            return vals.mean(axis=0)
        # Fallback: treat last axis as features
        axes = tuple(range(vals.ndim - 1))
        return vals.mean(axis=axes)

    if vals.ndim == 3:
        # We have seen (n_samples, n_features, n_outputs) in your run.
        # Let's detect which axis is the feature axis:
        shape = vals.shape
        feature_axis_candidates = [i for i, s in enumerate(shape) if s == n_features]
        if feature_axis_candidates:
            f_ax = feature_axis_candidates[0]
            # average over all other axes
            axes_to_avg = tuple(i for i in range(vals.ndim) if i != f_ax)
            mean_abs = vals.mean(axis=axes_to_avg)
            # After averaging, result should be (n_features,)
            return mean_abs

        # If no axis matches n_features, assume last axis is features
        axes = tuple(range(vals.ndim - 1))
        return vals.mean(axis=axes)

    # Higher dims: very unusual, but handle generically:
    axes = tuple(range(vals.ndim - 1))
    return vals.mean(axis=axes)


def ensure_1d(name, arr, n_features):
    """
    Ensure arr is a 1D vector of length n_features.

    - Flattens arr.
    - If length > n_features: trims.
    - If length < n_features: pads with zeros and warns.
    """
    arr = np.asarray(arr).ravel()

    if arr.shape[0] == n_features:
        return arr

    if arr.shape[0] > n_features:
        print(f"Warning: {name} has length {arr.shape[0]} > n_features {n_features}. "
              f"Trimming to first {n_features}.")
        return arr[:n_features]

    # length < n_features
    print(f"Warning: {name} has length {arr.shape[0]} < n_features {n_features}. "
          f"Padding with zeros.")
    padded = np.zeros(n_features, dtype=arr.dtype)
    padded[:arr.shape[0]] = arr
    return padded


def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    X, y, feature_names = load_data(DATA_PATH, TARGET_COLUMN)

    n_features = X.shape[1]
    if n_features != 78:
        print(f"Warning: expected 78 features, got {n_features}.")
    print(f"Using {n_features} features.")

    problem_type = detect_problem_type(y)
    print(f"Detected problem type: {problem_type}")

    print("Computing Pearson correlations...")
    corr_abs = compute_correlations(X, y, problem_type)

    print("Computing mutual information...")
    mi = compute_mutual_information(X, y, problem_type, RANDOM_STATE)

    print("Computing MICC...")
    micc, corr_norm, mi_norm = compute_micc(corr_abs, mi)

    print("Computing permutation importance...")
    perm_imp = compute_permutation_importance_metric(
        model, X, y, problem_type, RANDOM_STATE, n_repeats=N_PERM_REPEATS
    )

    if SHAP_AVAILABLE:
        print("Computing SHAP importance (this may take a while)...")
        shap_imp = compute_shap_importance(
            model, X, problem_type, n_features, RANDOM_STATE, max_samples=N_SHAP_SAMPLES
        )
    else:
        print("shap package not available. Skipping SHAP importance.")
        shap_imp = None

    # ---- Force everything to be 1D and aligned with n_features ----
    corr_abs   = ensure_1d("|corr|",         corr_abs,   n_features)
    mi         = ensure_1d("MI",             mi,         n_features)
    corr_norm  = ensure_1d("corr_norm",      corr_norm,  n_features)
    mi_norm    = ensure_1d("mi_norm",        mi_norm,    n_features)
    micc       = ensure_1d("MICC",           micc,       n_features)
    perm_imp   = ensure_1d("perm_importance", perm_imp,  n_features)
    if shap_imp is not None:
        shap_imp = ensure_1d("mean_abs_SHAP", shap_imp,  n_features)

    print("\nVector shapes after ensure_1d:")
    print("  |corr|:", corr_abs.shape)
    print("  MI:", mi.shape)
    print("  MICC:", micc.shape)
    print("  perm_importance:", perm_imp.shape)
    if shap_imp is not None:
        print("  mean_abs_SHAP:", shap_imp.shape)

    # Build results table
    data = {
        "feature": feature_names,
        "|corr|": corr_abs,
        "MI": mi,
        "corr_norm": corr_norm,
        "mi_norm": mi_norm,
        "MICC": micc,
        "perm_importance": perm_imp,
    }
    if shap_imp is not None:
        data["mean_abs_SHAP"] = shap_imp

    df = pd.DataFrame(data)
    df.sort_values("MICC", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\n=== Top features by MICC ===")
    print(df.head(TOP_K_PRINT).to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # Save full results
    out_path = "feature_importance_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
