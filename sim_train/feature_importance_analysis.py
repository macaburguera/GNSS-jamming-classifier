#!/usr/bin/env python3
# feature_importance_analysis_v4_no_shap.py

"""
Feature importance + model adequacy for multi-class classification (XGB, ~78 features),
WITHOUT SHAP (robust, dependency-light).

A) Data-driven feature ↔ class signal:
   - MI (mutual information)
   - nMI = MI / H(Y)  (normalized by label entropy)

B) Model adequacy on an evaluation split (VAL or TEST):
   - Accuracy, Balanced Accuracy, Macro F1, Log Loss (if predict_proba works)
   - Confusion matrix + normalized confusion matrix
   - Classification report
   - Optional confidence plots if predict_proba works

C) Model-based feature usefulness (evaluation split):
   - Permutation importance measured as macro-F1 drop (custom scorer that respects label encoding)

Outputs in ./feature_importance/run_<timestamp>/:
  results_features.csv
  metrics_summary.txt
  tables/*.csv
  plots/*.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance
import joblib


# ========================
# CONFIG: EDIT THIS BLOCK
# ========================
MODEL_PATH = r"..\artifacts\finetuned\finetune_continue_20251216_160409\xgb_20251216_160409\xgb_finetuned_continue.joblib"
DATA_PATH  = r"..\artifacts\finetuned\finetune_continue_20251216_160409\features\test_features.npz"

FEATURES_DIR = Path(DATA_PATH).parent
TRAIN_PATH = str(FEATURES_DIR / "train_features.npz")
VAL_PATH   = str(FEATURES_DIR / "val_features.npz")
TEST_PATH  = str(FEATURES_DIR / "test_features.npz")

# Recommended discipline:
# - While iterating: EVAL_SPLIT="val"
# - Final report:    EVAL_SPLIT="test"
EVAL_SPLIT = "test"   # "val" or "test"

# For stable nMI: "trainval" is usually best
NMI_SOURCE = "trainval"  # "train" or "trainval"

TARGET_COLUMN = "target"
OUT_ROOT = Path("feature_importance")

RANDOM_STATE = 42

# Permutation importance cost = n_features * n_repeats model evals
N_PERM_REPEATS = 10
PERM_SCORING = "macro_f1"   # currently only macro_f1 is implemented (best default)

TOP_K = 30
PLOT_TOP_ANNOTATE = 14
# ========================


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data(path: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """
    Load X, y, feature_names, class_names from CSV or NPZ.

    NPZ expects arrays: 'X' (N x F), 'y' (N,)
    optionally: 'feature_names', 'class_names'
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = df.drop(columns=[target_col]).columns.to_list()
        return X, y, feature_names, None

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "X" not in data or "y" not in data:
            raise ValueError(f"{path} must contain 'X' and 'y' arrays.")
        X = data["X"]
        y = data["y"]

        if "feature_names" in data:
            feature_names = [str(x) for x in data["feature_names"].tolist()]
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        class_names = None
        if "class_names" in data:
            class_names = [str(x) for x in data["class_names"].tolist()]

        return X, y, feature_names, class_names

    raise ValueError("Unsupported data format. Use .csv or .npz")


def try_load_split(path: str, target_col: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]]:
    if not Path(path).exists():
        return None
    return load_data(path, target_col)


def fit_label_encoder(y_source: np.ndarray, class_names_from_file: Optional[List[str]] = None) -> Tuple[LabelEncoder, List[str]]:
    le = LabelEncoder()
    le.fit(y_source)
    if class_names_from_file is not None and len(class_names_from_file) == len(le.classes_):
        class_names = class_names_from_file
    else:
        class_names = [str(c) for c in le.classes_.tolist()]
    return le, class_names


def entropy_labels_nats(y_enc: np.ndarray) -> float:
    counts = np.bincount(y_enc)
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def compute_mi_nmi(X: np.ndarray, y_enc: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    MI estimated using sklearn's mutual_info_classif (kNN-based estimator).
    We standardize X first because kNN distances are scale-sensitive.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mi = mutual_info_classif(Xs, y_enc, random_state=random_state)
    mi = np.nan_to_num(mi, nan=0.0)

    Hy = entropy_labels_nats(y_enc)
    nmi = mi / Hy if Hy > 0 else np.zeros_like(mi)
    return mi, nmi, Hy


def align_proba_to_le(proba: np.ndarray, model_classes: Any, le: LabelEncoder) -> Optional[np.ndarray]:
    """
    Align predict_proba columns to le.classes_ order if model exposes classes_.
    """
    if model_classes is None:
        return None
    model_classes_list = list(model_classes)
    le_classes_list = list(le.classes_)
    idx = []
    for c in le_classes_list:
        if c not in model_classes_list:
            return None
        idx.append(model_classes_list.index(c))
    return proba[:, idx]


def compute_metrics(model, X: np.ndarray, y_true_enc: np.ndarray, le: LabelEncoder, class_names: List[str]) -> Dict[str, Any]:
    y_pred_raw = model.predict(X)

    # Convert predicted labels (strings/ints) to encoded ints matching y_true_enc
    try:
        y_pred_enc = le.transform(y_pred_raw)
    except Exception:
        y_pred_enc = np.asarray(y_pred_raw)
        if y_pred_enc.dtype.kind not in ("i", "u"):
            raise

    acc = accuracy_score(y_true_enc, y_pred_enc)
    bacc = balanced_accuracy_score(y_true_enc, y_pred_enc)
    f1m = f1_score(y_true_enc, y_pred_enc, average="macro")

    K = len(class_names)
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=np.arange(K))
    cm_norm = cm.astype(float)
    rs = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, rs, out=np.zeros_like(cm_norm), where=rs != 0)

    report_dict = classification_report(
        y_true_enc, y_pred_enc,
        labels=np.arange(K),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true_enc, y_pred_enc,
        labels=np.arange(K),
        target_names=class_names,
        zero_division=0
    )

    proba_aligned = None
    ll = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            model_classes = getattr(model, "classes_", None)
            proba_aligned = align_proba_to_le(proba, model_classes, le)
            if proba_aligned is not None:
                ll = log_loss(y_true_enc, proba_aligned, labels=np.arange(K))
        except Exception:
            proba_aligned = None
            ll = None

    return {
        "y_pred_enc": y_pred_enc,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": f1m,
        "log_loss": ll,
        "cm": cm,
        "cm_norm": cm_norm,
        "report_dict": report_dict,
        "report_text": report_text,
        "proba_aligned": proba_aligned,
    }


# -----------------------------
# Permutation importance (robust)
# -----------------------------
def make_encoded_macro_f1_scorer(le: LabelEncoder) -> Callable[[Any, np.ndarray, np.ndarray], float]:
    """
    Returns a scorer(estimator, X, y_true_enc) that:
      - estimator.predict(X) -> raw labels
      - le.transform -> encoded ints
      - compute macro F1 vs y_true_enc
    """
    def scorer(estimator, X, y_true_enc):
        y_pred_raw = estimator.predict(X)
        try:
            y_pred_enc = le.transform(y_pred_raw)
        except Exception:
            y_pred_enc = np.asarray(y_pred_raw)
        return float(f1_score(y_true_enc, y_pred_enc, average="macro"))
    return scorer


def compute_permutation_importance_macro_f1(
    model,
    X: np.ndarray,
    y_true_enc: np.ndarray,
    le: LabelEncoder,
    n_repeats: int,
    random_state: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      perm_drop_mean: (F,) average macro-F1 drop when feature is permuted
      baseline_score: macro-F1 baseline
      perm_std: (F,) std of drop across repeats (optional, returned separately)
    """
    scorer = make_encoded_macro_f1_scorer(le)
    baseline = scorer(model, X, y_true_enc)

    result = permutation_importance(
        model,
        X,
        y_true_enc,
        scoring=scorer,       # custom callable scorer
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    # For permutation_importance with a scorer, sklearn returns:
    # importances = baseline_score - permuted_score
    # importances_mean is therefore the mean *drop* (positive = important).
    perm_drop_mean = np.asarray(result.importances_mean)
    perm_drop_std = np.asarray(result.importances_std)

    return perm_drop_mean, float(baseline), perm_drop_std


# -------- Plot helpers --------
def save_cm_plot(cm: np.ndarray, class_names: List[str], out_png: Path, title: str, fmt_float: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            s = f"{cm[i, j]:.3f}" if fmt_float else str(int(cm[i, j]))
            ax.text(j, i, s, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_sorted_barh_all(features: List[str], values: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    idx = np.argsort(values)[::-1]
    feats = [features[i] for i in idx][::-1]
    vals = values[idx][::-1]
    fig_h = max(7, 0.22 * len(feats))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(feats, vals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_barh_top(features: List[str], values: np.ndarray, out_png: Path, title: str, xlabel: str, top_k: int) -> None:
    idx = np.argsort(values)[::-1][:top_k]
    feats = [features[i] for i in idx][::-1]
    vals = values[idx][::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.25 * len(feats))))
    ax.barh(feats, vals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_cumulative_curve(values: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    v = np.nan_to_num(np.asarray(values), nan=0.0)
    v = np.sort(v)[::-1]
    s = v.sum()
    cum = np.cumsum(v) / s if s > 0 else np.zeros_like(v)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(cum) + 1), cum)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_scatter(features: List[str], x: np.ndarray, y: np.ndarray, out_png: Path, title: str, xlabel: str, ylabel: str, annotate_top: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_norm = x / (np.max(x) + 1e-12)
    y_norm = y / (np.max(y) + 1e-12)
    score = x_norm + y_norm
    idx = np.argsort(score)[::-1][:annotate_top]
    for i in idx:
        ax.annotate(features[i], (x[i], y[i]), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_class_support(y_enc: np.ndarray, class_names: List[str], out_png: Path, title: str) -> None:
    counts = np.bincount(y_enc, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, counts)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_per_class_f1(report_dict: Dict[str, Any], class_names: List[str], out_png: Path) -> None:
    f1s = [report_dict.get(c, {}).get("f1-score", 0.0) for c in class_names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, f1s)
    ax.set_title("Per-class F1 (eval)")
    ax.set_ylabel("F1")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_confidence_plots(proba: np.ndarray, y_true_enc: np.ndarray, y_pred_enc: np.ndarray, out_dir: Path) -> None:
    conf = proba.max(axis=1)
    correct = (y_true_enc == y_pred_enc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf, bins=30)
    ax.set_title("Confidence histogram (max prob, eval)")
    ax.set_xlabel("max predicted probability")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[correct], bins=30, alpha=0.7, label="correct")
    ax.hist(conf[~correct], bins=30, alpha=0.7, label="wrong")
    ax.legend()
    ax.set_title("Confidence: correct vs wrong (eval)")
    ax.set_xlabel("max predicted probability")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_correct_vs_wrong.png", dpi=220)
    plt.close(fig)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Rank features descending: best rank = 1.
    Ties handled by stable ordering.
    """
    order = np.argsort(values)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def main() -> None:
    run_dir = OUT_ROOT / f"run_{_run_id()}"
    plots_dir = run_dir / "plots"
    tables_dir = run_dir / "tables"
    _ensure_dir(plots_dir)
    _ensure_dir(tables_dir)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading splits...")
    tr = try_load_split(TRAIN_PATH, TARGET_COLUMN)
    va = try_load_split(VAL_PATH, TARGET_COLUMN)
    te = try_load_split(TEST_PATH, TARGET_COLUMN)

    if te is None:
        raise FileNotFoundError(f"Missing TEST_PATH: {TEST_PATH}")

    first = tr or va or te
    assert first is not None
    _, _, feature_names, class_names_file = first
    n_features = len(feature_names)

    # Fit label encoder on train if possible (best), else val/test
    if tr is not None:
        Xtr, ytr, _, class_names_tr = tr
        le, class_names = fit_label_encoder(ytr, class_names_tr or class_names_file)
        ytr_enc = le.transform(ytr)
    else:
        ytr_enc = None
        Xtr = None
        if va is not None:
            Xv0, yv0, _, class_names_v0 = va
            le, class_names = fit_label_encoder(yv0, class_names_v0 or class_names_file)
        else:
            Xte0, yte0, _, class_names_te0 = te
            le, class_names = fit_label_encoder(yte0, class_names_te0 or class_names_file)

    K = len(class_names)

    # Load/encode val/test
    if va is not None:
        Xv, yv, _, _ = va
        yv_enc = le.transform(yv)
    else:
        Xv, yv_enc = None, None

    Xte, yte, _, _ = te
    yte_enc = le.transform(yte)

    # Pick eval split
    eval_name = "test" if EVAL_SPLIT.lower() == "test" else "val"
    if eval_name == "val":
        if Xv is None:
            print("[WARN] VAL split not found. Falling back to TEST for evaluation.")
            eval_name = "test"
            Xeval, yeval_enc = Xte, yte_enc
        else:
            Xeval, yeval_enc = Xv, yv_enc
    else:
        Xeval, yeval_enc = Xte, yte_enc

    # Pick nMI source
    nmi_name = "trainval" if NMI_SOURCE.lower() == "trainval" else "train"
    if nmi_name == "train":
        if Xtr is None or ytr_enc is None:
            print("[WARN] TRAIN split not found. Falling back to TEST for nMI (not ideal).")
            Xnmi, ynmi_enc = Xte, yte_enc
            nmi_name = "test_fallback"
        else:
            Xnmi, ynmi_enc = Xtr, ytr_enc
    else:
        if Xtr is None or ytr_enc is None or Xv is None or yv_enc is None:
            print("[WARN] TRAIN or VAL split missing. Falling back to TRAIN-only if possible, else TEST.")
            if Xtr is not None and ytr_enc is not None:
                Xnmi, ynmi_enc = Xtr, ytr_enc
                nmi_name = "train_fallback"
            else:
                Xnmi, ynmi_enc = Xte, yte_enc
                nmi_name = "test_fallback"
        else:
            Xnmi = np.vstack([Xtr, Xv])
            ynmi_enc = np.concatenate([ytr_enc, yv_enc])

    print(f"Computing MI/nMI on: {nmi_name} ...")
    mi, nmi, Hy = compute_mi_nmi(Xnmi, ynmi_enc, RANDOM_STATE)

    print(f"Evaluating model on: {eval_name} ...")
    metrics = compute_metrics(model, Xeval, yeval_enc, le, class_names)

    print(f"Computing permutation importance (macro-F1 drop) on: {eval_name} ...")
    perm_drop_mean, perm_baseline_f1, perm_drop_std = compute_permutation_importance_macro_f1(
        model, Xeval, yeval_enc, le, N_PERM_REPEATS, RANDOM_STATE
    )

    # Build results table
    df = pd.DataFrame({
        "feature": feature_names,
        "MI": mi,
        "nMI": nmi,
        "nMI_source": nmi_name,
        "H_Y_nats": Hy,
        "eval_split": eval_name,
        "perm_macroF1_drop_mean": perm_drop_mean,
        "perm_macroF1_drop_std": perm_drop_std,
        "perm_macroF1_baseline": perm_baseline_f1,
    })

    # Ranking columns (1 = best)
    df["rank_nMI"] = _rank_desc(df["nMI"].to_numpy())
    df["rank_perm_macroF1_drop"] = _rank_desc(df["perm_macroF1_drop_mean"].to_numpy())

    # Normalized helper columns (for combined sorting/plotting)
    nmi_arr = df["nMI"].to_numpy()
    nmi_norm = nmi_arr / (np.nanmax(nmi_arr) + 1e-12) if np.nanmax(nmi_arr) > 0 else np.zeros_like(nmi_arr)

    perm_arr = df["perm_macroF1_drop_mean"].to_numpy()
    perm_norm = perm_arr / (np.nanmax(perm_arr) + 1e-12) if np.nanmax(perm_arr) > 0 else np.zeros_like(perm_arr)

    df["nMI_norm"] = nmi_norm
    df["perm_norm"] = perm_norm
    df["nMI_plus_perm_norm"] = df["nMI_norm"] + df["perm_norm"]

    # Save CSV
    results_csv = run_dir / "results_features.csv"
    df.sort_values(["nMI_plus_perm_norm", "nMI"], ascending=False).to_csv(results_csv, index=False)

    # Save eval predictions
    y_pred_enc = metrics["y_pred_enc"]
    eval_pred_df = pd.DataFrame({
        "y_true": yeval_enc,
        "y_pred": y_pred_enc,
        "y_true_name": [class_names[i] for i in yeval_enc],
        "y_pred_name": [class_names[i] for i in y_pred_enc],
        "correct": (yeval_enc == y_pred_enc),
    })
    proba_aligned = metrics.get("proba_aligned", None)
    if proba_aligned is not None:
        eval_pred_df["confidence"] = proba_aligned.max(axis=1)
        for k, cname in enumerate(class_names):
            eval_pred_df[f"p_{cname}"] = proba_aligned[:, k]
        save_confidence_plots(proba_aligned, yeval_enc, y_pred_enc, plots_dir)

    eval_pred_df.to_csv(tables_dir / f"eval_predictions_{eval_name}.csv", index=False)

    # Confusion matrices
    cm = metrics["cm"]
    cm_norm = metrics["cm_norm"]
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(tables_dir / f"confusion_matrix_{eval_name}.csv")
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(tables_dir / f"confusion_matrix_{eval_name}_normalized.csv")

    save_cm_plot(cm, class_names, plots_dir / f"confusion_matrix_{eval_name}.png", f"Confusion matrix (counts) [{eval_name.upper()}]", fmt_float=False)
    save_cm_plot(cm_norm, class_names, plots_dir / f"confusion_matrix_{eval_name}_normalized.png", f"Confusion matrix (row-normalized) [{eval_name.upper()}]", fmt_float=True)

    # Support + per-class F1
    save_class_support(yeval_enc, class_names, plots_dir / f"class_support_{eval_name}.png", f"Class support (true labels) [{eval_name.upper()}]")
    save_per_class_f1(metrics["report_dict"], class_names, plots_dir / f"per_class_f1_{eval_name}.png")

    # nMI plots (ALL features)
    save_sorted_barh_all(
        feature_names,
        df["nMI"].to_numpy(),
        plots_dir / f"all_features_nMI_sorted_{nmi_name}.png",
        title=f"ALL features: nMI sorted (source={nmi_name})",
        xlabel="nMI = MI / H(Y)"
    )
    save_barh_top(feature_names, df["nMI"].to_numpy(), plots_dir / f"top_{TOP_K}_nMI_{nmi_name}.png",
                  title=f"Top {TOP_K}: nMI (source={nmi_name})", xlabel="nMI", top_k=TOP_K)
    save_cumulative_curve(df["nMI"].to_numpy(), plots_dir / f"cumulative_nMI_{nmi_name}.png",
                          title=f"Cumulative nMI (sorted) (source={nmi_name})", xlabel="Top-N features")

    # Permutation importance plots (ALL features)
    save_sorted_barh_all(
        feature_names,
        df["perm_macroF1_drop_mean"].to_numpy(),
        plots_dir / f"all_features_perm_macroF1_drop_sorted_{eval_name}.png",
        title=f"ALL features: permutation importance (macro-F1 drop) sorted (eval={eval_name})",
        xlabel="Macro-F1 drop when permuted"
    )
    save_barh_top(feature_names, df["perm_macroF1_drop_mean"].to_numpy(), plots_dir / f"top_{TOP_K}_perm_macroF1_drop_{eval_name}.png",
                  title=f"Top {TOP_K}: permutation importance (macro-F1 drop) (eval={eval_name})",
                  xlabel="Macro-F1 drop when permuted", top_k=TOP_K)
    save_cumulative_curve(df["perm_macroF1_drop_mean"].to_numpy(), plots_dir / f"cumulative_perm_macroF1_drop_{eval_name}.png",
                          title=f"Cumulative permutation importance (macro-F1 drop) (sorted) (eval={eval_name})",
                          xlabel="Top-N features")

    # Cross-view plot: data signal vs model usefulness
    save_scatter(
        feature_names,
        df["nMI"].to_numpy(),
        df["perm_macroF1_drop_mean"].to_numpy(),
        plots_dir / f"scatter_nMI_vs_perm_macroF1_drop_{nmi_name}_vs_{eval_name}.png",
        title=f"Data signal (nMI) vs Model usefulness (perm macro-F1 drop) [{nmi_name} vs {eval_name}]",
        xlabel="nMI = MI/H(Y)",
        ylabel="Permutation importance (macro-F1 drop)",
        annotate_top=PLOT_TOP_ANNOTATE
    )

    # TXT summary
    txt_path = run_dir / "metrics_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Feature importance + model adequacy (classification, no-SHAP)\n")
        f.write("===========================================================\n\n")
        f.write(f"MODEL_PATH  : {MODEL_PATH}\n")
        f.write(f"FEATURES_DIR: {FEATURES_DIR}\n")
        f.write(f"TRAIN_PATH  : {TRAIN_PATH} {'(FOUND)' if Path(TRAIN_PATH).exists() else '(MISSING)'}\n")
        f.write(f"VAL_PATH    : {VAL_PATH}   {'(FOUND)' if Path(VAL_PATH).exists() else '(MISSING)'}\n")
        f.write(f"TEST_PATH   : {TEST_PATH}  {'(FOUND)' if Path(TEST_PATH).exists() else '(MISSING)'}\n\n")

        f.write(f"nMI source  : {nmi_name}\n")
        f.write(f"EVAL split  : {eval_name}\n\n")

        f.write(f"N samples (eval): {Xeval.shape[0]}\n")
        f.write(f"N features     : {n_features}\n")
        f.write(f"N classes      : {K}\n")
        f.write(f"Classes        : {class_names}\n\n")

        f.write("Data-driven signal (MI / nMI)\n")
        f.write("------------------------------\n")
        f.write(f"H(Y) (nats)    : {Hy:.6f}\n")
        f.write("MI estimated with mutual_info_classif on StandardScaler(X).\n")
        f.write("nMI = MI / H(Y) is used to normalize informativeness by label entropy.\n\n")

        f.write("Model adequacy (eval)\n")
        f.write("---------------------\n")
        f.write(f"Accuracy          : {metrics['accuracy']:.6f}\n")
        f.write(f"Balanced accuracy : {metrics['balanced_accuracy']:.6f}\n")
        f.write(f"Macro F1          : {metrics['macro_f1']:.6f}\n")
        f.write(f"Log loss          : {metrics['log_loss']:.6f}\n" if metrics["log_loss"] is not None
                else "Log loss          : (not available; predict_proba missing or alignment failed)\n")
        f.write("\nClassification report\n")
        f.write("---------------------\n")
        f.write(metrics["report_text"])
        f.write("\n\n")

        f.write("Permutation importance (macro-F1 drop)\n")
        f.write("--------------------------------------\n")
        f.write(f"Scoring          : macro-F1 (custom scorer with LabelEncoder alignment)\n")
        f.write(f"Baseline macro-F1: {perm_baseline_f1:.6f}\n")
        f.write(f"Repeats          : {N_PERM_REPEATS}\n")
        f.write("Interpretation   : larger positive drop => feature is more necessary for performance on eval.\n")
        f.write("                 : near-zero => feature is unused/redundant for this model on this eval split.\n\n")

        f.write(f"Top {TOP_K} by nMI\n")
        f.write("-----------------\n")
        top_nmi = df.sort_values("nMI", ascending=False).head(TOP_K)[["feature", "nMI", "MI", "rank_nMI"]]
        f.write(top_nmi.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write(f"Top {TOP_K} by permutation importance (macro-F1 drop)\n")
        f.write("----------------------------------------------------\n")
        top_perm = df.sort_values("perm_macroF1_drop_mean", ascending=False).head(TOP_K)[
            ["feature", "perm_macroF1_drop_mean", "perm_macroF1_drop_std", "rank_perm_macroF1_drop"]
        ]
        f.write(top_perm.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write("Files written\n")
        f.write("------------\n")
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                f.write(f"- {p.relative_to(run_dir)}\n")

    print(f"[DONE] Outputs written to: {run_dir}")
    print(f"  - {results_csv}")
    print(f"  - {txt_path}")
    print(f"  - plots/: {len(list(plots_dir.glob('*.png')))} images")
    print(f"  - tables/: {len(list(tables_dir.glob('*.csv')))} csv files")


if __name__ == "__main__":
    main()
