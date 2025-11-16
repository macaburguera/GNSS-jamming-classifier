# train_xgb_cascade.py
from pathlib import Path
from typing import Optional, List, Tuple
import argparse, time, csv, json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
)
import xgboost as xgb

# ----------------------- Defaults -----------------------
DEFAULT_ARTIFACTS_ROOT = Path("./artifacts")
DEFAULT_JSR_BINS = [-10, 0, 10, 25, 40]
DEFAULT_CNR_BINS = [20, 30, 40, 50]


# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage XGBoost cascade: S1 NoJam-vs-Jam gate + S2 Jam-type classifier."
    )
    p.add_argument("--features_dir", type=str, default=None,
                   help="Folder with train/val/test _features.npz "
                        "(defaults to latest featselect_run_* / prep_run_* / mr_run_* under ./artifacts).")
    p.add_argument("--out", type=str, default=str(DEFAULT_ARTIFACTS_ROOT),
                   help="Root artifacts dir (defaults to ./artifacts).")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional name suffix (defaults to xgbcascade_run_YYYYmmdd_HHMMSS).")

    # Shared XGB knobs
    p.add_argument("--n_estimators", type=int, default=800, help="Upper cap; early stopping will stop earlier.")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.10)  # eta
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.6)
    p.add_argument("--max_bin", type=int, default=64)
    p.add_argument("--early_rounds", type=int, default=50)
    p.add_argument("--xgb_threads", type=int, default=1)

    # Stage-1 (binary) thresholding & imbalance
    p.add_argument("--s1_thresh", type=float, default=None,
                   help="Fixed Jam probability threshold (p>=τ -> Jam). If set, overrides mode.")
    p.add_argument("--s1_thresh_mode", type=str, default="fpr",
                   choices=["fpr", "recall", "f1", "youden"],
                   help="How to choose τ on VAL if --s1_thresh not set.")
    p.add_argument("--s1_target_fpr", type=float, default=0.02,
                   help="For mode=fpr: pick largest τ with FPR≤target.")
    p.add_argument("--s1_target_recall", type=float, default=0.90,
                   help="For mode=recall: pick smallest τ with Recall≥target.")
    p.add_argument("--s1_min_precision", type=float, default=0.80,
                   help="Guard for mode=recall: require Precision≥this when choosing τ.")
    p.add_argument("--s1_scale_pos_weight", type=float, default=None,
                   help="Override scale_pos_weight for S1 (pos=Jam). Default uses n_nojam/n_jam from TRAIN.")
    p.add_argument("--s1_undersample_jam", type=float, default=0.0,
                   help="Undersample ratio for Jam in S1 TRAIN (0=off; 1.0 -> match NoJam count).")

    # Binning (kept for compatibility)
    p.add_argument("--jsr_bins", type=float, nargs="*", default=DEFAULT_JSR_BINS)
    p.add_argument("--cnr_bins", type=float, nargs="*", default=DEFAULT_CNR_BINS)

    return p.parse_args()


# ----------------------- FS helpers (strict: featselect_run_*) -----------------------
from typing import Optional
from pathlib import Path

REQUIRED_FILES = ("train_features.npz", "val_features.npz", "test_features.npz")
PREFERRED_PREFIX = "prep_run_"

def _has_feature_triplet(d: Path) -> bool:
    return all((d / f).exists() for f in REQUIRED_FILES)

def latest_prep_dir(root: Path) -> Optional[Path]:
    """Return the most recent artifacts/featselect_run_* that contains the 3 NPZs."""
    if not root.exists():
        return None
    cands = [p for p in root.iterdir()
             if p.is_dir() and p.name.startswith(PREFERRED_PREFIX) and _has_feature_triplet(p)]
    if not cands:
        return None

    # Sort by latest mtime of the NPZs (more robust than lexicographic name)
    def latest_mtime(p: Path) -> float:
        return max((p / f).stat().st_mtime for f in REQUIRED_FILES)

    return max(cands, key=latest_mtime)

def ensure_features_dir(arg: Optional[str], out_root: Path) -> Path:
    """If --features_dir is given, use it; else pick latest featselect_run_* with NPZs."""
    if arg:
        d = Path(arg)
        if not d.exists():
            raise FileNotFoundError(f"--features_dir does not exist: {d}")
        if not _has_feature_triplet(d):
            raise FileNotFoundError(f"{d} is missing one of {REQUIRED_FILES}")
        return d

    d = latest_prep_dir(out_root)
    if d is None:
        raise FileNotFoundError(
            "No features_dir provided and no valid featselect_run_* found under "
            f"{out_root}. Run your feature selection script or pass --features_dir."
        )
    return d



# ----------------------- IO / plotting -----------------------
def load_split(features_dir: Path, split: str):
    d = np.load(features_dir / f"{split}_features.npz", allow_pickle=True)
    X = d["X"]
    y = d["y"]
    class_names = d["class_names"].tolist()
    feature_names = d["feature_names"].tolist()
    jsr = d.get("jsr", None)
    cnr = d.get("cnr", None)
    return X, y, class_names, feature_names, jsr, cnr


def save_csv_dict(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], normalize: bool,
                          title: str, out_png: Path):
    M = cm.astype(float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(7, 6))
    plt.imshow(M, interpolation="nearest")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def eval_by_bins(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str],
                 values: np.ndarray, edges, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    rows = []
    rows.append({"bin": "OVERALL", "count": int(y_true.size),
                 "acc": accuracy_score(y_true, y_pred),
                 "macro_f1": f1_score(y_true, y_pred, average="macro")})
    edges = np.asarray(edges, float)
    bin_labels = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges)-1)]
    for b in range(len(edges)-1):
        mask = (values >= edges[b]) & (values < edges[b+1])
        if not np.any(mask):
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        acc = accuracy_score(yt, yp); f1m = f1_score(yt, yp, average="macro")
        cm = confusion_matrix(yt, yp, labels=list(range(K)))
        np.savetxt(out_dir / f"cm_{tag}_bin{b}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, classes, False, f"{tag} CM {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}.png")
        plot_confusion_matrix(cm, classes, True,  f"{tag} CM (row-norm) {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}_rownorm.png")
        rows.append({"bin": bin_labels[b], "count": int(mask.sum()), "acc": acc, "macro_f1": f1m})
    save_csv_dict(out_dir / f"metrics_{tag}.csv", rows, ["bin","count","acc","macro_f1"])


# ----------------------- Utils -----------------------
def best_trees_from(booster: xgb.Booster, fallback: int) -> int:
    attrs = booster.attributes()
    if "best_iteration" in attrs:
        return int(attrs["best_iteration"]) + 1
    if "best_ntree_limit" in attrs:
        return int(attrs["best_ntree_limit"])
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        return int(booster.best_iteration) + 1
    return int(fallback)


def undersample_jam(X: np.ndarray, y_bin: np.ndarray, ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Keep all NoJam; keep min(count_nojam * ratio, count_jam) Jam samples."""
    if ratio <= 0:
        return X, y_bin
    idx_nojam = np.where(y_bin == 0)[0]
    idx_jam = np.where(y_bin == 1)[0]
    keep_jam = int(min(idx_jam.size, round(idx_nojam.size * ratio)))
    if keep_jam <= 0:
        return X, y_bin
    rng = np.random.default_rng(seed)
    sel_j = rng.choice(idx_jam, size=keep_jam, replace=False)
    keep = np.concatenate([idx_nojam, sel_j])
    return X[keep], y_bin[keep]


def pick_threshold_grid(pjam: np.ndarray, y_true_bin: np.ndarray, mode: str,
                        target_fpr: float, target_recall: float, min_precision: float) -> float:
    """Choose τ on a grid according to different criteria."""
    thrs = np.linspace(0.05, 0.95, 91)
    # metrics for reporting/debug
    def stats_at(t):
        y = (pjam >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y, labels=[0,1]).ravel()
        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))
        fpr  = fp / max(1, (fp + tn))
        tpr  = rec
        youden = tpr - fpr
        f1 = (2*prec*rec) / max(1e-12, (prec+rec))
        return prec, rec, fpr, youden, f1

    best_t = 0.5
    if mode == "fpr":
        # pick largest τ with FPR ≤ target
        best_t = 0.05
        for t in thrs:
            _, _, fpr, _, _ = stats_at(t)
            if fpr <= target_fpr:
                best_t = t
        return float(best_t)

    if mode == "recall":
        # pick smallest τ with Recall ≥ target and Precision ≥ guard
        for t in thrs:
            prec, rec, _, _, _ = stats_at(t)
            if rec >= target_recall and prec >= min_precision:
                return float(t)
        # fallback: highest recall
        best = max(thrs, key=lambda t: stats_at(t)[1])
        return float(best)

    if mode == "f1":
        best = max(thrs, key=lambda t: stats_at(t)[4])
        return float(best)

    if mode == "youden":
        best = max(thrs, key=lambda t: stats_at(t)[3])
        return float(best)

    return float(best_t)


# ----------------------- Main -----------------------
def main():
    a = parse_args()
    out_root = Path(a.out)
    feats_dir = ensure_features_dir(a.features_dir, out_root)
    print(f"Using features_dir: {feats_dir}")

    run_root = out_root / (a.run_name or ("xgbcascade_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)
    out_dir = run_root / f"cascade_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    Xtr, ytr, classes, feat_names, jsr_tr, cnr_tr = load_split(feats_dir, "train")
    Xva, yva, _, _, jsr_va, cnr_va = load_split(feats_dir, "val")
    Xte, yte, _, _, jsr_te, cnr_te = load_split(feats_dir, "test")

    # Cast to float32
    Xtr = Xtr.astype(np.float32, copy=False)
    Xva = Xva.astype(np.float32, copy=False)
    Xte = Xte.astype(np.float32, copy=False)
    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")
    print(f"#Features: {len(feat_names)} | Classes: {classes}")

    # Identify NoJam class id
    try:
        nojam_id = classes.index("NoJam")
    except ValueError:
        nojam_id = int(np.argmin([np.mean(ytr == i) for i in np.unique(ytr)]))
        print(f"[WARN] 'NoJam' not found. Assuming class id {nojam_id} is NoJam.")

    jam_ids = [i for i in range(len(classes)) if i != nojam_id]
    jam_names = [classes[i] for i in jam_ids]

    # ----------- Stage-1: Binary NoJam vs Jam -----------
    ytr_bin = (ytr != nojam_id).astype(int)
    yva_bin = (yva != nojam_id).astype(int)
    yte_bin = (yte != nojam_id).astype(int)

    # Optional undersampling of Jam in TRAIN for S1
    Xtr_s1, ytr_s1 = undersample_jam(Xtr, ytr_bin, a.s1_undersample_jam)

    # scale_pos_weight (pos=Jam)
    if a.s1_scale_pos_weight is not None:
        spw = float(a.s1_scale_pos_weight)
    else:
        n_nojam = int((ytr_s1 == 0).sum())
        n_jam   = int((ytr_s1 == 1).sum())
        spw = n_nojam / max(1, n_jam)
    print(f"[S1] scale_pos_weight = {spw:.4f}  (pos=Jam)")

    params_s1 = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr", "auc"],
        "tree_method": "hist",
        "max_depth": a.max_depth,
        "eta": a.learning_rate,
        "subsample": a.subsample,
        "colsample_bytree": a.colsample_bytree,
        "max_bin": a.max_bin,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": spw,
        "nthread": a.xgb_threads,
        "verbosity": 0,
    }

    dtr_s1 = xgb.DMatrix(Xtr_s1, label=ytr_s1)
    dva_s1 = xgb.DMatrix(Xva, label=yva_bin)   # validation is full VAL

    print("[S1] training (early stopping on VAL)…")
    booster_s1 = xgb.train(
        params=params_s1,
        dtrain=dtr_s1,
        num_boost_round=a.n_estimators,
        evals=[(dtr_s1, "train"), (dva_s1, "val")],
        early_stopping_rounds=a.early_rounds,
        verbose_eval=False,
    )
    s1_best_trees = best_trees_from(booster_s1, a.n_estimators)
    print(f"[S1] best n_estimators: {s1_best_trees}")

    # Threshold on VAL
    pjam_val = booster_s1.predict(dva_s1, iteration_range=(0, s1_best_trees))  # prob of Jam
    # Quick diagnostics
    try:
        roc = roc_auc_score(yva_bin, pjam_val)
        ap = average_precision_score(yva_bin, pjam_val)
        print(f"[S1] ROC-AUC={roc:.4f}  PR-AUC={ap:.4f}")
    except Exception:
        pass

    if a.s1_thresh is not None:
        s1_thr = float(a.s1_thresh)
        mode_used = "fixed"
    else:
        s1_thr = pick_threshold_grid(
            pjam_val, yva_bin, a.s1_thresh_mode, a.s1_target_fpr, a.s1_target_recall, a.s1_min_precision
        )
        mode_used = a.s1_thresh_mode
    print(f"[S1] Threshold τ = {s1_thr:.3f}  (mode={mode_used})")

    # S1 VAL metrics + plots
    yhat_s1_val = (pjam_val >= s1_thr).astype(int)
    cm_s1_val = confusion_matrix(yva_bin, yhat_s1_val, labels=[0,1])
    np.savetxt(out_dir / "s1_val_cm.csv", cm_s1_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_s1_val, ["NoJam","Jam"], False, "S1 VAL CM", out_dir / "s1_val_cm.png")
    plot_confusion_matrix(cm_s1_val, ["NoJam","Jam"], True,  "S1 VAL CM (row-norm)", out_dir / "s1_val_cm_rownorm.png")

    # S1 TEST metrics + plots
    dte_s1 = xgb.DMatrix(Xte, label=yte_bin)
    pjam_test = booster_s1.predict(dte_s1, iteration_range=(0, s1_best_trees))
    yhat_s1_test = (pjam_test >= s1_thr).astype(int)
    cm_s1_test = confusion_matrix(yte_bin, yhat_s1_test, labels=[0,1])
    np.savetxt(out_dir / "s1_test_cm.csv", cm_s1_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_s1_test, ["NoJam","Jam"], False, "S1 TEST CM", out_dir / "s1_test_cm.png")
    plot_confusion_matrix(cm_s1_test, ["NoJam","Jam"], True,  "S1 TEST CM (row-norm)", out_dir / "s1_test_cm_rownorm.png")

    # Per-bin (S1, binary)
    if jsr_va is not None and cnr_va is not None:
        eval_by_bins(yva_bin, yhat_s1_val, ["NoJam","Jam"], np.asarray(jsr_va,float),
                     a.jsr_bins, out_dir / "s1_val_by_jsr", "S1_VAL_JSR")
        eval_by_bins(yva_bin, yhat_s1_val, ["NoJam","Jam"], np.asarray(cnr_va,float),
                     a.cnr_bins, out_dir / "s1_val_by_cnr", "S1_VAL_CNR")
    if jsr_te is not None and cnr_te is not None:
        eval_by_bins(yte_bin, yhat_s1_test, ["NoJam","Jam"], np.asarray(jsr_te,float),
                     a.jsr_bins, out_dir / "s1_test_by_jsr", "S1_TEST_JSR")
        eval_by_bins(yte_bin, yhat_s1_test, ["NoJam","Jam"], np.asarray(cnr_te,float),
                     a.cnr_bins, out_dir / "s1_test_by_cnr", "S1_TEST_CNR")

    # Save S1 model + threshold
    booster_s1.save_model(str(out_dir / "booster_s1.json"))
    with open(out_dir / "s1_threshold.json", "w") as f:
        json.dump({
            "threshold": s1_thr,
            "best_trees": s1_best_trees,
            "mode": mode_used,
            "scale_pos_weight": spw
        }, f, indent=2)

    # ----------- Stage-2: Jam-type multiclass (only Jam rows) -----------
    def jam_subset(X, y):
        mask = (y != nojam_id)
        return X[mask], y[mask], mask

    Xtr_jam, ytr_jam, _ = jam_subset(Xtr, ytr)
    Xva_jam, yva_jam, mask_va_jam = jam_subset(Xva, yva)
    Xte_jam, yte_jam, mask_te_jam = jam_subset(Xte, yte)

    # Map original jam class ids -> 0..J-1 for S2
    jam_id_to_s2 = {cid: i for i, cid in enumerate(jam_ids)}
    s2_to_jam_id = {i: cid for cid, i in jam_id_to_s2.items()}
    ytr_s2 = np.array([jam_id_to_s2[c] for c in ytr_jam], dtype=int)
    yva_s2 = np.array([jam_id_to_s2[c] for c in yva_jam], dtype=int)
    yte_s2 = np.array([jam_id_to_s2[c] for c in yte_jam], dtype=int)

    params_s2 = {
        "objective": "multi:softprob",
        "num_class": len(jam_ids),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": a.max_depth,
        "eta": a.learning_rate,
        "subsample": a.subsample,
        "colsample_bytree": a.colsample_bytree,
        "max_bin": a.max_bin,
        "lambda": 1.0,
        "alpha": 0.0,
        "nthread": a.xgb_threads,
        "verbosity": 0,
    }

    dtr_s2 = xgb.DMatrix(Xtr_jam, label=ytr_s2)
    dva_s2 = xgb.DMatrix(Xva_jam, label=yva_s2)
    dte_s2 = xgb.DMatrix(Xte_jam, label=yte_s2)

    print("[S2] training (early stopping on jam-only VAL)…")
    booster_s2 = xgb.train(
        params=params_s2,
        dtrain=dtr_s2,
        num_boost_round=a.n_estimators,
        evals=[(dtr_s2, "train"), (dva_s2, "val")],
        early_stopping_rounds=a.early_rounds,
        verbose_eval=False,
    )
    s2_best_trees = best_trees_from(booster_s2, a.n_estimators)
    print(f"[S2] best n_estimators: {s2_best_trees}")

    # S2 jam-only VAL (oracle)
    proba_s2_val = booster_s2.predict(dva_s2, iteration_range=(0, s2_best_trees))
    yhat_s2_val = np.argmax(proba_s2_val, axis=1)
    cm_s2_val = confusion_matrix(yva_s2, yhat_s2_val, labels=list(range(len(jam_ids))))
    np.savetxt(out_dir / "s2_val_oracle_cm.csv", cm_s2_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_s2_val, jam_names, False, "S2 VAL (jam-only, oracle) CM", out_dir / "s2_val_oracle_cm.png")
    plot_confusion_matrix(cm_s2_val, jam_names, True,  "S2 VAL (jam-only, oracle) CM (row-norm)", out_dir / "s2_val_oracle_cm_rownorm.png")

    # S2 jam-only TEST (oracle)
    proba_s2_test = booster_s2.predict(dte_s2, iteration_range=(0, s2_best_trees))
    yhat_s2_test = np.argmax(proba_s2_test, axis=1)
    cm_s2_test = confusion_matrix(yte_s2, yhat_s2_test, labels=list(range(len(jam_ids))))
    np.savetxt(out_dir / "s2_test_oracle_cm.csv", cm_s2_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_s2_test, jam_names, False, "S2 TEST (jam-only, oracle) CM", out_dir / "s2_test_oracle_cm.png")
    plot_confusion_matrix(cm_s2_test, jam_names, True,  "S2 TEST (jam-only, oracle) CM (row-norm)", out_dir / "s2_test_oracle_cm_rownorm.png")

    # Save S2 model
    booster_s2.save_model(str(out_dir / "booster_s2.json"))

    # ----------- Final cascade predictions -----------
    # VAL
    yhat_final_val = np.full_like(yva, fill_value=-1)
    jam_idx_val = np.where(yhat_s1_val == 1)[0]
    nojam_idx_val = np.where(yhat_s1_val == 0)[0]
    yhat_final_val[nojam_idx_val] = nojam_id
    if len(jam_idx_val) > 0:
        dva_s2_gate = xgb.DMatrix(Xva[jam_idx_val])
        proba_gate_val = booster_s2.predict(dva_s2_gate, iteration_range=(0, s2_best_trees))
        pred_gate_val = np.argmax(proba_gate_val, axis=1)
        yhat_final_val[jam_idx_val] = np.array([s2_to_jam_id[i] for i in pred_gate_val], dtype=int)

    # TEST
    yhat_final_test = np.full_like(yte, fill_value=-1)
    jam_idx_test = np.where(yhat_s1_test == 1)[0]
    nojam_idx_test = np.where(yhat_s1_test == 0)[0]
    yhat_final_test[nojam_idx_test] = nojam_id
    if len(jam_idx_test) > 0:
        dte_s2_gate = xgb.DMatrix(Xte[jam_idx_test])
        proba_gate_test = booster_s2.predict(dte_s2_gate, iteration_range=(0, s2_best_trees))
        pred_gate_test = np.argmax(proba_gate_test, axis=1)
        yhat_final_test[jam_idx_test] = np.array([s2_to_jam_id[i] for i in pred_gate_test], dtype=int)

    # 6-class CMs
    cm_final_val = confusion_matrix(yva, yhat_final_val, labels=list(range(len(classes))))
    cm_final_test = confusion_matrix(yte, yhat_final_test, labels=list(range(len(classes))))
    np.savetxt(out_dir / "cascade_val_cm.csv", cm_final_val, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "cascade_test_cm.csv", cm_final_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_final_val, classes, False, "Cascade VAL CM", out_dir / "cascade_val_cm.png")
    plot_confusion_matrix(cm_final_val, classes, True,  "Cascade VAL CM (row-norm)", out_dir / "cascade_val_cm_rownorm.png")
    plot_confusion_matrix(cm_final_test, classes, False, "Cascade TEST CM", out_dir / "cascade_test_cm.png")
    plot_confusion_matrix(cm_final_test, classes, True,  "Cascade TEST CM (row-norm)", out_dir / "cascade_test_cm_rownorm.png")

    # Reports
    rep_val = classification_report(yva, yhat_final_val, target_names=classes, digits=6, output_dict=True)
    rep_test = classification_report(yte, yhat_final_test, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "cascade_val_report.csv", rows_val, ["name","precision","recall","f1-score","support"])
    save_csv_dict(out_dir / "cascade_test_report.csv", rows_test, ["name","precision","recall","f1-score","support"])

    # Per-bin on final cascade (VAL/TEST)
    if jsr_va is not None and cnr_va is not None:
        eval_by_bins(yva, yhat_final_val, classes, np.asarray(jsr_va,float),
                     a.jsr_bins, out_dir / "cascade_val_by_jsr", "CASCADE_VAL_JSR")
        eval_by_bins(yva, yhat_final_val, classes, np.asarray(cnr_va,float),
                     a.cnr_bins, out_dir / "cascade_val_by_cnr", "CASCADE_VAL_CNR")
    if jsr_te is not None and cnr_te is not None:
        eval_by_bins(yte, yhat_final_test, classes, np.asarray(jsr_te,float),
                     a.jsr_bins, out_dir / "cascade_test_by_jsr", "CASCADE_TEST_JSR")
        eval_by_bins(yte, yhat_final_test, classes, np.asarray(cnr_te,float),
                     a.cnr_bins, out_dir / "cascade_test_by_cnr", "CASCADE_TEST_CNR")

    # Summaries
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"S1 threshold = {s1_thr:.6f} (mode={mode_used})\n")
        f.write(f"S1 best_trees = {s1_best_trees}\n")
        f.write(f"S2 best_trees = {s2_best_trees}\n")
        f.write(f"S1 VAL acc = {accuracy_score(yva_bin, yhat_s1_val):.6f}, "
                f"macroF1 = {f1_score(yva_bin, yhat_s1_val, average='macro'):.6f}\n")
        f.write(f"S1 TEST acc = {accuracy_score(yte_bin, yhat_s1_test):.6f}, "
                f"macroF1 = {f1_score(yte_bin, yhat_s1_test, average='macro'):.6f}\n")
        f.write(f"Cascade VAL macroF1 = {f1_score(yva, yhat_final_val, average='macro'):.6f}\n")
        f.write(f"Cascade TEST macroF1 = {f1_score(yte, yhat_final_test, average='macro'):.6f}\n")

    print(f"[cascade] done -> {out_dir}")


if __name__ == "__main__":
    main()
