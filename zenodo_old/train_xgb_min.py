# train_xgb_min.py
from pathlib import Path
from typing import Optional
import argparse, time, csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

# ----------------------- Defaults -----------------------
DEFAULT_ARTIFACTS_ROOT = Path("./artifacts")
DEFAULT_JSR_BINS = [-10, 0, 10, 25, 40]
DEFAULT_CNR_BINS = [20, 30, 40, 50]

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Minimal XGBoost trainer with ES (Booster API; memory-friendly).")
    p.add_argument("--features_dir", type=str, default=None,
                   help="Folder with train/val/test _features.npz (defaults to latest featselect_run_* under ./artifacts).")
    p.add_argument("--out", type=str, default=str(DEFAULT_ARTIFACTS_ROOT),
                   help="Root artifacts dir (defaults to ./artifacts).")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional name suffix (defaults to xgbmin_run_YYYYmmdd_HHMMSS).")

    # Train knobs
    p.add_argument("--n_estimators", type=int, default=800)   # upper cap; ES will stop earlier
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.10)  # aka eta
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.6)
    p.add_argument("--max_bin", type=int, default=64)         # lower -> less RAM (for 'hist' tree)
    p.add_argument("--early_rounds", type=int, default=50)
    p.add_argument("--xgb_threads", type=int, default=1)      # 1 thread keeps RAM stable
    return p.parse_args()

# ----------------------- FS helpers -----------------------
def latest_prep_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("mr_topk_run_"):
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]

def ensure_features_dir(arg: Optional[str], out_root: Path) -> Path:
    if arg:
        d = Path(arg)
        if not d.exists():
            raise FileNotFoundError(f"--features_dir does not exist: {d}")
        return d
    d = latest_prep_dir(out_root)
    if d is None:
        raise FileNotFoundError(
            f"No features_dir provided and no featselect_run_* folders found under {out_root}.\n"
            f"Run your data preparation / feature selection script first."
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

def plot_confusion_matrix(cm: np.ndarray, classes, normalize: bool, title: str, out_png: Path):
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

def eval_by_bins(y_true, y_pred, classes, values, edges, out_dir: Path, tag: str):
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
    """Return the number of boosting rounds to use for prediction/refit."""
    attrs = booster.attributes()
    if "best_iteration" in attrs:
        return int(attrs["best_iteration"]) + 1
    if "best_ntree_limit" in attrs:  # very old models
        return int(attrs["best_ntree_limit"])
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        return int(booster.best_iteration) + 1
    return int(fallback)

# ----------------------- Main -----------------------
def main():
    a = parse_args()
    out_root = Path(a.out)
    feats_dir = ensure_features_dir(a.features_dir, out_root)
    print(f"Using features_dir: {feats_dir}")

    run_root = out_root / (a.run_name or ("xgbmin_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)

    # Load splits
    Xtr, ytr, classes, feat_names, _, _ = load_split(feats_dir, "train")
    Xva, yva, _, _, _, _ = load_split(feats_dir, "val")
    Xte, yte, _, _, _, _ = load_split(feats_dir, "test")

    # Cast to float32 (halves RAM vs float64)
    Xtr = Xtr.astype(np.float32, copy=False)
    Xva = Xva.astype(np.float32, copy=False)
    Xte = Xte.astype(np.float32, copy=False)

    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")
    print(f"#Features: {len(feat_names)} | Classes: {classes}")

    K = len(classes)

    # --------- Booster API params (stable across versions) ---------
    params = {
        "objective": "multi:softprob",    # probabilities; we'll argmax to class id
        "num_class": K,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": a.max_depth,
        "eta": a.learning_rate,
        "subsample": a.subsample,
        "colsample_bytree": a.colsample_bytree,
        "max_bin": a.max_bin,
        "reg_lambda": 1.0,   # preferred key names
        "reg_alpha": 0.0,
        "nthread": a.xgb_threads,
        "verbosity": 0,
    }

    # DMatrix (saves memory vs sklearn wrapper)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)

    # 1) Fit with early stopping on VAL
    print("[xgb-min] fitting with early stopping…")
    evals = [(dtr, "train"), (dva, "val")]
    booster_es = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=a.n_estimators,
        evals=evals,
        early_stopping_rounds=a.early_rounds,
        verbose_eval=False,
    )

    best_trees = best_trees_from(booster_es, a.n_estimators)
    print(f"[xgb-min] best n_estimators from ES: {best_trees}")

    # Evaluate VAL using ES booster (XGBoost >=2.0: use iteration_range, not ntree_limit)
    proba_val = booster_es.predict(dva, iteration_range=(0, best_trees))
    yhat_val = np.argmax(proba_val, axis=1)
    cm_val = confusion_matrix(yva, yhat_val, labels=list(range(K)))

    out_dir = run_root / f"xgb_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, "XGB VAL CM", out_dir / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True,  "XGB VAL CM (row-norm)", out_dir / "val_cm_rownorm.png")

    rep_val = classification_report(yva, yhat_val, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "val_report.csv", rows_val, ["name","precision","recall","f1-score","support"])

    # 2) Refit FINAL on TRAIN+VAL with best_trees (clean model), then TEST
    X_trval = np.vstack([Xtr, Xva]).astype(np.float32, copy=False)
    y_trval = np.concatenate([ytr, yva])
    dtrval = xgb.DMatrix(X_trval, label=y_trval)

    print("[xgb-min] refitting on TRAIN+VAL with best n_estimators…")
    booster_final = xgb.train(
        params=params,
        dtrain=dtrval,
        num_boost_round=best_trees,
        verbose_eval=False,
    )

    # Save model JSON
    booster_final.save_model(str(out_dir / "xgb_model.json"))

    # TEST
    proba_test = booster_final.predict(dte)  # already exactly best_trees trees
    yhat_test = np.argmax(proba_test, axis=1)
    cm_test = confusion_matrix(yte, yhat_test, labels=list(range(K)))
    np.savetxt(out_dir / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, "XGB TEST CM", out_dir / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True,  "XGB TEST CM (row-norm)", out_dir / "test_cm_rownorm.png")

    rep_test = classification_report(yte, yhat_test, target_names=classes, digits=6, output_dict=True)
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "test_report.csv", rows_test, ["name","precision","recall","f1-score","support"])

    # Per-JSR/CNR (if present)
    for split, y_true, y_pred in [("val", yva, yhat_val), ("test", yte, yhat_test)]:
        npz = np.load(feats_dir / f"{split}_features.npz", allow_pickle=True)
        jsr = npz.get("jsr", None); cnr = npz.get("cnr", None)
        if jsr is None or cnr is None:
            continue
        eval_by_bins(y_true, y_pred, classes, np.asarray(jsr, float), DEFAULT_JSR_BINS, out_dir / f"{split}_by_jsr", f"{split}_JSR")
        eval_by_bins(y_true, y_pred, classes, np.asarray(cnr, float), DEFAULT_CNR_BINS, out_dir / f"{split}_by_cnr", f"{split}_CNR")

    # Feature importance (gain)
    try:
        fscore_gain = booster_final.get_score(importance_type="gain")  # {"f0": gain, ...}
        rows = []
        for k, v in fscore_gain.items():
            try:
                idx = int(k[1:])
                fname = feat_names[idx] if 0 <= idx < len(feat_names) else k
            except Exception:
                fname = k
            rows.append({"feature": fname, "gain": float(v)})
        rows.sort(key=lambda r: r["gain"], reverse=True)
        save_csv_dict(out_dir / "feature_importance_gain.csv", rows, ["feature","gain"])
    except Exception:
        pass

    # Summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"val_macroF1 = {f1_score(yva, yhat_val, average='macro'):.6f}\n")
        f.write(f"test_macroF1 = {f1_score(yte, yhat_test, average='macro'):.6f}\n")

    print(f"[xgb-min] done -> {out_dir}")

if __name__ == "__main__":
    main()
