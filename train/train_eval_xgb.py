# train_eval_xgb.py
from pathlib import Path
from typing import Optional
import argparse, time, csv
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from joblib import dump

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ----------------------- XGBoost availability -----------------------
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

DEFAULT_ARTIFACTS_ROOT = Path("../artifacts/jammertest_sim")
DEFAULT_JSR_BINS = [10, 25, 40, 60]
DEFAULT_CNR_BINS = [30, 40, 60]

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate XGBoost on cached features with per-JSR/CNR analysis.")
    p.add_argument("--features_dir", type=str, default=None,
                   help="Folder with train_features.npz / val_features.npz / test_features.npz "
                        "(defaults to latest prep_run_* under ./artifacts).")
    p.add_argument("--out", type=str, default=str(DEFAULT_ARTIFACTS_ROOT),
                   help="Root artifacts dir (defaults to ./artifacts).")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional name suffix for outputs (defaults to xgb_run_YYYYmmdd_HHMMSS).")
    return p.parse_args()

# ----------------------- FS helpers -----------------------
def latest_prep_dir(root: Path) -> Optional[Path]:
    """Return most recent ./artifacts/prep_run_* directory if present."""
    if not root.exists():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("prep_run_")],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None

def ensure_features_dir(arg: Optional[str], out_root: Path) -> Path:
    if arg:
        d = Path(arg)
        if not d.exists():
            raise FileNotFoundError(f"--features_dir does not exist: {d}")
        return d
    d = latest_prep_dir(out_root)
    if d is None:
        raise FileNotFoundError(
            f"No features_dir provided and no prep_run_* folders found under {out_root}.\n"
            f"Run data_preparation.py first."
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
    im = plt.imshow(M, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", color="white")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def eval_by_bins(y_true, y_pred, classes, values: np.ndarray, edges, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    rows = []

    acc_all = accuracy_score(y_true, y_pred)
    f1m_all = f1_score(y_true, y_pred, average="macro")
    rows.append({"bin": "OVERALL", "count": int(y_true.size), "acc": acc_all, "macro_f1": f1m_all})

    edges = np.asarray(edges, float)
    bin_labels = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges)-1)]

    for b in range(len(edges) - 1):
        mask = (values >= edges[b]) & (values < edges[b + 1])
        if not np.any(mask):
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        acc = accuracy_score(yt, yp); f1m = f1_score(yt, yp, average="macro")
        cm = confusion_matrix(yt, yp, labels=list(range(K)))

        np.savetxt(out_dir / f"cm_{tag}_bin{b}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, classes, normalize=False,
                              title=f"{tag} CM {bin_labels[b]}",
                              out_png=out_dir / f"cm_{tag}_bin{b}.png")
        plot_confusion_matrix(cm, classes, normalize=True,
                              title=f"{tag} CM (row-norm) {bin_labels[b]}",
                              out_png=out_dir / f"cm_{tag}_bin{b}_rownorm.png")

        rows.append({"bin": bin_labels[b], "count": int(mask.sum()), "acc": acc, "macro_f1": f1m})

    save_csv_dict(out_dir / f"metrics_{tag}.csv", rows, ["bin", "count", "acc", "macro_f1"])

# ----------------------- Main -----------------------
def main():
    if not HAVE_XGB:
        raise RuntimeError("xgboost not installed. `pip install xgboost`")

    a = parse_args()
    out_root = Path(a.out)

    feats_dir = ensure_features_dir(a.features_dir, out_root)
    print(f"Using features_dir: {feats_dir}")

    run_root = out_root / (a.run_name or ("xgb_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, classes, feat_names, _, _ = load_split(feats_dir, "train")
    Xva, yva, _, _, _, _ = load_split(feats_dir, "val")
    Xte, yte, _, _, _, _ = load_split(feats_dir, "test")

    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")
    print(f"#Features: {len(feat_names)} | Classes: {classes}")

    pipe = Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=len(classes),
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=600,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        ))
    ])

    grid = {
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.7, 1.0]
    }

    X_trval = np.vstack([Xtr, Xva]); y_trval = np.concatenate([ytr, yva])
    ps = PredefinedSplit(np.concatenate([-1 * np.ones_like(ytr), np.zeros_like(yva)]))

    print("[xgb] grid search…")
    search = GridSearchCV(pipe, grid, cv=ps, scoring="f1_macro", n_jobs=-1, refit=False, verbose=1)
    search.fit(X_trval, y_trval)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_root / f"xgb_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_params.txt", "w", encoding="utf-8") as f:
        for k, v in search.best_params_.items():
            f.write(f"{k} = {v}\n")
        f.write(f"best_val_macroF1 = {search.best_score_:.6f}\n")

    best = pipe
    best.set_params(**search.best_params_)
    best.fit(Xtr, ytr)

    yhat_val = best.predict(Xva)
    cm_val = confusion_matrix(yva, yhat_val, labels=list(range(len(classes))))
    np.savetxt(out_dir / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, "XGB VAL CM", out_dir / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True,  "XGB VAL CM (row-norm)", out_dir / "val_cm_rownorm.png")

    rep_val = classification_report(yva, yhat_val, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "val_report.csv", rows_val, ["name", "precision", "recall", "f1-score", "support"])

    best.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    dump(best, out_dir / "xgb_trainval.joblib")

    yhat_test = best.predict(Xte)
    cm_test = confusion_matrix(yte, yhat_test, labels=list(range(len(classes))))
    np.savetxt(out_dir / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, "XGB TEST CM", out_dir / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True,  "XGB TEST CM (row-norm)", out_dir / "test_cm_rownorm.png")

    rep_test = classification_report(yte, yhat_test, target_names=classes, digits=6, output_dict=True)
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "test_report.csv", rows_test, ["name", "precision", "recall", "f1-score", "support"])

    # Per-JSR/CNR (if present in NPZ caches)
    for split, y_true, y_pred in [("val", yva, yhat_val), ("test", yte, yhat_test)]:
        npz = np.load(feats_dir / f"{split}_features.npz", allow_pickle=True)
        jsr = npz.get("jsr", None); cnr = npz.get("cnr", None)
        if jsr is None or cnr is None:
            continue
        jsr = np.asarray(jsr, float); cnr = np.asarray(cnr, float)
        eval_by_bins(y_true, y_pred, classes, jsr, DEFAULT_JSR_BINS, out_dir / f"{split}_by_jsr", f"{split}_JSR")
        eval_by_bins(y_true, y_pred, classes, cnr, DEFAULT_CNR_BINS, out_dir / f"{split}_by_cnr", f"{split}_CNR")

    # Feature importance (gain)
    try:
        booster = best.named_steps["clf"]
        fscore_gain = booster.get_booster().get_score(importance_type="gain")
        rows = []
        for k, v in fscore_gain.items():
            try:
                idx = int(k[1:])
                fname = feat_names[idx] if 0 <= idx < len(feat_names) else k
            except Exception:
                fname = k
            rows.append({"feature": fname, "gain": float(v)})
        rows.sort(key=lambda r: r["gain"], reverse=True)
        save_csv_dict(out_dir / "feature_importance_gain.csv", rows, ["feature", "gain"])
    except Exception:
        pass

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"val_macroF1 = {f1_score(yva, yhat_val, average='macro'):.6f}\n")
        f.write(f"test_macroF1 = {f1_score(yte, yhat_test, average='macro'):.6f}\n")

    print(f"[xgb] done -> {out_dir}")

if __name__ == "__main__":
    main()
