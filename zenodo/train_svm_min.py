# train_svm_min.py
from pathlib import Path
from typing import Optional, List
import argparse, time, csv
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from joblib import dump

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---------------- Defaults ----------------
DEFAULT_ARTIFACTS_ROOT = Path("./artifacts")
DEFAULT_JSR_BINS = [-10, 0, 10, 25, 40]
DEFAULT_CNR_BINS = [20, 30, 40, 50]

# -------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal SVM trainer (linear or RBF) with tiny grid and low RAM use."
    )
    p.add_argument("--model", choices=["linear", "rbf"], default="linear",
                   help="Which SVM to train.")
    p.add_argument("--features_dir", type=str, default=None,
                   help="Dir with train/val/test _features.npz. Defaults to latest mr_topk_run_*/mr_run_*/prep_run_* in ./artifacts.")
    p.add_argument("--out", type=str, default=str(DEFAULT_ARTIFACTS_ROOT),
                   help="Root artifacts dir.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional name (defaults to svmmin_run_YYYYmmdd_HHMMSS).")
    return p.parse_args()

# -------------- FS helpers ----------------
def latest_features_dir(root: Path) -> Optional[Path]:
    """Pick newest directory among featselect_run_*, mr_topk_run_*, mr_run_*, prep_run_*."""
    if not root.exists():
        return None
    prefixes: List[str] = ["mr_topk_run_"]
    cands = []
    for p in root.iterdir():
        if p.is_dir() and any(p.name.startswith(px) for px in prefixes):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.name, reverse=True)
    return cands[0]

def ensure_features_dir(arg: Optional[str], out_root: Path) -> Path:
    if arg:
        d = Path(arg)
        if not d.exists():
            raise FileNotFoundError(f"--features_dir does not exist: {d}")
        return d
    d = latest_features_dir(out_root)
    if d is None:
        raise FileNotFoundError(
            f"No features_dir provided and no mr_topk_run_*/mr_run_*/prep_run_* under {out_root}."
        )
    return d

# -------------- IO / plotting ----------------
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
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def eval_by_bins(y_true, y_pred, classes, values, edges, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    rows = []
    # OVERALL
    rows.append({
        "bin": "OVERALL",
        "count": int(y_true.size),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    })
    edges = np.asarray(edges, float)
    bin_labels = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges)-1)]
    for b in range(len(edges)-1):
        mask = (values >= edges[b]) & (values < edges[b+1])
        if not np.any(mask):
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        cm = confusion_matrix(yt, yp, labels=list(range(K)))
        np.savetxt(out_dir / f"cm_{tag}_bin{b}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, classes, False, f"{tag} CM {bin_labels[b]}",
                              out_dir / f"cm_{tag}_bin{b}.png")
        plot_confusion_matrix(cm, classes, True, f"{tag} CM (row-norm) {bin_labels[b]}",
                              out_dir / f"cm_{tag}_bin{b}_rownorm.png")
        rows.append({
            "bin": bin_labels[b],
            "count": int(mask.sum()),
            "acc": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro"),
        })
    save_csv_dict(out_dir / f"metrics_{tag}.csv", rows, ["bin","count","acc","macro_f1"])

# -------------- Trainer ----------------
def run_model(name, pipe, grid, Xtr, ytr, Xva, yva, Xte, yte, classes, feats_dir: Path, out_root: Path):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{name}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predefined split: train fit / val score
    X_trval = np.vstack([Xtr, Xva])
    y_trval = np.concatenate([ytr, yva])
    test_fold = np.concatenate([-1 * np.ones_like(ytr), np.zeros_like(yva)])
    ps = PredefinedSplit(test_fold)

    print(f"[{name}] grid search…")
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=ps,
        scoring="f1_macro",
        n_jobs=1,         # low RAM usage
        refit=False,
        verbose=2,
    )
    search.fit(X_trval, y_trval)

    # Save best params
    with open(out_dir / "best_params.txt", "w", encoding="utf-8") as f:
        for k, v in search.best_params_.items():
            f.write(f"{k} = {v}\n")
        f.write(f"best_val_macroF1 = {search.best_score_:.6f}\n")
    print("Best:", search.best_params_)

    # Refit on TRAIN, eval on VAL
    best = pipe
    best.set_params(**search.best_params_)
    best.fit(Xtr, ytr)

    yhat_val = best.predict(Xva)
    cm_val = confusion_matrix(yva, yhat_val, labels=list(range(len(classes))))
    np.savetxt(out_dir / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, f"{name} VAL CM", out_dir / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True, f"{name} VAL CM (row-norm)", out_dir / "val_cm_rownorm.png")
    rep_val = classification_report(yva, yhat_val, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "val_report.csv", rows_val, ["name","precision","recall","f1-score","support"])

    # Final fit on TRAIN+VAL, eval on TEST
    best.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    dump(best, out_dir / f"{name}_trainval.joblib")  # no lambdas -> safe

    yhat_test = best.predict(Xte)
    cm_test = confusion_matrix(yte, yhat_test, labels=list(range(len(classes))))
    np.savetxt(out_dir / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, f"{name} TEST CM", out_dir / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True, f"{name} TEST CM (row-norm)", out_dir / "test_cm_rownorm.png")
    rep_test = classification_report(yte, yhat_test, target_names=classes, digits=6, output_dict=True)
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / "test_report.csv", rows_test, ["name","precision","recall","f1-score","support"])

    # Per-JSR/CNR if present
    for split, y_true, y_pred in [("val", yva, yhat_val), ("test", yte, yhat_test)]:
        npz = np.load(feats_dir / f"{split}_features.npz", allow_pickle=True)
        jsr = npz.get("jsr", None); cnr = npz.get("cnr", None)
        if jsr is None or cnr is None:
            continue
        eval_by_bins(y_true, y_pred, classes, np.asarray(jsr, float), DEFAULT_JSR_BINS,
                     out_dir / f"{split}_by_jsr", f"{split}_JSR")
        eval_by_bins(y_true, y_pred, classes, np.asarray(cnr, float), DEFAULT_CNR_BINS,
                     out_dir / f"{split}_by_cnr", f"{split}_CNR")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"model = {name}\n")
        f.write(f"val_macroF1 = {f1_score(yva, yhat_val, average='macro'):.6f}\n")
        f.write(f"test_macroF1 = {f1_score(yte, yhat_test, average='macro'):.6f}\n")
    print(f"[{name}] done -> {out_dir}")

# -------------- Main ----------------
def main():
    a = parse_args()
    out_root = Path(a.out)
    feats_dir = ensure_features_dir(a.features_dir, out_root)
    print(f"Using features_dir: {feats_dir}")

    run_root = out_root / (a.run_name or ("svmmin_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, classes, feat_names, _, _ = load_split(feats_dir, "train")
    Xva, yva, _, _, _, _ = load_split(feats_dir, "val")
    Xte, yte, _, _, _, _ = load_split(feats_dir, "test")

    # Cast to float32 to save RAM
    Xtr = Xtr.astype(np.float32, copy=False)
    Xva = Xva.astype(np.float32, copy=False)
    Xte = Xte.astype(np.float32, copy=False)

    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")
    print(f"#Features: {len(feat_names)} | Classes: {classes}")

    if a.model == "linear":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,        # n_samples >> n_features
                max_iter=30000,
                tol=1e-3,
                random_state=42
            )),
        ])
        grid = {
            "scaler": [StandardScaler(), RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10, 90))],
            "clf__C": [0.3, 1, 3],
        }
        run_model("linear", pipe, grid, Xtr, ytr, Xva, yva, Xte, yte, classes, feats_dir, run_root)

    else:  # RBF
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42)),
        ])
        grid = {
            "scaler": [StandardScaler()],     # keep tiny for RAM
            "clf__C": [1, 10],
            "clf__gamma": ["scale", 1e-3],
        }
        run_model("rbf", pipe, grid, Xtr, ytr, Xva, yva, Xte, yte, classes, feats_dir, run_root)

if __name__ == "__main__":
    main()
