#!/usr/bin/env python3
# retrain_xgb.py

"""
Fine-tune an existing 4-class synthetic XGBoost model on REAL labelled NPZ data
loaded from TWO (or more) *_labels.csv files produced by your GUI.

Use case (your case):
- Folder A: NoJam, Chirp, NB
- Folder B: NoJam, NB, WB
=> Train on the union while KEEPING the original 4 output classes:
   ["NoJam", "Chirp", "NB", "WB"]

Goal:
- Keep WB capability learned from synthetic model.
- Adapt decision boundaries for classes present in real data.

How:
- Continue boosting from the old booster (add EXTRA_ROUNDS trees),
  rather than refitting from scratch.

NO CLI: edit CONFIG and run:
    python retrain_xgb.py

Outputs:
  <OUT_ROOT>/<RUN_NAME>/
    xgb_<timestamp>/
      xgb_finetuned_continue.joblib
      metrics.json, summary.txt
      val/test confusion matrices + reports
      pred_log_test.csv
    features/
      train_features.npz, val_features.npz, test_features.npz  (with metadata)

Notes:
- Caching: if REUSE_SPLIT_FEATURES_IF_EXISTS=True, the script will reuse cached
  split features only if they were created from the SAME set of LABELS_CSVS.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import load as joblib_load
from joblib import dump as joblib_dump

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from xgboost import XGBClassifier

from feature_extractor import extract_features, FEATURE_NAMES


# =============================================================================
# CONFIG (EDIT THESE)
# =============================================================================

# REAL labelled dataset CSVs from your GUI (two folders => two CSVs)
LABELS_CSVS = [
    r"E:\Jammertest23\23.09.20 - Jammertest 2023 - Day 3\Roadside test\alt01004-labelled\alt01004_labels.csv",
    r"E:\Jammertest23\23.09.19 - Jammertest 2023 - Day 2\alt06-meac-afternoon-labelled\alt06 - Meaconing afternoon_labels.csv",
]

# Old synthetic-trained model (Pipeline or XGBClassifier)
MODEL_IN = r"..\artifacts\jammertest_sim\xgb_run_20251215_222542\xgb_20251215_222636\xgb_trainval.joblib"

# Output root folder
OUT_ROOT = r"../artifacts/finetuned"

# Run name (auto if empty)
RUN_NAME = ""  # e.g. "finetune_day3_two_folders"

# IMPORTANT: keep the original 4 classes so WB output remains possible
CLASSES_TO_USE = ["NoJam", "Chirp", "NB", "WB"]

# Any other labels are ignored (case-insensitive)
IGNORED_LABELS = {"Interference"}

# If the same NPZ appears in both CSVs, keep only the first occurrence
DEDUPLICATE_BY_PATH = True

# Splitting
SPLIT_MODE = "random" # time" or "random"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42

# Fine-tuning strategy
CONTINUE_BOOSTING = True        # <-- keep WB capability
EXTRA_ROUNDS = 250              # number of new trees to add per stage
FINETUNE_LEARNING_RATE = 0.05   # smaller = gentler updates (None to keep old)

BALANCE_CLASSES = True
FINAL_REFIT_ON_TRAINVAL = True  # after val eval, continue-boost on train+val

# Cache split features inside RUN folder (speeds reruns if RUN_NAME fixed)
REUSE_SPLIT_FEATURES_IF_EXISTS = True

# Optional "rehearsal" to prevent forgetting WB:
# Point this to an existing synthetic features folder that contains train_features.npz etc.
# Example: r"..\artifacts\jammertest_sim\prep_run_20251215_221122"
SYNTH_FEATURES_DIR = ""  # "" disables rehearsal

# If rehearsal enabled, how many synthetic samples to mix in per class (0 = all)
SYNTH_MAX_PER_CLASS = 1000
# Which synthetic classes to include (None = all). For WB retention, set ["WB"].
SYNTH_ONLY_CLASSES = ["WB"]

# Per-sample test log
WRITE_TEST_PRED_LOG = True


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SampleRow:
    label: str
    iq_path: Path
    block_idx: int
    gps_week: Optional[int]
    tow_s: Optional[float]
    utc_iso: str
    sbf_path: str
    source_csv: str  # for debugging


# =============================================================================
# Helpers
# =============================================================================

def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_path_maybe_relative(p: str, base_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    p1 = (base_dir / pp).resolve()
    if p1.exists():
        return p1
    return (base_dir / pp.name).resolve()


def normalize_label(lbl: str) -> Optional[str]:
    """
    Make label parsing robust across casing and minor variants.
    Returns canonical class name or None if empty/ignored.
    """
    if lbl is None:
        return None
    raw = str(lbl).strip()
    if not raw:
        return None

    low = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    ignored_low = {s.lower().replace(" ", "").replace("_", "").replace("-", "") for s in IGNORED_LABELS}
    if low in ignored_low:
        return None

    # canonical mapping
    mapping = {
        "nojam": "NoJam",
        "nojamming": "NoJam",
        "clean": "NoJam",
        "chirp": "Chirp",
        "sweep": "Chirp",   # if your GUI used "Sweep" sometimes
        "nb": "NB",
        "narrowband": "NB",
        "wb": "WB",
        "wideband": "WB",
    }
    return mapping.get(low, raw)  # fall back to raw if unknown


def read_labels_csv(csv_path: Path) -> List[SampleRow]:
    base_dir = csv_path.parent
    rows: List[SampleRow] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            label_raw = r.get("label")
            label = normalize_label(label_raw)
            if not label:
                continue

            iq_path_raw = (r.get("iq_path") or "").strip()
            if not iq_path_raw:
                continue

            iq_path = _resolve_path_maybe_relative(iq_path_raw, base_dir)

            try:
                block_idx = int(r.get("block_idx", "-1"))
            except Exception:
                block_idx = -1

            gps_week = None
            tow_s = None
            try:
                v = r.get("gps_week")
                gps_week = int(v) if v not in (None, "", "None") else None
            except Exception:
                pass
            try:
                v = r.get("tow_s")
                tow_s = float(v) if v not in (None, "", "None") else None
            except Exception:
                pass

            utc_iso = (r.get("utc_iso") or "").strip()
            sbf_path = (r.get("sbf_path") or "").strip()

            rows.append(SampleRow(
                label=label,
                iq_path=iq_path,
                block_idx=block_idx,
                gps_week=gps_week,
                tow_s=tow_s,
                utc_iso=utc_iso,
                sbf_path=sbf_path,
                source_csv=str(csv_path),
            ))
    return rows


def load_rows_from_csvs(csv_paths: List[Path]) -> List[SampleRow]:
    all_rows: List[SampleRow] = []
    for p in csv_paths:
        rs = read_labels_csv(p)
        print(f"[load] {p.name}: rows={len(rs)}")
        all_rows.extend(rs)

    if DEDUPLICATE_BY_PATH:
        seen = set()
        uniq: List[SampleRow] = []
        dup = 0
        for r in all_rows:
            key = str(r.iq_path.resolve()) if r.iq_path.exists() else str(r.iq_path)
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            uniq.append(r)
        if dup:
            print(f"[load] deduplicated by iq_path: removed {dup} duplicates")
        all_rows = uniq

    return all_rows


def safe_npz_load(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    except Exception:
        return None


def build_features_from_rows(
    rows: List[SampleRow],
    classes_to_use: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[SampleRow], List[Path], List[str]]:
    allowed = set(classes_to_use)
    name_to_idx = {c: i for i, c in enumerate(classes_to_use)}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    used_rows: List[SampleRow] = []
    used_paths: List[Path] = []

    kept = 0
    skipped_label = 0
    skipped_missing = 0
    skipped_bad = 0

    for r in rows:
        if r.label not in allowed:
            skipped_label += 1
            continue
        if not r.iq_path.exists():
            skipped_missing += 1
            continue

        npz = safe_npz_load(r.iq_path)
        if npz is None or ("iq" not in npz) or ("fs_hz" not in npz):
            skipped_bad += 1
            continue

        iq = npz["iq"]
        fs_hz_arr = npz["fs_hz"]
        fs = float(fs_hz_arr) if np.ndim(fs_hz_arr) == 0 else float(fs_hz_arr.ravel()[0])

        try:
            feats = extract_features(iq, fs)
            feats = np.asarray(feats, dtype=float).ravel()
        except Exception:
            skipped_bad += 1
            continue

        if feats.size != len(FEATURE_NAMES):
            raise RuntimeError(
                f"Feature length mismatch for {r.iq_path.name}: got {feats.size}, expected {len(FEATURE_NAMES)}"
            )

        X_list.append(feats)
        y_list.append(name_to_idx[r.label])
        used_rows.append(r)
        used_paths.append(r.iq_path)

        kept += 1
        if kept % 100 == 0:
            print(f"[features] extracted {kept} samples...")

    if not X_list:
        raise RuntimeError("No valid samples loaded. Check LABELS_CSVS and NPZ paths.")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int).reshape(-1)

    print(f"[dataset] kept={kept} | skipped_label={skipped_label} | skipped_missing={skipped_missing} | skipped_bad={skipped_bad}")
    return X, y, used_rows, used_paths, list(classes_to_use)


def make_splits(
    y: np.ndarray,
    used_rows: List[SampleRow],
    split_mode: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    N = y.size
    ntr = int(round(train_frac * N))
    nva = int(round(val_frac * N))

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N).astype(np.int64)
        return perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    gps_week = np.array([r.gps_week if r.gps_week is not None else -1 for r in used_rows], dtype=np.int64)
    tow_s = np.array([r.tow_s if r.tow_s is not None else -1.0 for r in used_rows], dtype=np.float64)
    block_idx = np.array([r.block_idx for r in used_rows], dtype=np.int64)

    order = np.lexsort((block_idx, tow_s, gps_week)).astype(np.int64)
    return order[:ntr], order[ntr:ntr + nva], order[ntr + nva:]


def compute_sample_weights(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    counts = np.bincount(y, minlength=K).astype(float)
    w = np.zeros(y.shape[0], dtype=float)
    for i in range(y.shape[0]):
        c = int(y[i])
        w[i] = 1.0 / max(1.0, counts[c])
    w /= max(1e-12, float(np.mean(w)))
    return w


def ensure_pipeline(model_obj) -> Pipeline:
    if isinstance(model_obj, Pipeline):
        return model_obj
    if isinstance(model_obj, XGBClassifier):
        return Pipeline([("clf", model_obj)])
    raise TypeError(f"Unsupported model type in MODEL_IN: {type(model_obj)}")


def clone_pipeline_with_new_xgb(pipe: Pipeline, xgb_params: Dict) -> Pipeline:
    steps = list(pipe.steps)
    new_steps = steps[:-1] + [(steps[-1][0], XGBClassifier(**xgb_params))]
    return Pipeline(new_steps)


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
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", color="white")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_split_npz(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    classes: List[str],
    feat_names: List[str],
    meta_rows: List[SampleRow],
    paths: List[Path],
    labels_csvs: List[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        class_names=np.array(classes, dtype=object),
        feature_names=np.array(feat_names, dtype=object),
        paths=np.array([str(p) for p in paths], dtype=object),
        gps_week=np.array([r.gps_week if r.gps_week is not None else -1 for r in meta_rows], dtype=np.int64),
        tow_s=np.array([r.tow_s if r.tow_s is not None else -1.0 for r in meta_rows], dtype=np.float64),
        block_idx=np.array([r.block_idx for r in meta_rows], dtype=np.int64),
        utc_iso=np.array([r.utc_iso for r in meta_rows], dtype=object),
        sbf_path=np.array([r.sbf_path for r in meta_rows], dtype=object),
        source_csv=np.array([r.source_csv for r in meta_rows], dtype=object),
        labels_sources=np.array(labels_csvs, dtype=object),
    )


def cached_features_match_sources(npz_path: Path, labels_csvs: List[str]) -> bool:
    try:
        d = np.load(npz_path, allow_pickle=True)
        if "labels_sources" not in d.files:
            return False
        cached = [str(x) for x in d["labels_sources"].tolist()]
        return cached == labels_csvs
    except Exception:
        return False


def eval_and_save_split(split_name: str, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_dir: Path) -> Dict[str, float]:
    K = len(classes)
    labels = list(range(K))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / f"{split_name}_cm.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, classes, False, f"XGB {split_name.upper()} CM", out_dir / f"{split_name}_cm.png")
    plot_confusion_matrix(cm, classes, True,  f"XGB {split_name.upper()} CM (row-norm)", out_dir / f"{split_name}_cm_rownorm.png")

    rep = classification_report(
        y_true, y_pred,
        labels=labels,                 # keep 4-class report even if class is missing
        target_names=classes,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    rows_rep = [{"name": k, **v} for k, v in rep.items() if isinstance(v, dict)]
    save_csv_dict(out_dir / f"{split_name}_report.csv", rows_rep, ["name", "precision", "recall", "f1-score", "support"])

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


# -------------------- optional rehearsal from synthetic caches --------------------

def load_synth_rehearsal(features_dir: Path, classes_target: List[str], feat_names_target: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads synthetic cached features and returns (Xr, yr) remapped to classes_target indices.

    Expected files: train_features.npz and/or val_features.npz (we'll try both if present).

    If SYNTH_ONLY_CLASSES is set, only those classes are returned.
    """
    Xs_all = []
    ys_all = []

    for split in ["train", "val"]:
        p = features_dir / f"{split}_features.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        Xs = d["X"]
        ys = d["y"].astype(int).reshape(-1)
        classes_src = d["class_names"].tolist()
        feat_src = d["feature_names"].tolist()

        if feat_src != feat_names_target:
            raise RuntimeError("Synthetic feature_names do not match current FEATURE_NAMES.")

        src_name_by_idx = {i: classes_src[i] for i in range(len(classes_src))}
        tgt_idx_by_name = {n: i for i, n in enumerate(classes_target)}

        keep_mask = np.ones(ys.shape[0], dtype=bool)
        if SYNTH_ONLY_CLASSES:
            allowed = set(SYNTH_ONLY_CLASSES)
            keep_mask = np.array([src_name_by_idx[int(v)] in allowed for v in ys], dtype=bool)

        ys_kept = ys[keep_mask]
        Xs_kept = Xs[keep_mask]

        # remap y to target indices by class name
        ys_remap = np.array([tgt_idx_by_name[src_name_by_idx[int(v)]] for v in ys_kept], dtype=int)

        Xs_all.append(Xs_kept)
        ys_all.append(ys_remap)

    if not Xs_all:
        return np.zeros((0, len(feat_names_target))), np.zeros((0,), dtype=int)

    Xr = np.vstack(Xs_all)
    yr = np.concatenate(ys_all)

    # per-class cap
    if SYNTH_MAX_PER_CLASS and SYNTH_MAX_PER_CLASS > 0:
        rng = np.random.default_rng(SEED)
        out_idx = []
        for c in range(len(classes_target)):
            idx_c = np.where(yr == c)[0]
            if idx_c.size == 0:
                continue
            if idx_c.size > SYNTH_MAX_PER_CLASS:
                idx_c = rng.choice(idx_c, size=SYNTH_MAX_PER_CLASS, replace=False)
            out_idx.append(idx_c)
        if out_idx:
            out_idx = np.concatenate(out_idx)
            Xr = Xr[out_idx]
            yr = yr[out_idx]

    return Xr, yr


# =============================================================================
# Main
# =============================================================================

def main():
    # --- resolve CSV list ---
    labels_csv_paths = [Path(p).expanduser().resolve() for p in LABELS_CSVS]
    for p in labels_csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"LABELS_CSV not found: {p}")

    labels_csvs_str = [str(p) for p in labels_csv_paths]

    model_in = Path(MODEL_IN).expanduser().resolve()
    if not model_in.exists():
        raise FileNotFoundError(f"MODEL_IN not found: {model_in}")

    out_root = Path(OUT_ROOT).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = RUN_NAME.strip() or f"finetune_continue_{_now_id()}"
    run_root = out_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    out_dir = run_root / f"xgb_{_now_id()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_dir = run_root / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    train_npz = feats_dir / "train_features.npz"
    val_npz   = feats_dir / "val_features.npz"
    test_npz  = feats_dir / "test_features.npz"

    # --- load/build real features ---
    can_reuse = (
        REUSE_SPLIT_FEATURES_IF_EXISTS
        and train_npz.exists() and val_npz.exists() and test_npz.exists()
        and cached_features_match_sources(train_npz, labels_csvs_str)
        and cached_features_match_sources(val_npz, labels_csvs_str)
        and cached_features_match_sources(test_npz, labels_csvs_str)
    )

    if can_reuse:
        print(f"[features] loading cached split features from: {feats_dir}")
        dtr = np.load(train_npz, allow_pickle=True)
        dva = np.load(val_npz, allow_pickle=True)
        dte = np.load(test_npz, allow_pickle=True)

        Xtr = dtr["X"]; ytr = dtr["y"].astype(int).reshape(-1)
        Xva = dva["X"]; yva = dva["y"].astype(int).reshape(-1)
        Xte = dte["X"]; yte = dte["y"].astype(int).reshape(-1)

        classes = dtr["class_names"].tolist()
        feat_names = dtr["feature_names"].tolist()

        paths_te = dte.get("paths", None)
        gpsw_te = dte.get("gps_week", None)
        tow_te = dte.get("tow_s", None)
        blk_te = dte.get("block_idx", None)
        utc_te = dte.get("utc_iso", None)
        sbf_te = dte.get("sbf_path", None)
        src_te = dte.get("source_csv", None)

    else:
        print("[load] reading labels CSVs...")
        rows = load_rows_from_csvs(labels_csv_paths)
        if not rows:
            raise RuntimeError("No rows found in labels CSVs.")

        print("[features] extracting features from NPZ...")
        X, y, used_rows, used_paths, classes = build_features_from_rows(rows, CLASSES_TO_USE)
        feat_names = list(FEATURE_NAMES)

        idx_tr, idx_va, idx_te = make_splits(y, used_rows, SPLIT_MODE, TRAIN_FRAC, VAL_FRAC, SEED)

        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xva, yva = X[idx_va], y[idx_va]
        Xte, yte = X[idx_te], y[idx_te]

        rows_tr = [used_rows[int(i)] for i in idx_tr]
        rows_va = [used_rows[int(i)] for i in idx_va]
        rows_te = [used_rows[int(i)] for i in idx_te]

        paths_tr = [used_paths[int(i)] for i in idx_tr]
        paths_va = [used_paths[int(i)] for i in idx_va]
        paths_te = [used_paths[int(i)] for i in idx_te]

        save_split_npz(train_npz, Xtr, ytr, classes, feat_names, rows_tr, paths_tr, labels_csvs_str)
        save_split_npz(val_npz,   Xva, yva, classes, feat_names, rows_va, paths_va, labels_csvs_str)
        save_split_npz(test_npz,  Xte, yte, classes, feat_names, rows_te, paths_te, labels_csvs_str)

        paths_te = np.array([str(p) for p in paths_te], dtype=object)
        gpsw_te = np.array([r.gps_week if r.gps_week is not None else -1 for r in rows_te], dtype=np.int64)
        tow_te  = np.array([r.tow_s if r.tow_s is not None else -1.0 for r in rows_te], dtype=np.float64)
        blk_te  = np.array([r.block_idx for r in rows_te], dtype=np.int64)
        utc_te  = np.array([r.utc_iso for r in rows_te], dtype=object)
        sbf_te  = np.array([r.sbf_path for r in rows_te], dtype=object)
        src_te  = np.array([r.source_csv for r in rows_te], dtype=object)

    K = len(classes)
    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")
    print(f"#Features: {len(feat_names)} | Classes: {classes}")

    # --- optional rehearsal data ---
    Xreh = np.zeros((0, Xtr.shape[1]), dtype=float)
    yreh = np.zeros((0,), dtype=int)
    if SYNTH_FEATURES_DIR.strip():
        synth_dir = Path(SYNTH_FEATURES_DIR).expanduser().resolve()
        print(f"[rehearsal] loading synthetic rehearsal from: {synth_dir}")
        Xreh, yreh = load_synth_rehearsal(synth_dir, classes, feat_names)
        print(f"[rehearsal] got X={Xreh.shape} y={yreh.shape}")

    # --- weights ---
    wtr = compute_sample_weights(ytr, K) if BALANCE_CLASSES else None

    # --- load old model ---
    print(f"[model] loading: {model_in}")
    base_obj = joblib_load(model_in)
    base_pipe = ensure_pipeline(base_obj)

    old_clf: XGBClassifier = base_pipe.named_steps.get("clf", base_pipe.steps[-1][1])
    old_params = old_clf.get_params()

    # enforce correct class count
    xgb_params = dict(old_params)
    xgb_params["num_class"] = K
    xgb_params.setdefault("objective", "multi:softmax")
    xgb_params.setdefault("eval_metric", "mlogloss")
    xgb_params["random_state"] = SEED
    xgb_params["n_jobs"] = -1

    if FINETUNE_LEARNING_RATE is not None:
        xgb_params["learning_rate"] = float(FINETUNE_LEARNING_RATE)

    # For continuing boosting: sklearn wrapper trains "n_estimators" more rounds.
    if CONTINUE_BOOSTING:
        xgb_params["n_estimators"] = int(EXTRA_ROUNDS)

    pipe = clone_pipeline_with_new_xgb(base_pipe, xgb_params)

    # --- train: continue boosting ---
    if CONTINUE_BOOSTING:
        old_booster = old_clf.get_booster()

        # Train on TRAIN (+ optional rehearsal)
        X_fit = Xtr
        y_fit = ytr
        w_fit = wtr

        if Xreh.shape[0] > 0:
            X_fit = np.vstack([X_fit, Xreh])
            y_fit = np.concatenate([y_fit, yreh])
            w_fit = compute_sample_weights(y_fit, K) if BALANCE_CLASSES else None

        fit_kwargs = {"clf__xgb_model": old_booster}
        if w_fit is not None:
            fit_kwargs["clf__sample_weight"] = w_fit

        print(f"[train] continue boosting (+{EXTRA_ROUNDS} trees) on TRAIN{' + rehearsal' if Xreh.shape[0] else ''}...")
        pipe.fit(X_fit, y_fit, **fit_kwargs)

    else:
        fit_kwargs = {}
        if wtr is not None:
            fit_kwargs["clf__sample_weight"] = wtr
        print("[train] refit from scratch (WARNING: may forget WB)...")
        pipe.fit(Xtr, ytr, **fit_kwargs)

    # --- eval VAL ---
    print("[eval] val...")
    yhat_val = pipe.predict(Xva)
    metrics = {"val": eval_and_save_split("val", yva, yhat_val, classes, out_dir)}

    # --- final refit on TRAIN+VAL (continue boosting again from the *current* model) ---
    if FINAL_REFIT_ON_TRAINVAL:
        print("[save] final continue-boost on TRAIN+VAL...")
        booster_now = pipe.named_steps["clf"].get_booster()

        X_trval = np.vstack([Xtr, Xva])
        y_trval = np.concatenate([ytr, yva])

        if Xreh.shape[0] > 0:
            X_trval = np.vstack([X_trval, Xreh])
            y_trval = np.concatenate([y_trval, yreh])

        w_trval = compute_sample_weights(y_trval, K) if BALANCE_CLASSES else None

        fit_kwargs = {"clf__xgb_model": booster_now}
        if w_trval is not None:
            fit_kwargs["clf__sample_weight"] = w_trval

        # add EXTRA_ROUNDS more trees on top
        pipe.named_steps["clf"].set_params(n_estimators=int(EXTRA_ROUNDS))
        pipe.fit(X_trval, y_trval, **fit_kwargs)

    # --- eval TEST ---
    print("[eval] test...")
    yhat_test = pipe.predict(Xte)
    metrics["test"] = eval_and_save_split("test", yte, yhat_test, classes, out_dir)

    # --- save model ---
    model_out = out_dir / "xgb_finetuned_continue.joblib"
    joblib_dump(pipe, model_out)

    summary = {
        "labels_csvs": labels_csvs_str,
        "model_in": str(model_in),
        "model_out": str(model_out),
        "classes": classes,
        "split_mode": SPLIT_MODE,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "continue_boosting": bool(CONTINUE_BOOSTING),
        "extra_rounds_each_stage": int(EXTRA_ROUNDS),
        "finetune_learning_rate": FINETUNE_LEARNING_RATE,
        "balance_classes": bool(BALANCE_CLASSES),
        "final_refit_on_trainval": bool(FINAL_REFIT_ON_TRAINVAL),
        "rehearsal_enabled": bool(SYNTH_FEATURES_DIR.strip()),
        "rehearsal_only_classes": SYNTH_ONLY_CLASSES,
        "rehearsal_max_per_class": SYNTH_MAX_PER_CLASS,
        "val_acc": metrics["val"]["acc"],
        "val_macro_f1": metrics["val"]["macro_f1"],
        "test_acc": metrics["test"]["acc"],
        "test_macro_f1": metrics["test"]["macro_f1"],
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"val_macroF1  = {summary['val_macro_f1']:.6f}\n")
        f.write(f"test_macroF1 = {summary['test_macro_f1']:.6f}\n")
        f.write(f"val_acc      = {summary['val_acc']:.6f}\n")
        f.write(f"test_acc     = {summary['test_acc']:.6f}\n")
        f.write(f"model_out    = {model_out}\n")

    # --- per-sample test log ---
    if WRITE_TEST_PRED_LOG:
        idx_to_name = {i: classes[i] for i in range(len(classes))}
        rows_log = []
        for i in range(len(yte)):
            rows_log.append({
                "iq_path": str(paths_te[i]) if paths_te is not None else "",
                "label_true": idx_to_name[int(yte[i])],
                "label_pred": idx_to_name[int(yhat_test[i])],
                "gps_week": int(gpsw_te[i]) if gpsw_te is not None else "",
                "tow_s": float(tow_te[i]) if tow_te is not None else "",
                "block_idx": int(blk_te[i]) if blk_te is not None else "",
                "utc_iso": str(utc_te[i]) if utc_te is not None else "",
                "sbf_path": str(sbf_te[i]) if sbf_te is not None else "",
                "source_csv": str(src_te[i]) if src_te is not None else "",
            })
        save_csv_dict(
            out_dir / "pred_log_test.csv",
            rows_log,
            ["iq_path", "label_true", "label_pred", "gps_week", "tow_s", "block_idx", "utc_iso", "sbf_path", "source_csv"]
        )

    print("\nDONE")
    print(f"Output folder: {out_dir}")
    print(f"Saved model:   {model_out}")


if __name__ == "__main__":
    main()
