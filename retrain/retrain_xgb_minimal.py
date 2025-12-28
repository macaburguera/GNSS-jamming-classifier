#!/usr/bin/env python3
"""
Retrain an XGBoost GNSS interference classifier using a minimal,
physically motivated feature subset selected from permutation importance.

This script:
- Uses the existing feature_extractor.py unchanged
- Extracts all features, then selects ONLY 10 top features
- Trains XGBoost FROM SCRATCH (no continued boosting)
- Is suitable for clean ablation and comparison vs DL models
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from joblib import dump as joblib_dump

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from xgboost import XGBClassifier

from feature_extractor import extract_features, FEATURE_NAMES


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Label CSVs produced by your GUI ----
LABELS_CSVS = [
    r"E:\Jammertest23\23.09.20 - Jammertest 2023 - Day 3\Roadside test\alt01004-labelled\alt01004_labels.csv",
    r"E:\Jammertest23\23.09.19 - Jammertest 2023 - Day 2\alt06-meac-afternoon-labelled\alt06 - Meaconing afternoon_labels.csv",
]


# ---- Output directory ----
OUT_ROOT = r"..\artifacts\xgb_minimal_features"
RUN_NAME = ""   # leave empty for auto timestamp

# ---- Classes ----
CLASSES_TO_USE = ["NoJam", "Chirp", "NB", "WB"]
IGNORED_LABELS = {"Interference"}

# ---- Dataset split ----
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED = 42

# =============================================================================
# FEATURE SELECTION (TOP-10 FROM PERMUTATION IMPORTANCE)
# =============================================================================
#
# These features were selected because they jointly cover:
# - Spectral sparsity (NB detection)
# - Spectral geometry (bandwidth / flatness)
# - Time–frequency dynamics (chirps / sweeps)
# - Envelope modulation (pulsed / AM interference)
#
# Redundant and unstable features were intentionally excluded.
#

SELECTED_FEATURES = [
    # Narrowband dominance: how much power is concentrated in few bins
    "nb_peak_salience",

    # Extreme spectral dominance (NB / CW vs others)
    "spec_peakiness_ratio",

    # Flatness vs structure (WB vs NB / Chirp)
    "spec_flatness",

    # Global spectral location
    "spec_centroid_Hz",

    # Effective occupied bandwidth
    "spec_spread_Hz",

    # Time–frequency variability (is energy moving?)
    "stft_centroid_std_Hz",

    # Speed of spectral movement (chirps / sweeps)
    "stft_centroid_absderiv_med_Hzps",

    # Chirp slope (direction and rate)
    "chirp_slope_Hzps",

    # Chirp linearity confirmation (true chirp vs noisy drift)
    "chirp_r2",

    # Envelope modulation strength (AM / pulsed emitters)
    "env_mod_index",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SampleRow:
    label: str
    iq_path: Path
    source_csv: str


# =============================================================================
# HELPERS
# =============================================================================

def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_label(lbl: Optional[str]) -> Optional[str]:
    if not lbl:
        return None
    key = lbl.lower().replace(" ", "").replace("_", "").replace("-", "")
    mapping = {
        "nojam": "NoJam",
        "clean": "NoJam",
        "chirp": "Chirp",
        "sweep": "Chirp",
        "nb": "NB",
        "narrowband": "NB",
        "wb": "WB",
        "wideband": "WB",
    }
    return mapping.get(key)


def apply_feature_selection(
    X: np.ndarray,
    feat_names: List[str],
    selected: List[str],
) -> Tuple[np.ndarray, List[str]]:
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    missing = [n for n in selected if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing selected features: {missing}")
    idx = [name_to_idx[n] for n in selected]
    return X[:, idx], selected


# =============================================================================
# MAIN
# =============================================================================

def main():

    rng = np.random.default_rng(SEED)

    out_root = Path(OUT_ROOT).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = RUN_NAME.strip() or f"xgb_minimal_{now_id()}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load labelled samples
    # -------------------------------------------------------------------------

    rows: List[SampleRow] = []

    for csv_path in map(Path, LABELS_CSVS):
        with open(csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                label = normalize_label(r.get("label"))
                if not label or label not in CLASSES_TO_USE:
                    continue

                iq_path = Path(r.get("iq_path", "")).expanduser()
                if not iq_path.exists():
                    continue

                rows.append(SampleRow(
                    label=label,
                    iq_path=iq_path,
                    source_csv=str(csv_path),
                ))

    if not rows:
        raise RuntimeError("No valid samples loaded from LABELS_CSVS.")

    print(f"[load] loaded {len(rows)} labelled samples")

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------

    X_list, y_list = [], []
    class_to_idx = {c: i for i, c in enumerate(CLASSES_TO_USE)}

    for i, r in enumerate(rows, 1):
        d = np.load(r.iq_path)
        iq = d["iq"]
        fs = float(d["fs_hz"])
        feats = extract_features(iq, fs)
        X_list.append(feats)
        y_list.append(class_to_idx[r.label])

        if i % 100 == 0:
            print(f"[features] extracted {i} samples")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)

    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------

    X, feat_names = apply_feature_selection(
        X,
        list(FEATURE_NAMES),
        SELECTED_FEATURES,
    )

    print(f"[features] using {len(feat_names)} selected features:")
    for f in feat_names:
        print(f"  - {f}")

    # -------------------------------------------------------------------------
    # Train / Val / Test split
    # -------------------------------------------------------------------------

    N = len(y)
    perm = rng.permutation(N)

    n_tr = int(TRAIN_FRAC * N)
    n_va = int(VAL_FRAC * N)

    idx_tr = perm[:n_tr]
    idx_va = perm[n_tr:n_tr + n_va]
    idx_te = perm[n_tr + n_va:]

    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_va], y[idx_va]
    Xte, yte = X[idx_te], y[idx_te]

    print(f"[split] train={len(ytr)} val={len(yva)} test={len(yte)}")

    # -------------------------------------------------------------------------
    # Model (low-dimensional, stable configuration)
    # -------------------------------------------------------------------------

    clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(CLASSES_TO_USE),
        eval_metric="mlogloss",
        max_depth=4,            # shallow trees for low feature count
        learning_rate=0.1,
        n_estimators=400,
        subsample=0.9,
        colsample_bytree=1.0,
        random_state=SEED,
        n_jobs=-1,
    )

    pipe = Pipeline([("clf", clf)])

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    print("[train] training XGBoost from scratch...")
    pipe.fit(Xtr, ytr)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def eval_split(name, Xt, yt):
        yp = pipe.predict(Xt)
        acc = accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, average="macro")
        print(f"[{name}] acc={acc:.4f} macroF1={mf1:.4f}")
        return {"acc": acc, "macro_f1": mf1}

    metrics = {
        "val":  eval_split("val",  Xva, yva),
        "test": eval_split("test", Xte, yte),
    }

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    model_path = out_dir / "xgb_minimal_features.joblib"
    joblib_dump(pipe, model_path)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "features_used.txt", "w", encoding="utf-8") as f:
        for name in feat_names:
            f.write(name + "\n")

    print("\nDONE")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
