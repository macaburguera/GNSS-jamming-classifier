#!/usr/bin/env python3
"""
Validate a MINIMAL-feature XGB model (10 features) on a labelled NPZ dataset
produced by the SBF labelling GUI.

This version:
- Extracts the full 78-feature vector (unchanged)
- Selects ONLY the 10 features used during training
- Feeds those features to the minimal XGB model
- Logs FULL timing statistics
- Produces DETAILED confusion matrices:
    * absolute counts
    * row-normalized percentages
    * per-class totals and recall
- Exports CSV + PNG matching the level of detail in the PDF screenshots
"""

from pathlib import Path
import sys, json, csv, time
from typing import Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import load as joblib_load
from sklearn.metrics import confusion_matrix, accuracy_score

# ============================ USER VARIABLES ============================

LABELS_CSV = r"E:\Jammertest23\23.09.18 - Jammertest 2023 - Day 1\Altus06 - 150m\labelled\alt06001_labels.csv"
OUT_DIR    = r"E:\Jammertest23\23.09.18 - Jammertest 2023 - Day 1\plots\alt06001_eval_10feat"
MODEL_PATH = r"..\artifacts\xgb_minimal_features\xgb_minimal_20251228_135846\xgb_minimal_features.joblib"

SAVE_IMAGES         = False
SAVE_PER_SAMPLE_CSV = True
SUMMARY_JSON        = True
DEBUG_PRINT_SAMPLE_LABELS = True

MODEL_CLASS_NAMES = ["NoJam", "Chirp", "NB", "WB"]

# ====================== MINIMAL FEATURE SET ======================

SELECTED_FEATURES = [
    "nb_peak_salience",
    "spec_peakiness_ratio",
    "spec_flatness",
    "spec_centroid_Hz",
    "spec_spread_Hz",
    "stft_centroid_std_Hz",
    "stft_centroid_absderiv_med_Hzps",
    "chirp_slope_Hzps",
    "chirp_r2",
    "env_mod_index",
]

# ====================== FEATURE EXTRACTOR ======================

from feature_extractor import extract_features, FEATURE_NAMES

PRE_RMS_IDX = FEATURE_NAMES.index("pre_rms")
_NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_SELECTED_IDXS = [_NAME_TO_IDX[n] for n in SELECTED_FEATURES]

# ====================== HELPERS ======================

def timing_stats(values):
    if not values:
        return {}
    a = np.asarray(values) * 1e3
    return {
        "count": int(a.size),
        "mean_ms": float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "std_ms": float(np.std(a)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
    }

def normalize_label(lbl: Optional[str]) -> Optional[str]:
    if lbl is None:
        return None
    lbl = lbl.lower()
    if lbl in ("nojam", "clean"):
        return "NoJam"
    if lbl == "chirp":
        return "Chirp"
    if lbl == "nb":
        return "NB"
    if lbl == "wb":
        return "WB"
    return None

def normalize_model_labeler(model):
    def predict_fn(feat_vec):
        yhat = model.predict([feat_vec])[0]
        name = MODEL_CLASS_NAMES[int(yhat)]
        probs = model.predict_proba([feat_vec])[0]
        proba = {MODEL_CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
        return name, proba
    return predict_fn

def confusion_matrix_with_percentages(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = 100.0 * cm / np.maximum(row_sums, 1)
    return cm_pct, row_sums.ravel()

# ====================== MAIN ======================

def main():

    labels_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(LABELS_CSV)
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib_load(MODEL_PATH)
    predict_fn = normalize_model_labeler(model)

    y_true, y_pred = [], []
    rows_log = []

    t_npz, t_feat, t_inf = [], [], []

    with open(labels_csv, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            gt_label = normalize_label(row.get("label"))
            if gt_label is None:
                continue

            iq_path = Path(row.get("iq_path", ""))
            if not iq_path.exists():
                continue

            t0 = time.perf_counter()
            data = np.load(iq_path)
            t_npz.append(time.perf_counter() - t0)

            iq = data["iq"]
            fs = float(data["fs_hz"])

            t0 = time.perf_counter()
            feats_full = extract_features(iq, fs)
            t_feat.append(time.perf_counter() - t0)

            feats = feats_full[_SELECTED_IDXS]

            t0 = time.perf_counter()
            pred_label, pred_proba = predict_fn(feats)
            t_inf.append(time.perf_counter() - t0)

            y_true.append(gt_label)
            y_pred.append(pred_label)

            if DEBUG_PRINT_SAMPLE_LABELS:
                print(f"GT={gt_label} | Pred={pred_label}")

            row_log = {
                "iq_path": str(iq_path),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "pre_rms": float(feats_full[PRE_RMS_IDX]),
            }
            for k, v in pred_proba.items():
                row_log[f"p_{k}"] = v

            rows_log.append(row_log)

    # ====================== METRICS ======================

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=MODEL_CLASS_NAMES)
    cm_pct, row_totals = confusion_matrix_with_percentages(cm)

    print("\n=== CONFUSION MATRIX (COUNTS) ===")
    for i, lbl in enumerate(MODEL_CLASS_NAMES):
        print(f"{lbl:>7s} | {cm[i]} | total={row_totals[i]}")

    print("\n=== CONFUSION MATRIX (ROW-NORMALIZED %) ===")
    for i, lbl in enumerate(MODEL_CLASS_NAMES):
        pct_row = " ".join(f"{v:6.1f}%" for v in cm_pct[i])
        print(f"{lbl:>7s} | {pct_row}")

    print("\n=== PER-CLASS RECALL ===")
    for i, lbl in enumerate(MODEL_CLASS_NAMES):
        if row_totals[i] > 0:
            recall = cm[i, i] / row_totals[i]
            print(f"{lbl:>7s}: recall = {recall:.4f} ({cm[i,i]}/{row_totals[i]})")

    print(f"\nOverall accuracy: {acc:.6f}")

    # ====================== SAVE MATRICES ======================

    np.savetxt(out_dir / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "confusion_matrix_rowpct.csv", cm_pct, fmt="%.2f", delimiter=",")

    if SAVE_IMAGES:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_pct, cmap=plt.cm.Blues, vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="Row-normalized %")

        ax.set_xticks(range(len(MODEL_CLASS_NAMES)))
        ax.set_yticks(range(len(MODEL_CLASS_NAMES)))
        ax.set_xticklabels(MODEL_CLASS_NAMES, rotation=45)
        ax.set_yticklabels(MODEL_CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (counts + row %)")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9
                )

        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix_detailed.png", dpi=150)
        plt.close(fig)

    # ====================== SAVE CSV / SUMMARY ======================

    if SAVE_PER_SAMPLE_CSV and rows_log:
        with open(out_dir / "samples_eval.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows_log[0].keys())
            w.writeheader()
            w.writerows(rows_log)

    if SUMMARY_JSON:
        summary = {
            "model_path": MODEL_PATH,
            "labels_csv": str(labels_csv),
            "feature_count": len(SELECTED_FEATURES),
            "feature_names": SELECTED_FEATURES,
            "accuracy": acc,
            "timing": {
                "npz_load": timing_stats(t_npz),
                "feature_extraction": timing_stats(t_feat),
                "inference": timing_stats(t_inf),
            },
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print("\nDONE")

if __name__ == "__main__":
    main()
