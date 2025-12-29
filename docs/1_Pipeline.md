# 1. Pipeline (Repo Overview)

This repository provides a full end-to-end pipeline for **tile-based GNSS interference/jamming classification** from raw baseband (IQ) recordings.

The workflow is:

1. **Data preparation**
2. **Model training / retraining**
3. **Evaluation and test-time scanning**

---

## 1.1 Data preparation

Two data sources are used:

- **Synthetic**: MATLAB-generated `.mat` tiles (see `docs/2_Data.md`).
- **Real (labelled)**: Jammertest 2023 baseband recordings processed and labelled with `sbf-labeller`.

For the feature-based pipeline, synthetic/real `.mat` tiles are converted into feature datasets:
- Script: `train/data_preparation_xgb.py`
- Output: `.npz` feature arrays stored under `artifacts/.../features/`

For the DL pipeline, tiles are converted on-the-fly into spectrogram tensors inside the training code.

---

## 1.2 Training

### XGBoost
- Baseline training: `train/train_eval_xgb.py`
- Retraining (domain adaptation): `retrain/retrain_xgb.py`
- Minimal 10-feature ablation: `retrain/retrain_xgb_minimal.py`

### Deep learning (Spectrogram SE-CNN)
- Baseline training: `train/train_eval_cnn_spectrogram.py`
- Retraining (domain adaptation): `retrain/retrain_dl.py`

---

## 1.3 Validation

Validation is performed on a labelled real dataset (Alt06001). Outputs live under `results/`.

Each validation run folder contains:
- `samples_eval.csv`: per-tile predictions
- `summary.json`: aggregate metrics and settings
- confusion matrices (`confusion_matrix.png`, row-normalized variants)

DL validation folders also include:
- `timing_summary.json`: breakdown of inference costs

For a metrics-driven comparison, see `docs/6_Validation_extended.md`.

---

## 1.4 Test-time scanning (unlabelled)

The `test/` folder contains scanning scripts to run inference over unseen SBF recordings and optionally generate plots.

This stage is designed for:
- operational sanity checking,
- long-run behaviour analysis,
- event clustering on roadtest data.

---

## 1.5 Reproducibility notes

- All key configs are either:
  - encoded in scripts under `train/` and `retrain/`, or
  - stored in artifacts (`run_meta.json`, `meta.json`, `best_params.txt`).
- Every evaluation run stores per-sample predictions and confusion matrices under `results/`.
