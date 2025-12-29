# Usage

This document describes **how to use this repository end-to-end**:

- Train models (XGBoost and Deep Learning)
- Retrain / fine-tune models on real labelled data
- Validate models on labelled datasets
- **Test models on unlabelled data** (two testing modalities)

This file intentionally **does not** cover environment setup or dependency installation.  
See `README.md` for installation.

---

## 0. A note on “CLI arguments” (and how to avoid them)

Some scripts in this repo expose **CLI arguments** via `argparse`. Two important points:

1) **Arguments are not mandatory**  
   You can run those scripts with **no arguments** and they will use the defaults defined in `parse_args()`.

2) **If you dislike CLI usage (recommended for your workflow)**  
   Open the script and **hardcode your values** by editing the defaults inside `parse_args()` (or by editing the top-level `CONFIG` / “USER VARIABLES” section for non-CLI scripts).  
   This is the intended “plug-and-play” style for most of the repository.

In practice, this means you have two equivalent ways of working:

- CLI style:
  `python train/data_preparation_xgb.py --base "..."`
- Hardcoded style:
  open `train/data_preparation_xgb.py` and change the default of `--base` inside `parse_args()`.

---

## 1. Expected dataset format (MATLAB generator)

The training and preparation scripts are designed around the dataset format produced by the MATLAB generator:

- Generator repository: `https://github.com/macaburguera/GNSS_generator`
- Dataset builder entry point: `signal_generation_jammertest.m`

That generator produces stratified datasets with the following structure:

```
BASE/
  TRAIN/<Class>/*.mat
  VAL/<Class>/*.mat
  TEST/<Class>/*.mat
```

By default, each `.mat` tile includes:
- `GNSS_plus_Jammer_awgn` : complex baseband IQ (column vector)
- `meta` : metadata struct (band, jammer class, C/N0, JSR, etc.)

If you use a different generator or different MAT variable names, you must adapt:
- XGB prep: `--var` (or edit its default)
- DL training: `CONFIG["VAR"]`

---

## 2. Entry Points (Where to Run Things)

The repository is organized around four operational folders:

- `train/` – training on controlled datasets (synthetic / curated `.mat`)
- `retrain/` – fine-tuning on **real labelled** samples exported by the labelling workflow (`*_labels.csv`)
- `validation/` – controlled evaluation on labelled data (metrics + confusion matrices)
- `test/` – **test-time scanning on unseen/unlabelled recordings**, plus plot generation

Most scripts fall into one of two styles:

1. **CLI scripts (argparse)**  
   You *can* run them with arguments, but you can also run them with no arguments (defaults), or hardcode defaults in the file.

2. **Config-at-top scripts**  
   You edit a `CONFIG = dict(...)`, a `class Config: ...`, or a `USER VARIABLES` section at the top, then run:
   `python <script>.py`

The test pipeline scripts are primarily **config-at-top** scripts.

---

## 3. Training

### 3.1 XGBoost training (feature-based)

XGBoost training in this repo is a two-step process:

1) **Prepare cached feature datasets (`.npz`)**  
2) **Train + evaluate XGBoost on those cached features**

Both steps assume the dataset layout produced by `GNSS_generator` (see Section 1).

#### Step 1 — Create feature caches

Script:
- `train/data_preparation_xgb.py`

You can run it in two ways:

- **Hardcoded style (no CLI):** edit defaults in `parse_args()` and run:
  ```bash
  python train/data_preparation_xgb.py
  ```

- **CLI style:** pass parameters explicitly:
  ```bash
  python train/data_preparation_xgb.py --base "<BASE>"
  ```

Key arguments (optional):
- `--base` : dataset root containing `TRAIN / VAL / TEST`
- `--var` : MAT variable name (default matches `GNSS_generator`: `GNSS_plus_Jammer_awgn`)
- `--fs` : sampling rate in Hz
- `--out` : output artifacts root
- `--classes` : class list (folder names)

Output:
- A new folder is created under `--out` named `prep_run_YYYYmmdd_HHMMSS/`
- It contains: `train_features.npz`, `val_features.npz`, `test_features.npz`

#### Step 2 — Train & evaluate XGBoost

Script:
- `train/train_eval_xgb.py`

Run (again: CLI optional; defaults can be hardcoded inside `parse_args()`):

```bash
python train/train_eval_xgb.py
```

Supported arguments (optional):
- `--features_dir` : folder containing `train_features.npz / val_features.npz / test_features.npz`
- `--out` : artifacts root
- `--run_name` : optional suffix

Output:
- A new folder under `--out` named `xgb_run_YYYYmmdd_HHMMSS/`
- Contains:
  - trained model (joblib)
  - confusion matrices
  - per-class metrics
  - additional analysis outputs saved by the script

---

### 3.2 Deep Learning training (spectrogram CNN)

Script:
- `train/train_eval_cnn_spectrogram.py`

This is a **config-at-top** script (explicitly no CLI).  
Open the file and edit the `CONFIG = dict(...)` block.

The dataset structure and MAT variable are expected to match `GNSS_generator` defaults (Section 1).

Run:

```bash
python train/train_eval_cnn_spectrogram.py
```

Output:
- A run directory under the configured `OUT_ROOT`
- Includes the trained `model.pt` and training/validation artifacts created by the script

---

### 3.3 Deep Learning training (raw-IQ 1D CNN)

Script:
- `train/train_eval_cnn_rawiq.py`

This is a **CLI script**, but the same rule applies:
- arguments are optional (defaults exist)
- if you prefer, hardcode defaults inside `parse_args()`

Run:

```bash
python train/train_eval_cnn_rawiq.py
```

(or pass args if you want)

---

## 4. Retraining / Fine-Tuning on Real Labelled Data

Retraining scripts use **label CSVs** produced by the external labelling workflow (the `*_labels.csv` files).  
These CSVs point to saved samples (typically `.npz`) and include class labels.

### 4.1 XGBoost fine-tuning (continue boosting)

Script:
- `retrain/retrain_xgb.py`

This is a **config-at-top** script. Edit the config section:

- `LABELS_CSVS` : list of one or more `*_labels.csv`
- `MODEL_IN` : existing trained model to start from
- `OUT_ROOT` : output folder
- `RUN_NAME` : optional name

Run:

```bash
python retrain/retrain_xgb.py
```

Output:
- A new run folder under `OUT_ROOT` with:
  - fine-tuned XGB model
  - metrics and confusion matrices produced by the script

---

### 4.2 XGBoost retrain from scratch using only 10 features (ablation)

Script:
- `retrain/retrain_xgb_minimal.py`

Also **config-at-top**. Edit:
- `LABELS_CSVS`
- `OUT_ROOT`
- any model/training parameters defined in the script

Run:

```bash
python retrain/retrain_xgb_minimal.py
```

Purpose:
- controlled comparison vs full feature set
- designed for ablation and timing comparisons

---

### 4.3 DL spectrogram fine-tuning on real labelled data

Script:
- `retrain/retrain_dl.py`

This is **config-at-top** with:

- `LABELS_CSVS` : list of `*_labels.csv`
- `MODEL_IN` : starting checkpoint (`.pt`)
- `OUT_ROOT` : output root
- split controls and fine-tuning hyperparameters (defined in the script)

Run:

```bash
python retrain/retrain_dl.py
```

Output:
- a run folder under `OUT_ROOT` containing the fine-tuned model and evaluation artifacts created by the script

---

## 5. Validation (Controlled, Labelled Evaluation)

Validation scripts evaluate a chosen model on a **labelled dataset** (via `*_labels.csv`) and produce:
- metrics
- confusion matrices
- (optionally) per-sample logs / plots
- (in timed variants) computational profiling

All validation scripts live under `validation/` and are **config-at-top** scripts.

### 5.1 XGBoost validation on labelled dataset

Script:
- `validation/validation_xgb_labelled.py`

Edit the **USER VARIABLES** section:
- `LABELS_CSV`
- `OUT_DIR`
- `MODEL_PATH`
- `SAVE_IMAGES` and other save options

Run:

```bash
python validation/validation_xgb_labelled.py
```

### 5.2 XGBoost validation with timing

Script:
- `validation/validation_xgb_labelled_timed.py`

Run:

```bash
python validation/validation_xgb_labelled_timed.py
```

### 5.3 XGBoost validation using 10 features

Script:
- `validation/validation_xgb_10features.py`

Run:

```bash
python validation/validation_xgb_10features.py
```

### 5.4 DL validation on labelled dataset

Script:
- `validation/validation_dl_labelled.py`

Run:

```bash
python validation/validation_dl_labelled.py
```

### 5.5 DL validation with timing

Script:
- `validation/validation_dl_labelled_timed.py`

Run:

```bash
python validation/validation_dl_labelled_timed.py
```

---

## 6. Testing (Unlabelled / In-the-Wild)

This repository supports **two testing modalities**, both under `test/`:

1) **Folder scanning** (scan folders referenced by `incidents.txt`, save predictions → plot later)
2) **Single SBF scanning** (scan one file end-to-end, optionally generating plots inline)

These are intended for **unlabelled, in-the-wild** recordings, where no accuracy metrics are computed.

---

### 6.1 Testing modality A — Folder scanning + plotting from saved predictions

This is the “scan first, plot later” workflow.

#### A1) Folder scanning (prediction only, no plots)

Script:
- `test/predict_incidents_folder.py`

This script:
- reads an `incidents.txt`
- scans only the folders referenced there
- runs blockwise DL inference on SBF blocks
- writes prediction CSV(s) and merged detections to a timestamped run directory

**Configuration**

Edit the `class Config:` section at the top of the file. Key fields:

- `ROOT_DIR` : root containing your test folders
- `OUT_ROOT` : where run outputs will be written
- `MODEL_PT` : model checkpoint used for inference
- `INCIDENTS_TXT` : path to `incidents.txt`

Context-window behavior is controlled by:
- `INCIDENT_CONTEXT_TOTAL_MINUTES`
- `PROCESS_ONLY_CONTEXT`
- `REQUIRE_TIME_TAGS_FOR_CONTEXT_FILTER`

**Run**

```bash
python test/predict_incidents_folder.py
```

**Outputs**

A timestamped folder is created under `OUT_ROOT`, containing prediction CSVs and merged detections (filenames are set in the script).

---

#### A2) Plot generation from predictions (offline plotting)

Script:
- `test/plot_from_predictions.py`

This script:
- loads the predictions CSV produced by the folder scan
- re-opens the corresponding SBF files
- locates blocks by `block_index` (and other fields saved in the CSV)
- generates spectrogram plots for the selected detections

**Configuration**

Edit `class Config:` in the script. Key fields:

- `ROOT_DIR` : where the SBF folders live (same root as in the scan)
- `RUN_DIR` : the specific run folder created by `predict_incidents_folder.py`
- `PRED_CSV_NAME` : which predictions CSV to use
- filtering controls (class include/exclude, thresholds, max plots)

**Run**

```bash
python test/plot_from_predictions.py
```

**Outputs**

A plot folder is created inside `RUN_DIR` (name controlled in the script), containing PNG plots for selected detections.

---

### 6.2 Testing modality B — Single SBF scanning (full-file, plug-and-play)

Script:
- `test/predict_incidents_sbf.py`

This script scans **one SBF file**, runs inference on **every** `BBSamples` block, and can **optionally** generate spectrogram plots.

#### B1) Configure the scan

Edit `class Config:` at the top of the script:

- `SBF_FILE` : path to the input `.sbf`
- `OUT_DIR` : output folder for CSV and plots
- `MODEL_PT` : model checkpoint for inference
- `DEVICE` : `"auto"` or explicit device
- plot toggles and plot output naming options (defined in the script)

#### B2) Run

```bash
python test/predict_incidents_sbf.py
```

#### B3) Outputs

Inside `OUT_DIR`:
- a predictions CSV (one row per `BBSamples` block)
- if plotting is enabled, a plots folder with PNG spectrograms (filename pattern defined in the script)

---

## 7. Recommended workflow summary

- **Train (synthetic / controlled `.mat`)**
  - XGB: `train/data_preparation_xgb.py` → `train/train_eval_xgb.py`
  - DL: `train/train_eval_cnn_spectrogram.py` (edit `CONFIG`) or `train/train_eval_cnn_rawiq.py`

- **Retrain (real labelled `*_labels.csv`)**
  - XGB: `retrain/retrain_xgb.py` or `retrain/retrain_xgb_minimal.py`
  - DL: `retrain/retrain_dl.py`

- **Validate (labelled `*_labels.csv`)**
  - XGB: `validation/validation_xgb_labelled.py` (or `_timed`)
  - DL: `validation/validation_dl_labelled.py` (or `_timed`)

- **Test (unlabelled)**
  - Folder-based: `test/predict_incidents_folder.py` → `test/plot_from_predictions.py`
  - Single-file: `test/predict_incidents_sbf.py`

For experiment-specific reporting and results interpretation, see the documents under `docs/` (especially `6_Validation_extended.md` and `7_Test_Rodby.md`).
