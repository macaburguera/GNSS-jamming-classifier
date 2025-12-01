# GNSS Jamming Classifier – Pipeline Overview

This document explains how the repository fits together, from **synthetic training**
to **real-world validation** on Jammertest 2023.

The working label set is:

- `NoJam`
- `Chirp`
- `NB`  (narrowband jamming)
- `WB`  (wideband jamming)

These four class names are used consistently across:

- synthetic dataset folder names,
- training scripts in `sim_train/`,
- validation scripts in `jammertest_day1/` and `jammertest_day2/`,
- plotting and metrics code.

---

## 1. Directory overview

### 1.1 Training: `sim_train/`

- `data_preparation_xgb.py`  
  Extracts **78 hand-crafted features** from Jammertest-style `.mat` tiles and
  saves them into NPZ caches (train / val / test).

- `train_eval_xgb.py`  
  Loads NPZ caches, trains an XGBoost classifier (with simple grid search),
  evaluates on the test split, and saves:
  - the trained model (`xgb_trainval.joblib`),
  - classification reports,
  - confusion matrices.

- `train_eval_cnn_rawiq.py`  
  Trains a **1-D CNN** directly on raw IQ tiles:
  - builds PyTorch `Dataset` objects for TRAIN / VAL / TEST,
  - crops/pads IQ to a fixed length (e.g. 2048),
  - applies normalization and optional data augmentation,
  - evaluates and saves `model.pt` plus plots and confusion matrices.

### 1.2 Real Jammertest validation

- `jammertest_day1/` (Day 1, Altus SBF):

  - `label_jammertest.py`  
    Per-block feature extraction + XGB prediction + metrics + spectrogram plots
    every fixed number of seconds.

  - `validation_xgb_day1.py`  
    Main validation script for XGB:
    - enforces a **canonical label namespace** (4 classes),
    - fixes the confusion matrix class order,
    - includes optional “NoJam veto” logic for low-confidence predictions.

  - `validation_cnn_day1.py`  
    Equivalent validation, but using the 1-D CNN model instead of XGB.

- `jammertest_day2/` (Day 2, Bleik / Ramnan, SBF + SDR):

  - `sbf/sbf_iq_bb_alt02-ref.py`  
    SBF → STFT plots for a specific Altus configuration (multi-band).

  - `sbf/sbf_iq_generic_plotter_all.py`  
    Generic SBF spectrogram explorer for Day 2.

  - `sbf/sbf_xgb_bleik_day2.py`  
    Samples IQ every *X* seconds for each LO/band, computes the
    78-dim features, runs the XGB model, saves spectrograms and metrics.

  - `sdr/sdr_iq_open.py`  
    Basic reader for Kraken SDR raw IQ files (float32, interleaved I/Q).

  - `sdr/bleik_d2.py`, `sdr/grunvatn_d2.py`  
    Site-specific exploration scripts with plots and checks.

  - `sdr/sdr_xgb_bleik_d2.py`  
    Main SDR-based validation for the Bleik ramp test:
    - reads Kraken IQ,
    - takes periodic IQ “snaps”,
    - computes the same 78-dim feature vector,
    - runs the XGB model,
    - infers ramp phase (up, down, outside),
    - looks up logbook GT and maps to canonical class,
    - saves spectrograms annotated with:
      - testplan info,
      - ramp phase,
      - GT class,
      - predicted class + probability,
      - STFT and decimation settings.
    - aggregates accuracy and confusion matrices across the ramp.

### 1.3 Exploratory scripts: `explore/`

- `explore/sbf/`:

  - `sbf_iq.py`  
    Standalone SBF → spectrogram pipeline:
    - samples one block every fixed period,
    - parses logbook labels,
    - overlays them on the spectrogram axis.

  - `sbf_iq_export.py`  
    Helper to export IQ from SBF to other formats.

  - `sbf_session_stats.py`  
    Computes simple session-level statistics from SBF files.

- `explore/sim/`:

  - `jammertest_sim_plots.py`  
    Quick-and-dirty plotting for synthetic `.mat` datasets:
    envelope, PSD, spectrogram, etc.

### 1.4 Legacy experiments: `zenodo_old/`

All scripts here target an older **public Zenodo GNSS+jamming dataset** and are
useful mainly as reference:

- `data_preparation.py`  
  Feature extraction similar in spirit to `sim_train/data_preparation_xgb.py`,
  but tailored to the Zenodo format.

- `prepare_minirocket.py`  
  Uses `sktime`’s **MiniROCKET** to produce large sets of convolutional features
  from IQ time series.

- `train_encoder_supcon.py`  
  Trains a SimCLR-like **SupCon encoder** on IQ.

- `select_and_fuse_features.py`  
  Feature selection / fusion between hand-crafted and MiniROCKET features.

- `train_eval_logreg.py`, `train_eval_svm.py`, `train_eval_xgb.py`,
  `train_svm_min.py`, `train_xgb_cascade.py`, `train_xgb_min.py`,
  `zenodo_basic.py`, `zenodo_plots.py`  
  A zoo of classic machine-learning experiments (logreg, SVM, XGB) plus plots.

---

## 2. Synthetic training pipeline (`sim_train/`)

### 2.1 Dataset structure

The synthetic dataset is Jammertest-style and structured as:

```text
BASE/
  TRAIN/<Class>/*.mat
  VAL/<Class>/*.mat
  TEST/<Class>/*.mat
```

with `<Class> ∈ {NoJam, Chirp, NB, WB}`.

Each `.mat` file is expected to provide:

- `GNSS_plus_Jammer_awgn` : complex IQ vector,
- optionally `meta` with fields such as:
  - `JSR_dB`,
  - `CNR_dBHz` or `CNo_dBHz`,
  - `band`, `fs_Hz`,
  - jammer code/name and parameters.

### 2.2 Feature space

The **78 hand-crafted features** are defined in `docs/features.md` and implemented
in `data_preparation_xgb.py` via `extract_features(iq, fs)`.

Broad categories (see the feature document for formulas):

- power and RMS measures,
- spectral flatness / peaky-ness and roll-off,
- in-band vs out-of-band energy ratios,
- chirp / frequency-slope indicators,
- amplitude modulation / pulsed behaviour,
- GNSS-like chip-period cyclostationarity,
- non-Gaussianity and non-circularity measures.

The same function is reused for SBF/SDR validation, ensuring that **SIM training
and real-data validation live in the same feature space**.

### 2.3 XGBoost training (`train_eval_xgb.py`)

1. **Feature extraction**

   ```bash
   cd sim_train
   python data_preparation_xgb.py        --base /path/to/datasets_jammertest        --out ../artifacts/jammertest_sim        --var GNSS_plus_Jammer_awgn        --fs 60000000        --classes "NoJam,Chirp,NB,WB"
   ```

   This scans TRAIN / VAL / TEST, computes 78-dim features for each tile,
   and saves NPZ caches under `../artifacts/jammertest_sim/prep_run_*`.

2. **Model training**

   ```bash
   python train_eval_xgb.py        --prep_root ../artifacts/jammertest_sim        --out ../artifacts/jammertest_sim
   ```

   The script:

   - loads `train_features.npz`, `val_features.npz`, `test_features.npz`,
   - builds an `sklearn` `Pipeline` with standardization + XGBoost,
   - performs a small grid search over XGB hyperparameters,
   - trains on TRAIN+VAL, evaluates on TEST,
   - saves:
     - `xgb_run_*/xgb_trainval.joblib`,
     - confusion matrices (.csv + .png),
     - CSV metrics (including per-JSR/CNR bins, if available).

The resulting `.joblib` file is the **canonical XGB model** for Jammertest
validation (Day 1 and Day 2).

### 2.4 1-D CNN training (`train_eval_cnn_rawiq.py`)

This script implements a lightweight 1-D CNN for raw IQ:

- Input: complex IQ sequence, cropped/padded to `target_len` samples.
- Preprocessing: mean removal, scaling to unit RMS.
- Architecture: several 1-D convolutional blocks + pooling + dense head.
- Loss: cross-entropy over the 4 classes.
- Optional augmentations (phase jitter, mild CFO, light AWGN).

Run, for example:

```bash
python train_eval_cnn_rawiq.py     --base /path/to/datasets_jammertest     --var GNSS_plus_Jammer_awgn     --fs 60000000     --target_len 2048     --classes "NoJam,Chirp,NB,WB"     --out ../artifacts/jammertest_sim     --batch_size 256     --epochs 100     --device cuda
```

Outputs:

- `run_*/model.pt` (PyTorch state dict),
- training curves,
- validation / test reports and confusion matrices.

---

## 3. Real-world validation on Jammertest 2023

### 3.1 Canonical label mapping

The Jammertest logbooks use codes such as:

- `"NO JAM"`, `"no jamming"`,
- `"h1.1"`, `"h1.2"`, `"u1.1"`, `"u1.2"`,
- `"s1.2"`, `"h3.1"`, `"s2.1"`, …

Validation scripts implement a mapping to the canonical 4-class namespace:

- `"NO JAM"`, `"no jamming"` → `NoJam`
- codes corresponding to narrowband jamming (e.g. `"h1.1"`) → `NB`
- chirp / swept / more complex high-power codes  
  (`"h1.2"`, `"u1.1"`, `"u1.2"`, `"s1.2"`, `"h3.1"`, `"s2.1"`, …) → `Chirp`
- codes for wideband high-power jammers → `WB` (when present)

Unknown or ambiguous labels can be flagged and excluded from GT statistics.

This mapping is centralised so that:

- synthetic training labels,
- Jammertest GT labels,
- model outputs,

all share the same 4-class semantic space.

### 3.2 Day 1: Altus SBF (`jammertest_day1/`)

**Key scripts:**

- `validation_xgb_day1.py`:

  - Reads SBF `BBSamples` via `SbfParser` in chunks.
  - Uses a fixed sampling cadence (e.g. one block every 10 or 30 seconds UTC).
  - For each selected block:
    - extracts the 78-dim feature vector,
    - runs the XGB model,
    - looks up the logbook label at that time,
    - maps it to the canonical class.
  - Accumulates:
    - per-class counts of GT vs predictions,
    - confusion matrices with fixed class order,
    - optional “NoJam veto” overrides when model confidence is low.
  - Produces:
    - spectrograms annotated with GT + prediction + metadata,
    - CSV with per-sample predictions and GT,
    - summary JSON/CSV metrics.

- `validation_cnn_day1.py`:

  - Same overall flow, but feeding **raw IQ tiles** to the CNN model instead of features to XGB.
  - Ensures the same label mapping and metric computation.

- `label_jammertest.py`:

  - A slightly more exploratory script that:
    - grabs one IQ block every fixed number of seconds,
    - extracts features,
    - runs the XGB model,
    - computes accuracy / confusion matrix on blocks where GT is well defined,
    - saves labelled spectrograms.

All three scripts follow the pattern: **edit user config at the top, then run**.

### 3.3 Day 2: Bleik / Ramnan, SBF + SDR (`jammertest_day2/`)

Two data paths:

1. **SBF path** (`jammertest_day2/sbf/`):

   - `sbf_xgb_bleik_day2.py`:
     - reads SBF `BBSamples` chunks,
     - keeps track of LO/band identifiers,
     - every `SAMPLE_PERIOD_SEC` seconds (UTC) per LO/band:
       - extracts IQ,
       - decimates (optional),
       - computes the 78-dim feature vector,
       - runs XGB,
       - saves spectrograms with GT label, predicted class and probability, LO/band, STFT parameters.

   - `sbf_iq_bb_alt02-ref.py` and `sbf_iq_generic_plotter_all.py`:
     pure exploration / plotting for Day 2 SBF sessions.

2. **SDR path** (`jammertest_day2/sdr/`):

   - `sdr_iq_open.py`:
     - low-level reader for Kraken raw IQ (float32, interleaved I/Q).

   - `sdr_xgb_bleik_d2.py`:
     - opens a specific Bleik ramp-test Kraken file,
     - every `SNAP_PERIOD_SEC` seconds:
       - takes a fixed-length IQ “snap”,
       - computes the 78-dim feature vector,
       - runs the XGB model,
       - infers ramp phase (up, down, outside),
       - looks up logbook GT and maps to canonical class,
       - saves spectrograms annotated with:
         - testplan info,
         - ramp phase,
         - GT class,
         - predicted class + probability,
         - STFT and decimation settings.
     - aggregates accuracy and confusion matrices across the ramp.

   - `bleik_d2.py`, `grunvatn_d2.py`:
     - more ad-hoc exploration / plotting for the Day 2 sites.

---

## 4. Legacy Zenodo experiments (`zenodo_old/`)

This folder documents an earlier exploration phase based on a public Zenodo dataset:

- Feature extraction similar to the Jammertest pipeline.
- MiniROCKET features via `sktime`.
- SupCon encoder on IQ.
- A battery of ML baselines (logreg, SVM, XGB in various flavours).

These scripts are **not required** for the main Jammertest pipeline, but they
capture design decisions and older experiments.

---

## 5. Where to look next

- If you want to change or extend the **feature set**, start from:
  - `docs/features.md`
  - `sim_train/data_preparation_xgb.py`
  - any script calling `extract_features(iq, fs)`

- If you want to adjust the **label taxonomy** or GT mapping, look into:
  - `jammertest_day1/validation_xgb_day1.py`
  - `jammertest_day2/sdr/sdr_xgb_bleik_d2.py`

- If you want to plug in **new models** (e.g., different CNNs or transformers),
  use:
  - `train_eval_cnn_rawiq.py` as a template for training,
  - `validation_cnn_day1.py` / `sdr_xgb_bleik_d2.py` as templates for inference.
