# GNSS Jamming Classifier

This repository implements an end-to-end pipeline to **detect and classify GNSS jamming** from complex baseband IQ data.

It has three main pillars:

1. **Synthetic training** on Jammertest-style `.mat` datasets.
2. **Real-world validation** on Jammertest 2023 (Day 1 & Day 2, SBF + SDR).
3. A set of **hand-crafted features** (78 dimensions) specifically designed for GNSS+jamming IQ  
   (documented in `docs/features.md`).

Two model families are currently supported:

- **XGBoost** on the 78-dim feature vector.
- **1-D CNN** on raw IQ tiles.

Canonical classes used across the code:

- `NoJam`
- `Chirp`
- `NB` (narrowband jamming)
- `WB` (wideband jamming)

These class names are shared by:
- the synthetic dataset folder structure,
- the training scripts,
- the Jammertest validation scripts.

---

## Repository structure

Top-level layout:

```text
.
├── sim_train/          # Training on synthetic Jammertest-style .mat datasets
│   ├── data_preparation_xgb.py   # Extract 78-dim features → NPZ caches
│   ├── train_eval_xgb.py         # Train & eval XGBoost on cached features
│   └── train_eval_cnn_rawiq.py   # Train & eval 1-D CNN on raw IQ tiles
│
├── jammertest_day1/    # Validation on Jammertest 2023 Day 1 (Altus SBF)
│   ├── label_jammertest.py       # Per-block XGB prediction + metrics + plots
│   ├── validation_xgb_day1.py    # Main XGB validation pipeline (canonical labels)
│   └── validation_cnn_day1.py    # CNN validation pipeline on SBF IQ
│
├── jammertest_day2/    # Validation on Jammertest 2023 Day 2 (Bleik, Ramnan,…)
│   ├── sbf/
│   │   ├── sbf_iq_bb_alt02-ref.py          # STFT/plots for Altus SBF (multi-band)
│   │   ├── sbf_iq_generic_plotter_all.py   # Generic SBF spectrogram explorer
│   │   └── sbf_xgb_bleik_day2.py           # XGB inference on Day 2 SBF (per band)
│   └── sdr/
│       ├── sdr_iq_open.py          # Kraken SDR raw IQ reader utilities
│       ├── bleik_d2.py             # Bleik-specific SDR exploration / plotting
│       ├── grunvatn_d2.py          # Grunvatn-specific SDR exploration / plotting
│       └── sdr_xgb_bleik_d2.py     # XGB inference on Kraken SDR (ramp test)
│
├── explore/              # Small exploratory scripts
│   ├── sbf/
│   │   ├── sbf_iq.py              # Standalone SBF → spectrogram + labels
│   │   ├── sbf_iq_export.py       # Export SBF IQ to simpler formats
│   │   └── sbf_session_stats.py   # IQ/session statistics from SBF
│   └── sim/
│       └── jammertest_sim_plots.py   # Quick plots for synthetic .mat datasets
│
├── docs/
│   ├── features.md       # Detailed description of the 78 hand-crafted features
│   └── pipeline.md       # High-level overview of the full pipeline
│
├── zenodo_old/           # Legacy experiments on a public Zenodo dataset
│   ├── data_preparation.py
│   ├── prepare_minirocket.py      # MiniROCKET feature extraction (sktime)
│   ├── train_encoder_supcon.py    # SupCon encoder on IQ
│   ├── select_and_fuse_features.py
│   ├── train_eval_logreg.py
│   ├── train_eval_svm.py
│   ├── train_eval_xgb.py
│   ├── train_svm_min.py
│   ├── train_xgb_cascade.py
│   ├── train_xgb_min.py
│   ├── zenodo_basic.py
│   └── zenodo_plots.py
│
├── requirements.txt      # Python dependencies (pip/conda)
└── .git/                 # Git metadata (not part of the pipeline)
```

The **current mainline pipeline** for your Jammertest work is:

- Training: `sim_train/`
- Validation on real data: `jammertest_day1/` and `jammertest_day2/`
- Feature documentation: `docs/features.md`
- Conceptual overview: `docs/pipeline.md`

Everything in `zenodo_old/` can be treated as **archived experiments**.

---

## Installation

### 1. Create and activate a conda environment

```bash
# Create environment (Python version is flexible; 3.9–3.11 are fine)
conda create -n gnss-jamming-classifier python=3.10
conda activate gnss-jamming-classifier
```

### 2. Install Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

This installs:

- core scientific stack: `numpy`, `scipy`, `pandas`, `matplotlib`
- ML stack: `scikit-learn`, `xgboost`, `torch`
- IO/utils: `h5py`, `joblib`
- optional: `sktime` (only needed for `zenodo_old/prepare_minirocket.py`)

> **Important — `sbf_parser`**
>
> Several scripts import `sbf_parser.SbfParser` to parse Septentrio SBF files.
> That module is **not** shipped here.  
> You must provide your own `sbf_parser` implementation (or adapt the imports)
> capable of reading `BBSamples` and returning IQ plus timestamps.

---

## Data expectations

### Synthetic Jammertest-style dataset (for training)

Training scripts in `sim_train/` expect:

```text
BASE/
  TRAIN/
    NoJam/*.mat
    Chirp/*.mat
    NB/*.mat
    WB/*.mat
  VAL/
    NoJam/*.mat
    Chirp/*.mat
    NB/*.mat
    WB/*.mat
  TEST/
    NoJam/*.mat
    Chirp/*.mat
    NB/*.mat
    WB/*.mat
```

Each `.mat` file should contain:

- `GNSS_plus_Jammer_awgn` : complex IQ vector (column),
- optionally a `meta` struct with fields like:
  - `JSR_dB`
  - `CNR_dBHz` or `CNo_dBHz`
  - `band`, `fs_Hz`, jammer code, etc.

Default sampling frequency in the scripts is around `60e6` Hz, and can be changed via CLI flags.

### Real Jammertest 2023 data (for validation)

Not included in the repo.  
You will need:

- **SBF** files from the Septentrio receiver (Day 1 & Day 2),
- Text **logbooks** describing jammer states vs time (codes like `"u1.1"`, `"h1.1"`, `"NO JAM"`, …),
- For Day 2 SDR scripts: raw Kraken IQ files (float32, interleaved I/Q).

Paths to these files are configured directly in the `USER VARIABLES` / `USER CONFIG` sections inside each script.

---

## Quick start

### 1. Train XGBoost on synthetic features

From the repo root:

```bash
cd sim_train

# 1) Extract features → NPZ caches
python data_preparation_xgb.py     --base /path/to/datasets_jammertest     --out ../artifacts/jammertest_sim     --var GNSS_plus_Jammer_awgn     --fs 60000000     --classes "NoJam,Chirp,NB,WB"

# 2) Train + evaluate XGB on those caches
python train_eval_xgb.py     --prep_root ../artifacts/jammertest_sim     --out ../artifacts/jammertest_sim
```

Outputs (under `../artifacts/...`):

- `prep_run_*/train_features.npz`, `val_features.npz`, `test_features.npz`
- `xgb_run_*/xgb_trainval.joblib` (trained model)
- CSV + PNG metrics and confusion matrices

That `.joblib` model path is referenced later by the Jammertest validation scripts.

---

### 2. Train a 1-D CNN on raw IQ

Still in `sim_train/`:

```bash
python train_eval_cnn_rawiq.py     --base /path/to/datasets_jammertest     --var GNSS_plus_Jammer_awgn     --fs 60000000     --target_len 2048     --classes "NoJam,Chirp,NB,WB"     --out ../artifacts/jammertest_sim     --batch_size 256     --epochs 100     --device cuda    # or "cpu"
```

Outputs (under `../artifacts/...`):

- `run_*/model.pt` (PyTorch checkpoint)
- training curves (.png)
- validation / test confusion matrices and reports

---

### 3. Validate on Jammertest Day 1 (Altus SBF)

XGBoost-based validation: `jammertest_day1/validation_xgb_day1.py`.

What it does:

- Parses SBF `BBSamples` IQ blocks via `SbfParser`.
- Samples one block every fixed number of seconds (UTC).
- Computes the **same 78 features** used during training.
- Loads your **SIM-trained XGB model**.
- Maps logbook codes (like `"NO JAM"`, `"u1.1"`, `"h1.2"`, etc.) into the canonical classes:
  - `NoJam`, `Chirp`, `NB`, `WB`.
- Computes metrics (including confusion matrices with fixed class order).
- Saves annotated spectrograms and per-sample CSVs.

How to run:

1. Open `jammertest_day1/validation_xgb_day1.py`.
2. Edit the `USER VARIABLES` section:
   - `SBF_PATH`
   - `OUT_DIR`
   - `LOGBOOK_PATH`
   - `MODEL_PATH`
   - `LOCAL_DATE`, `LOCAL_UTC_OFFSET`
   - sampling cadence (e.g. `SAVE_EVERY_SEC`).
3. Then:

   ```bash
   cd jammertest_day1
   python validation_xgb_day1.py
   ```

CNN-based validation: `jammertest_day1/validation_cnn_day1.py` mirrors the same logic,
but uses the 1-D CNN model (`model.pt`) instead of XGB.

For quick per-block experiments with visualization, you can use
`jammertest_day1/label_jammertest.py`.

---

### 4. Validate on Jammertest Day 2 (Bleik / Ramnan, SBF + SDR)

Under `jammertest_day2/`:

- **SBF path** (`jammertest_day2/sbf/`):
  - `sbf_xgb_bleik_day2.py`  
    Samples IQ from SBF every *X* seconds *per LO/band*, computes features, runs XGB, and saves spectrograms with prediction, LO/band, and basic metadata.

- **SDR path** (`jammertest_day2/sdr/`):
  - `sdr_xgb_bleik_d2.py`  
    Opens Kraken raw IQ from the Bleik ramp test, takes periodic IQ “snaps”, computes the 78-dim feature vector, runs XGB, and annotates spectrograms with:
    - testplan context,
    - approximate time,
    - ramp phase (e.g. `ramp_up` / `ramp_down`),
    - logbook label → canonical GT class,
    - predicted class + probability.

Exploratory helpers:

- `sdr_iq_open.py` : low-level Kraken reader.
- `bleik_d2.py`, `grunvatn_d2.py` : site-specific exploration / plotting.

All these scripts have a `USER CONFIG` section at the top where you set:
paths, model, sampling period, STFT settings, etc.

---

## Documentation

- `docs/features.md`  
  Detailed documentation of the 78 hand-crafted features produced by  
  `extract_features(iq, fs)`:
  - categories (power, spectral shape, chirpiness, cyclostationarity, etc.),
  - formulas using $...$ / $$...$$,
  - intuitive explanations of what each feature measures.

- `docs/pipeline.md`  
  High-level overview of the full pipeline:
  synthetic training → model artifacts → Day 1 / Day 2 validation → legacy Zenodo experiments.


