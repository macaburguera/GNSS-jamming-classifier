# GNSS Jamming Classifier

## Overview

This repository contains an **end-to-end GNSS interference and jamming detection pipeline** operating directly on **raw GNSS receiver baseband data**.  
The project focuses on detecting and classifying common GNSS interference types under both **controlled conditions** and **real-world operational environments**, with particular attention to **robustness, interpretability, and computational scalability**.

The system combines:

- Synthetic-data-based training
- Real-world data retraining and validation
- Feature-based machine learning models (XGBoost)
- Deep learning models based on spectrogram representations
- Detailed timing and performance analysis
- Long-duration testing on unlabelled data

The repository is structured to support **research, experimentation, and reproducible evaluation**, rather than providing a black-box detector.

---

## Interference Classes

The implemented models classify each baseband block into one of the following classes:

- **NoJam** – Nominal GNSS signal conditions  
- **Chirp** – Frequency-swept interference  
- **NB (Narrowband)** – Continuous-wave or very narrowband interference  
- **WB (Wideband)** – Broadband, noise-like interference  

These classes cover the most common and operationally relevant GNSS interference types.

---

## Repository Structure

The repository is organized as follows:

```
GNSS-jamming-classifier/
│
├── docs/                 # Project documentation (usage, pipeline, data, models, validation, test, conclusions)
│
├── explore/              # Exploratory scripts and analysis experiments
│
├── retrain/              # Model retraining / domain adaptation with real labelled data
│
├── test/                 # Test-time scripts (unlabelled, in-the-wild data)
│
├── train/                # Training scripts (synthetic-data-based training)
│
├── validation/           # Validation and benchmarking on labelled real data
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

No directories or files outside this structure are required to run the provided experiments.

---

## Data Sources

Two complementary data sources are used.

### Synthetic Data

Synthetic GNSS interference data is generated using a dedicated external generator:

- https://github.com/macaburguera/GNSS_generator

This generator produces controlled instances of:
- Narrowband interference
- Chirp interference
- Wideband interference

Synthetic data is used for:
- Initial model training
- Controlled experiments
- Feature and model ablation studies

---

### Real Data (Jammertest 2023)

Real labelled data originates from **Jammertest 2023**, recorded using Septentrio receivers and raw baseband output.

Labelling is performed with:

- https://github.com/macaburguera/sbf-labeller

This data is used for:
- Quantitative validation
- Model retraining (domain adaptation)
- Performance benchmarking under realistic RF conditions

---

### Real Data (Roadtest – Rodby Highway)

A long-duration roadtest dataset was collected by operating a receiver on the **Rodby highway (Denmark)** for approximately **15 days**.

Key characteristics:
- No ground truth labels
- Strong environmental variability
- Realistic operational conditions

This dataset is used strictly for **test-time evaluation**, not validation.

---

## Modelling Approaches

Two modelling strategies are implemented.

### Feature-Based Models (XGBoost)

- Operate on hand-crafted features extracted from IQ samples
- Two configurations:
  - Full feature set (≈78 features)
  - Reduced feature set (10 features)
- Advantages:
  - Interpretability
  - Feature-level diagnostics
  - Strong performance on structured interference

---

### Deep Learning Models (Spectrogram-Based CNN)

- Operate on STFT spectrograms computed from raw IQ samples
- End-to-end convolutional neural networks
- Advantages:
  - Very low inference latency
  - High scalability
  - Strong performance after retraining with real data

---

## Validation

Validation is performed using labelled real data from Jammertest 2023.

Key aspects:
- Identical datasets used across models
- Detailed confusion matrices and per-class metrics
- Explicit timing measurements for each pipeline stage

Relevant scripts are located in:
- `validation/validation_xgb_*`
- `validation/validation_dl_*`

Validation artifacts (metrics, confusion matrices, predictions) are stored under:
- `artifacts/finetuned/`

---

## Test Phase

Testing is performed on **unlabelled roadtest data**, focusing on:

- Detection plausibility
- Temporal consistency
- False positive behavior
- Long-duration stability

Scripts supporting this phase are located in:
- `test/`

No quantitative accuracy metrics are computed during testing.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The codebase has been developed and tested with Python 3.9+.

---

## Usage Notes

This repository is designed for **experimentation and analysis** rather than turnkey deployment.

Typical workflows include:
- Training or retraining models
- Running validation on labelled datasets
- Inspecting confusion matrices and timing statistics
- Running long-duration scans on unlabelled recordings

Refer to the individual scripts for usage details.

---

## Scope and Limitations

- The system focuses on **interference detection and classification**, not mitigation
- It operates at the **baseband signal level**
- Certain low-SNR cases remain physically ambiguous
- Results depend on receiver front-end characteristics

---

## License

This repository is intended for research and educational use.  
Refer to the LICENSE file or the repository metadata for licensing details.
