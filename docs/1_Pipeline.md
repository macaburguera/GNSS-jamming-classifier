# 1. Pipeline

## Overview

This document provides a high-level overview of the end-to-end processing pipeline implemented in this repository. The pipeline describes the flow of data, models, and evaluation stages from raw GNSS baseband recordings to interference detections and performance assessment.

The pipeline has been designed to support both **controlled experimentation** and **real-world deployment-style analysis**, while maintaining a clear separation between training, validation, and test phases.

---

## High-Level Architecture

At a conceptual level, the system follows the pipeline below:

```
Raw GNSS Baseband Data (SBF)
        │
        ▼
Block Extraction & Alignment
        │
        ▼
Preprocessing
(IQ normalization, windowing)
        │
        ├───────────────┐
        ▼               ▼
Feature Extraction   Time–Frequency Transform
        │               │
        ▼               ▼
XGBoost Classifier   CNN Spectrogram Model
        │               │
        └───────┬───────┘
                ▼
        Block-Level Predictions
                │
                ▼
Post-processing & Aggregation
                │
                ▼
Evaluation / Analysis / Reporting
```

Each stage is implemented explicitly and is configurable through scripts and configuration files.

---

## Input Data Handling

### Raw Baseband Recordings

The primary input to the pipeline consists of **raw GNSS baseband recordings** stored in Septentrio Binary Format (SBF). These recordings contain time-tagged blocks of complex IQ samples for multiple local oscillators (LOs) and frequency bands.

Key characteristics:
- High sampling rates (tens of MHz)
- Time-aligned baseband blocks
- Multi-band GNSS coverage

The pipeline deliberately operates at this level to remain independent of receiver-specific tracking or navigation internals.

---

## Block-Based Processing

All downstream processing is performed on **fixed-size baseband blocks**, corresponding to the native block structure produced by the receiver.

This design choice provides:
- Precise time alignment between detections and RF events
- Consistent model input dimensionality
- Straightforward scalability to long recordings

Block indices and timestamps are preserved throughout the pipeline to enable traceability back to the original recordings.

---

## Preprocessing Stage

Before any model-specific processing, baseband samples undergo a minimal preprocessing step, which may include:

- Centering and normalization
- Optional zero-padding or cropping to a fixed length
- Windowing for spectral analysis

Preprocessing is intentionally lightweight and deterministic to avoid introducing model-dependent biases.

---

## Dual-Branch Modelling Strategy

The pipeline diverges into two parallel modelling branches, reflecting two complementary detection philosophies.

### Feature-Based Branch

In the feature-based branch:

1. Explicit signal features are computed from each baseband block.
2. Features capture spectral shape, temporal behavior, envelope statistics, and modulation patterns.
3. Feature vectors are passed to an XGBoost classifier.
4. The classifier outputs a block-level class prediction.

This branch emphasizes interpretability and controlled experimentation.

### Deep Learning Branch

In the deep learning branch:

1. Baseband blocks are converted into time–frequency representations using STFT.
2. Log-power spectrograms are computed with fixed FFT and hop parameters.
3. Spectrograms are normalized and formatted as CNN inputs.
4. A convolutional neural network produces block-level predictions.

This branch emphasizes inference speed, scalability, and reduced manual feature engineering.

---

## Post-Processing and Aggregation

Model outputs are produced at the **block level**. Depending on the use case, these outputs can be:

- Logged directly as time-aligned detections
- Aggregated temporally to identify sustained interference events
- Filtered by frequency band or interference type

Post-processing scripts are kept separate from model inference to allow flexible experimentation.

---

## Training, Validation, and Test Separation

A strict separation is enforced throughout the pipeline:

- **Training**  
  Uses synthetic, labelled interference data only.

- **Validation**  
  Uses labelled real-world data from controlled experiments (Jammertest 2023).

- **Test**  
  Uses unlabelled, real-world recordings (Roadtest dataset).

No data, labels, or statistics are shared across these phases.

---

## Evaluation and Profiling

Evaluation scripts compute:

- Classification accuracy and confusion matrices
- Per-class performance metrics
- Computational cost breakdowns, including:
  - Data loading
  - Preprocessing
  - Feature extraction or spectrogram generation
  - Model inference

Timing measurements are treated as first-class outputs and are used to guide model selection.

---

## Repository Organization

The repository structure mirrors the pipeline stages:

- `scripts/` – Orchestration of training, inference, and evaluation
- `features/` – Feature extraction logic
- `dl/` – Deep learning models and preprocessing
- `utils/` – Shared utilities
- `artifacts/` – Stored models and experiment outputs
- `docs/` – Technical documentation

This organization is intended to make the data flow and model dependencies explicit.

---

## Design Rationale

The pipeline reflects a deliberate balance between:

- Signal-processing transparency
- Machine-learning flexibility
- Operational realism

By supporting both feature-based and deep-learning approaches within a unified framework, the pipeline enables systematic comparison and informed trade-offs between accuracy, interpretability, and computational cost.
