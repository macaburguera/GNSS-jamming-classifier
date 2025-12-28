# 2. Data

## Overview

This document describes in detail the data sources used throughout the project, the rationale behind their selection, and the way they are partitioned into training, validation, and test sets.

A central design goal of the project is to ensure that **model evaluation reflects realistic operational conditions**, while still allowing controlled experimentation during development. To achieve this, three distinct data regimes are used, each serving a specific role in the pipeline.

---

## Data Regimes and Their Roles

The project distinguishes clearly between:

- **Training data**: used to fit model parameters
- **Validation data**: used to evaluate generalization under controlled conditions
- **Test data**: used to assess robustness on unseen, unlabelled, real-world recordings

No data overlap exists between these regimes.

---

## Synthetic Interference Data (Training)

### Purpose

Synthetic data is used primarily for **training** and early-stage model development. The key motivation is the need for:

- Precise and unambiguous ground truth
- Systematic coverage of interference parameter spaces
- Reproducibility and scalability

Real-world GNSS interference data is scarce, heterogeneous, and often poorly labelled. Synthetic generation addresses these limitations.

---

### Synthetic Generator

Synthetic data is produced using a dedicated GNSS interference generator:

https://github.com/macaburguera/GNSS_generator

The generator operates at the baseband level and produces complex IQ samples compatible with the downstream pipeline.

---

### Interference Types

The following interference classes are generated:

- **No interference** (clean GNSS-like noise conditions)
- **Narrowband interference**
  - Continuous wave or quasi-continuous tones
  - Fixed or slowly varying center frequency
- **Chirp / sweep interference**
  - Linear or near-linear frequency sweeps
  - Configurable sweep rates and bandwidths
- **Wideband interference**
  - Noise-like signals spanning a large fraction of the band

Each class is parameterized to span a wide range of realistic scenarios.

---

### Parameter Coverage

Synthetic data generation covers variations in:

- Signal-to-noise ratio (SNR)
- Signal-to-interference ratio (SIR)
- Interference bandwidth
- Sweep slope and duration
- Time occupancy within a block

This diversity is critical to avoid overfitting to narrow signal archetypes.

---

### Labelling

All synthetic data is **perfectly labelled by construction**. Labels are assigned at the block level and remain fixed throughout training and validation.

---

## Real-World Labelled Data (Validation)

### Jammertest 2023

Validation relies on labelled real-world recordings obtained during **Jammertest 2023**, a controlled GNSS interference testing campaign.

These recordings provide:

- Real RF propagation effects
- Hardware-induced distortions
- Realistic interference dynamics
- Known jammer configurations and timelines

---

### Recording Characteristics

The Jammertest dataset consists of:

- Septentrio receiver baseband recordings (SBF format)
- Multiple days and test scenarios
- Multiple GNSS frequency bands
- Diverse interference types, including mixed scenarios

Compared to synthetic data, these recordings exhibit significantly higher variability and complexity.

---

### Labelling Process

Labelling of Jammertest data is performed using a dedicated tool:

https://github.com/macaburguera/sbf-labeller

This tool enables:

- Time-aligned inspection of baseband blocks
- Manual and semi-automatic annotation
- Export of labels aligned with SBF block indices

Labels are defined consistently with the synthetic data taxonomy.

---

### Role in the Pipeline

Jammertest data is used exclusively for **validation**, not for training. This allows:

- Quantitative performance evaluation
- Confusion matrix analysis
- Detection of domain gaps between synthetic and real data

---

## Real-World Unlabelled Data (Test)

### Roadtest Dataset

The test phase relies on a long-duration, unlabelled dataset referred to as the **Roadtest** dataset.

This dataset consists of approximately **15 consecutive days of GNSS baseband recording**, collected using a Septentrio receiver deployed along a highway near Rødby, Denmark.

---

### Characteristics

Key characteristics of the Roadtest dataset include:

- Continuous recording over extended periods
- Real traffic and environmental RF conditions
- Absence of controlled jammer setups
- Presence of persistent and intermittent interference sources

This dataset is representative of realistic operational monitoring conditions.

---

### Interpretation of “Test”

Within this project, the term *test* is used deliberately to mean:

- Evaluation on **unseen and unlabelled data**
- No ground-truth-based accuracy computation
- Emphasis on qualitative behavior, stability, and plausibility

Results on the Roadtest dataset are interpreted cautiously and are not used to tune model parameters.

---

## Data Integrity and Separation

Strict measures are taken to prevent data leakage:

- Synthetic data is never mixed with real data during validation
- Jammertest recordings are never used for training
- Roadtest data is never labelled retroactively for model tuning

All datasets are tracked separately in scripts and directory structures.

---

## Data Limitations and Biases

Despite the diversity of data sources, certain limitations remain:

- Synthetic data may not fully capture real-world RF impairments
- Real-world labelled data remains limited in volume
- Certain interference types (e.g., wideband) are underrepresented in some real datasets

These limitations are explicitly considered in the validation and conclusions documents.

---

## Summary

The data strategy adopted in this project prioritizes:

- Controlled learning through synthetic generation
- Realism through validated experimental data
- Robustness assessment through long-term, unlabelled recordings

This layered approach enables systematic development while maintaining operational relevance.
