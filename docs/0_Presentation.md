# 0. Presentation

## Context and Motivation

Global Navigation Satellite Systems (GNSS) are a critical dependency for positioning, navigation, and timing (PNT) services across civil, commercial, and safety‑critical domains. The low received power of GNSS signals at the Earth’s surface makes them particularly vulnerable to radio‑frequency interference (RFI), whether intentional (jamming, spoofing) or unintentional.

Among the different classes of GNSS interference, jamming remains one of the most prevalent and operationally relevant threats. Commercially available jammers, low‑cost devices, and improvised transmitters can significantly degrade or fully deny GNSS services over localized or extended areas. Detecting, classifying, and characterizing such interference in a reliable and scalable way is therefore a key technical challenge.

This repository addresses that challenge by focusing on **GNSS jamming detection directly from raw receiver baseband recordings**, without relying on navigation‑level observables or receiver‑internal proprietary metrics.

---

## Scope of the Work

The work documented in this repository focuses on:

- Detection of GNSS interference at the **RF / baseband level**
- Classification of interference into a limited but operationally relevant taxonomy:
  - No interference
  - Narrowband interference
  - Chirp / sweep interference
  - Wideband interference
- Evaluation of multiple machine‑learning approaches under both controlled and real‑world conditions
- Analysis of the trade‑off between **detection performance** and **computational cost**

The scope explicitly excludes:
- GNSS spoofing detection beyond incidental interference‑like effects
- Receiver tracking or navigation solution mitigation
- Real‑time embedded implementation

All processing is performed offline on recorded baseband data.

---

## Design Philosophy

Several guiding principles shape the design of the system:

1. **Raw‑signal driven**  
   All models operate on raw or minimally processed baseband samples, avoiding reliance on receiver‑specific observables.

2. **Model diversity**  
   Both classical feature‑based machine learning and deep learning approaches are implemented and compared.

3. **Controlled and uncontrolled data**  
   Training and validation rely on labelled data generated under controlled conditions, while testing includes long‑duration, unlabelled, in‑the‑wild recordings.

4. **Operational realism**  
   Computational cost, scalability, and robustness to recording artifacts are treated as first‑order concerns.

---

## Data Strategy Overview

Three distinct data regimes are used throughout the work:

### Training
Training data consists primarily of **synthetically generated GNSS interference**, allowing precise control over:
- Interference type
- Bandwidth and sweep parameters
- Signal‑to‑noise and signal‑to‑interference ratios

This synthetic generator is documented separately and referenced here for completeness:
https://github.com/macaburguera/GNSS_generator

### Validation
Validation uses **real, labelled GNSS baseband recordings** obtained during the Jammertest 2023 event. These data provide:
- Known transmission scenarios
- Controlled jammer configurations
- Ground‑truth labels suitable for quantitative evaluation

### Test
Testing is performed on **unseen, unlabelled, real‑world data**, referred to throughout the documentation as the *Roadtest* dataset.  
This dataset consists of approximately **15 days of continuous GNSS recording** from a Septentrio receiver placed along a highway near Rødby, Denmark.

The test phase is explicitly treated as:
- A qualitative and semi‑quantitative assessment
- A robustness and scalability check
- An exploration of persistent and recurring interference phenomena

---

## Labelling and Ground Truth

Label generation for real data relies on an external, purpose‑built labelling tool:

https://github.com/macaburguera/sbf-labeller

This tool enables:
- Time‑aligned annotation of SBF baseband blocks
- Consistent class definitions across datasets
- Export of labels compatible with both feature‑based and deep‑learning pipelines

The separation between labelled and unlabelled data is strictly enforced to avoid leakage between training, validation, and test stages.

---

## Implemented Approaches

Two complementary detection approaches are implemented:

### Feature‑Based Machine Learning
Explicit signal features are extracted from baseband samples, capturing spectral, temporal, and modulation‑related characteristics. These features are used to train XGBoost classifiers with different feature set sizes and retraining strategies.

### Deep Learning on Spectrograms
A convolutional neural network operates directly on time‑frequency representations derived from short‑time Fourier transforms (STFT). This approach minimizes manual feature engineering and emphasizes inference speed and scalability.

Both approaches are treated as first‑class citizens throughout the repository and are evaluated using consistent datasets and metrics.

---

## Intended Audience

This repository is intended for:
- GNSS researchers and engineers
- Signal processing practitioners working with RF recordings
- Developers evaluating ML‑based interference detection strategies

The documentation assumes familiarity with:
- GNSS signal fundamentals
- Basic digital signal processing concepts
- Machine‑learning terminology

---

## Document Structure

The documentation is organized as follows:

- **0. Presentation** – Context, scope, and high‑level design
- **1. Pipeline** – End‑to‑end system overview
- **2. Data** – Detailed description of datasets and splits
- **3. Features** – Feature taxonomy and signal interpretation
- **4. XGB** – Feature‑based model design and training
- **5. DL Spectrogram** – Deep learning model design
- **6. Validation** – Quantitative evaluation and comparisons
- **7. Test (Roadtest)** – Real‑world deployment and observations
- **8. Conclusions** – Summary and recommendations

Each document is written to be readable independently, while remaining consistent with the overall system description.
