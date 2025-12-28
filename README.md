# GNSS Jamming Classifier

This repository contains a complete machine-learning pipeline for **GNSS interference and jamming detection** using raw receiver baseband data.

The system operates directly on recorded IQ samples and produces time-aligned, multi-class interference detections, covering:

- No interference
- Narrowband interference
- Chirp / sweep interference
- Wideband interference

Two complementary modelling approaches are implemented and evaluated:

- Feature-based classical machine learning (XGBoost)
- Deep learning on time–frequency representations (spectrogram CNN)

The repository supports controlled experimentation using synthetic data as well as validation and testing on real-world GNSS interference recordings.

---

## Repository Structure

```
GNSS-jamming-classifier/
│
├── artifacts/              # Trained models and experiment outputs
├── docs/                   # Technical documentation
├── scripts/                # Training, inference, validation scripts
├── features/               # Feature extraction logic
├── dl/                     # Deep learning models and preprocessing
├── utils/                  # Shared utilities
├── requirements.txt
└── README.md
```

---

## Data Sources

Two complementary data sources are used:

### 1. Synthetic GNSS Interference Generator
Controlled generation of narrowband, chirp, and wideband interference is used for training and stress testing of the models.  
This enables precise control over interference parameters and clean ground-truth labels.

### 2. Real-World GNSS Recordings
- **Jammertest 2023**: Controlled large-scale GNSS interference experiments with known transmission scenarios and labelled data.
- **Roadtest (Denmark)**: A 15-day continuous highway recording using a Septentrio receiver, representing in-the-wild, unlabelled GNSS RF environments.

A strict separation is maintained between training, validation, and test data.

---

## Models Implemented

### XGBoost (Feature-based)
- Full feature model (78 engineered features)
- Minimal feature model (10 features)
- Retrained variants using updated datasets

These models rely on explicit spectral, temporal, and statistical features extracted from baseband data.

### Deep Learning (Spectrogram-based)
- Convolutional neural network operating on STFT-based log-power spectrograms
- Blockwise inference aligned with SBF baseband sample blocks
- Designed for efficient large-scale scanning of recordings

---

## Validation and Performance

The repository includes tools for:

- Accuracy and confusion matrix computation
- Per-class performance evaluation
- Detailed computational cost profiling (I/O, preprocessing, inference)

Comparative results show that:

- Feature-based models provide strong accuracy and interpretability
- The deep learning model achieves significantly lower inference latency and better scalability for long-duration recordings

---

## Intended Use

This repository is intended for:

- GNSS interference and jamming research
- Offline analysis of GNSS baseband recordings
- Evaluation of detection algorithms under controlled and real-world conditions

It is not intended to function as a real-time GNSS receiver.

---

## License

This project is provided for research and experimental use.
