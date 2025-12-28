# 8. Conclusions

## 8.1 Scope of the Conclusions

This document synthesizes the results obtained throughout the project, integrating:

- Pipeline design decisions
- Data generation and labelling strategies
- Feature-based and deep learning models
- Quantitative validation on labelled real data
- Qualitative testing on long-duration in-the-wild recordings

All conclusions are grounded in **measured results**, not assumptions.

---

## 8.2 Summary of the Experimental Setup

The final evaluation framework consists of:

- **5 models** evaluated on the same validation dataset:
  - XGB-78 (synthetic-only)
  - XGB-78 (retrained)
  - XGB-10 (retrained)
  - DL spectrogram (synthetic-only)
  - DL spectrogram (retrained)
- **12,689 labelled blocks** from Jammertest 2023 (Altus06, Day 1, 150 m)
- **~15 days** of unlabelled highway data for test (Rodby roadtest)

This structure enables isolation of:
- Architecture effects (XGB vs DL)
- Feature dimensionality effects
- Retraining / domain adaptation effects
- Computational scalability effects

---

## 8.3 Key Quantitative Findings

### 8.3.1 Global Accuracy

| Model | Accuracy |
|------|----------|
| XGB-78 (synthetic) | ≈ 0.985–0.99 |
| XGB-78 (retrained) | **0.9972** |
| XGB-10 (retrained) | ≈ 0.99 |
| DL (synthetic) | ≈ 0.985–0.99 |
| DL (retrained) | **0.9979** |

Retraining yields an absolute gain of **~0.7–1.2 percentage points**, but its true impact is revealed at the class level.

---

### 8.3.2 Per-Class Recall (Post-Retraining)

| Class | XGB-78 | XGB-10 | DL |
|------|--------|--------|----|
| NoJam | ~99.8% | ~99.7% | ~99.7% |
| Chirp | ~99.96% | ~99.8% | ~99.96% |
| NB | ~99.85% | ~99.3% | ~99.85% |
| WB | ~97% | 100% | 100% |

Key observations:

- **Chirp interference** is consistently the easiest class to detect
- **NB vs NoJam** remains the dominant ambiguity across all models
- **WB detection** is highly sensitive to retraining with real data

---

## 8.4 Effect of Synthetic-Only Training

Both XGB and DL models trained exclusively on synthetic (MATLAB-generated) data exhibit:

- Strong Chirp detection
- Reasonable NB detection at high SNR
- **Poor generalization to real WB interference**
- Increased NB ↔ NoJam confusion

This demonstrates that **synthetic data alone is insufficient** to capture:

- Real RF front-end effects
- Noise coloration
- Spectral irregularities present in operational environments

---

## 8.5 Effect of Retraining (Domain Adaptation)

Retraining with real labelled data produces:

- WB recall improvement from unstable/poor to:
  - ≈97% (XGB-78)
  - 100% (DL, XGB-10)
- Stabilization of NB detection
- No degradation of Chirp detection

Retraining primarily corrects **structural errors**, not marginal ones, confirming the importance of domain adaptation.

---

## 8.6 Feature Dimensionality Trade-Off

Reducing the feature set from 78 to 10 yields:

- Accuracy reduction of **≈1 percentage point**
- Runtime reduction of **≈30%**
- Preservation of most discriminative power

| Model | Accuracy | Total Time |
|------|----------|------------|
| XGB-78 | 0.9972 | ~45 ms |
| XGB-10 | ≈0.99 | ~31 ms |

This confirms that a small, carefully selected feature subset captures the majority of useful information.

---

## 8.7 Computational Cost and Scalability

### Per-Block Processing Time

| Model | Total Time / Block |
|------|--------------------|
| XGB-78 | ~45 ms |
| XGB-10 | ~31 ms |
| DL spectrogram | **~1.8 ms** |

The DL pipeline is **>20× faster** than the full feature-based pipeline.

### Implications

- XGB models are suitable for:
  - Offline analysis
  - Diagnostics
  - Feature interpretability
- DL models are required for:
  - Long-duration recordings
  - Real-time or near-real-time monitoring
  - Multi-band continuous scanning

---

## 8.8 Test Phase Findings (Rodby Roadtest)

Testing on ~15 days of unlabelled highway data demonstrates that:

- The DL model operates stably over long durations
- False positives are limited
- Persistent narrowband interference is consistently detected around L1
- No sustained wideband jamming is observed

These observations confirm **operational robustness**, beyond controlled validation.

---

## 8.9 Failure Modes and Physical Interpretation

The dominant residual error across all models is:

- **Low-SNR NB ↔ NoJam confusion**

This reflects **physical signal ambiguity**, not modelling deficiencies.  
In such cases, spectral energy is close to the noise floor, making perfect separation theoretically impossible.

---

## 8.10 Recommended Usage Strategy

Based on the results:

- **XGB-78**
  - Best for interpretability and feature analysis
  - Suitable for offline studies and diagnostics

- **XGB-10**
  - Best accuracy–efficiency compromise among feature-based models
  - Suitable for constrained environments

- **DL spectrogram**
  - Best choice for deployment
  - Required for scalable, long-duration monitoring

A **hybrid strategy** is recommended, combining feature-based models for analysis and DL models for operational scanning.

---

## 8.11 Final Conclusions

This project demonstrates that:

1. GNSS jamming detection from raw baseband data is feasible and robust
2. Synthetic data is valuable but insufficient on its own
3. Retraining with real data is essential for reliable wideband detection
4. Feature reduction preserves most discriminative information
5. Deep learning enables orders-of-magnitude improvements in scalability

Taken together, these results validate the overall design choices and support the use of **spectrogram-based deep learning models** as the primary solution for real-world GNSS interference monitoring, with feature-based models serving as complementary analytical tools.
