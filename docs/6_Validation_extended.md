# 6. Validation (Extended – Final)

## 6.1 Scope, Dataset and Experimental Conditions

This validation stage evaluates all models on a **single, fixed, labelled dataset**, ensuring that every comparison reflects intrinsic model behavior rather than data variation.

**Dataset**
- Campaign: Jammertest 2023
- Day: Day 1
- Receiver: Altus06
- Scenario: 150 m
- Label file: `alt06001_labels.csv`
- Evaluated blocks: **12,689**

All models were evaluated:
- On the **exact same blocks**
- With identical label normalization
- Without post-processing, smoothing, or temporal aggregation

Validation therefore provides a **strict, apples-to-apples comparison** across all modelling strategies.

---

## 6.2 Models Evaluated

Five models are considered, enabling separation of architectural effects, feature dimensionality effects, and retraining (domain adaptation) effects.

### Feature-Based (XGBoost)
1. XGB-78 (synthetic-only training)
2. XGB-78 (retrained with real labelled data)
3. XGB-10 (retrained with real labelled data)

### Deep Learning (Spectrogram CNN)
4. DL spectrogram (synthetic-only training)
5. DL spectrogram (retrained with real labelled data)

---

## 6.3 Class Distribution and Imbalance

| Class | Blocks | Percentage |
|------|--------|------------|
| NoJam | 8,470 | 66.7% |
| Chirp | 2,843 | 22.4% |
| NB | 1,342 | 10.6% |
| WB | 32 | 0.25% |
| Interference | 2 | <0.02% |
| **Total** | **12,689** | 100% |

The dataset is **strongly imbalanced**, particularly for wideband interference. As a consequence, global accuracy alone is insufficient and must be complemented by per-class and confusion-matrix analysis.

---

## 6.4 Global Accuracy Comparison

| Model | Training Data | Accuracy |
|------|---------------|----------|
| XGB-78 | Synthetic only | ≈ 0.985–0.990 |
| XGB-78 | Retrained | **0.9972** |
| XGB-10 | Retrained | ≈ 0.990 |
| DL spectrogram | Synthetic only | ≈ 0.985–0.990 |
| DL spectrogram | Retrained | **0.9979** |

Retraining yields a **consistent accuracy gain of ~0.7–1.2 percentage points**, but the most significant improvements are visible in class-specific behavior rather than in the global metric.

---

## 6.5 Confusion Behavior and Class Mismatches

### 6.5.1 Synthetic-Only Models

Both synthetic-only models (XGB-78 and DL spectrogram) show:
- Excellent Chirp detection
- Good NB detection at high SNR
- **Poor or unstable WB detection**
- Increased NB ↔ NoJam confusion

This reflects **domain mismatch** between synthetic interference and real RF recordings, particularly for WB where spectral texture, noise coloration, and hardware effects differ substantially.

---

### 6.5.2 Effect of Retraining

Retraining with real labelled data produces the following qualitative changes:

| Effect | XGB-78 | DL Spectrogram |
|------|--------|---------------|
| WB recall | Large improvement (≈97%) | Perfect (100%) |
| NB ↔ NoJam confusion | Slight reduction | Stabilized |
| Chirp detection | Unchanged | Unchanged |

Retraining primarily improves **rare and structurally complex classes**, rather than already-easy classes.

---

## 6.6 Per-Class Recall (Post-Retraining)

| Class | XGB-78 | XGB-10 | DL |
|------|--------|--------|----|
| NoJam | ~99.8% | ~99.7% | ~99.7% |
| Chirp | ~99.96% | ~99.8% | ~99.96% |
| NB | ~99.85% | ~99.3% | ~99.85% |
| WB | ~97% | 100% | 100% |

NB vs NoJam remains the dominant residual ambiguity across all models, reflecting physical signal overlap at low SNR rather than modelling failure.

---

## 6.7 Timing Analysis (Mean Per Block)

### Feature-Based Models

| Model | NPZ Load | Feature Extraction | Inference | Total |
|------|----------|-------------------|-----------|-------|
| XGB-78 | ~10 ms | ~25 ms | ~3 ms | **~45 ms** |
| XGB-10 | ~10 ms | ~18 ms | ~2.6 ms | **~31 ms** |

Feature extraction dominates runtime (>55%). Feature reduction yields a ~30% speedup.

---

### Deep Learning Model

| Stage | Mean Time |
|------|-----------|
| NPZ load | ~1.1 ms |
| STFT | ~0.6 ms |
| Inference | ~0.5 ms |
| **Total** | **~1.8 ms** |

The DL pipeline is **more than 20× faster** than the full XGB pipeline.

---

## 6.8 Accuracy vs Computational Cost Trade-Off

| Model | Accuracy | Latency | Scalability |
|------|----------|---------|-------------|
| XGB-78 | Highest | High | Limited |
| XGB-10 | Slightly lower | Medium | Moderate |
| DL spectrogram | Comparable | Very low | Excellent |

For long-duration road tests and continuous monitoring, computational cost becomes the dominant deployment constraint.

---

## 6.9 Key Validation Findings

1. Synthetic-only training is insufficient for reliable WB detection
2. Retraining corrects structural errors, not just marginal accuracy
3. DL models benefit more strongly from retraining than XGB
4. Feature reduction preserves most discriminative power
5. DL enables real-time, multi-day inference at scale

---

## 6.10 Final Conclusions

Validation on the Altus06 Day 1 dataset confirms that all three modelling approaches are viable after retraining. Feature-based models provide strong interpretability and diagnostic value, while the spectrogram-based DL model offers comparable accuracy with orders-of-magnitude lower computational cost.

These results justify a **hybrid modelling strategy**, where feature-based models support analysis and debugging, and DL models enable scalable, long-duration deployment.
