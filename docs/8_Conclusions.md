# 8. Conclusions

This document summarizes the measured outcomes of the project and updates any earlier conclusions that relied on approximate validation numbers.

All quantitative statements below are grounded in:
- `results/*/samples_eval.csv`
- `results/*/summary.json`
- `results/*/timing_summary.json` (DL)

---

## 8.1 What worked best (on labelled real data)

On the fixed validation dataset (**Altus06, 150 m, 12,689 blocks**), retraining on real labelled data produced near-ceiling performance for both approaches:

| Model | Acc (all) | Acc (4-class) | Macro-F1 (4-class) | FAR NoJam | Recall NB | Recall WB | Mean ms/block |
|---|---|---|---|---|---|---|---|
| XGB-78 (synthetic-only) | 0.913153 | 0.913297 | 0.620660 | 0.15% | 30.48% | 56.25% | 31.657 |
| XGB-78 (retrained) | 0.997163 | 0.997320 | 0.992355 | 0.17% | 98.66% | 96.88% | 30.428 |
| XGB-10 (retrained) | 0.994955 | 0.994955 | 0.833918 | 0.33% | 99.25% | 21.88% | 24.770 |
| DL spectrogram (synthetic-only) | 0.744582 | 0.744699 | 0.599334 | 32.51% | 65.13% | 100.00% | 1.118 |
| DL spectrogram (retrained) | 0.997872 | 0.998029 | 0.997375 | 0.26% | 99.85% | 100.00% | 1.688 |

Key points:
- **DL retrained** achieves the best overall balance (Macro-F1 ≈ 0.997 on 4 classes).
- **XGB retrained** is essentially tied in accuracy, slightly below DL on Macro-F1.
- **XGB-10 retrained** keeps NB strong but fails on WB; it is a clear speed/robustness trade.

---

## 8.2 The most important technical lesson: synthetic-only is not enough

The project confirms a strong **synthetic → real domain gap**, and the gap is model-dependent:

- **DL (synthetic-only)**  
  - Accuracy collapses to **~0.745** on real data.
  - Dominant error: **NoJam → Chirp** (high false alarm rate on clean data).

- **XGB (synthetic-only)**  
  - Accuracy remains **~0.913**, but Macro-F1 is only **~0.621** because:
    - NB recall is **~0.305**
    - WB recall is **~0.563**
    - NB is often predicted as NoJam.

Practical interpretation:
- If you want a detector that is usable in operational recordings, **real-data retraining (or strong domain randomization) is mandatory**.

---

## 8.3 WB is a fragile class (in this dataset)

WB appears only **32 times** (0.25%).
This has two consequences:

1. **Metrics can look “perfect” by chance** (small-N effect).
2. Feature selection decisions that do not preserve WB structure can break WB detection (seen in XGB-10).

Example from the 10-feature model:
- WB recall drops to **~21.9%**, mainly because WB is classified as NB.

---

## 8.4 Computational cost and deployment implications

Per-block mean timings (from the evaluation outputs):

| Model | NPZ load (ms) | Feat. extract (ms) | Inference (ms) | Spectrogram (ms) | Total (ms) |
|---|---|---|---|---|---|
| XGB-78 (synthetic-only) | 0.576 | 26.553 | 2.676 | — | 31.657 |
| XGB-78 (retrained) | 0.558 | 24.999 | 3.145 | — | 30.428 |
| XGB-10 (retrained) | 0.523 | 22.098 | 2.149 | — | 24.770 |
| DL spectrogram (synthetic-only) | 0.691 | — | 0.487 | 0.345 | 1.118 |
| DL spectrogram (retrained) | 1.016 | — | 0.529 | 0.543 | 1.688 |

Interpretation:
- **XGB** is dominated by feature extraction (≈22–26 ms in this configuration).
- **DL** is extremely fast *in this measurement setup* (≈1–2 ms), because inference ran on CUDA (recorded in DL summaries).

Deployment note:
- If CUDA is unavailable, DL throughput should be re-profiled on CPU. The current timings are still a strong indicator that the spectrogram pipeline is computationally lightweight compared to the 78-feature pipeline.

---

## 8.5 What to improve next

1. **Broader validation**
   - More days, more receivers, more distances, and more WB examples.
   - WB performance should be validated on larger counts before any strong claim.

2. **Better synthetic-to-real transfer**
   - Increase realism of the synthetic generator (front-end effects, noise coloration, AGC, frequency-dependent artifacts).
   - Consider domain randomization and augmentation targeted at the *observed* failure modes (e.g., NoJam→Chirp false alarms for DL).

3. **Thresholding / veto logic**
   - Current evaluation ran with veto disabled (see `veto` entries in summaries).
   - A calibrated veto could reduce false positives further, especially for operational “scan” use.

4. **Feature set design**
   - The 10-feature model shows that aggressive reduction can harm specific classes.
   - A class-aware selection process (or adding WB-specific features) is the safer path if the goal is a minimal XGB model.

---

## 8.6 Bottom line

- **Retraining on real labelled data is the decisive step** for both ML (XGB) and DL (spectrogram CNN) in this pipeline.
- **DL retrained** offers the best combination of accuracy and throughput (in the measured CUDA setup).
- **XGB retrained** remains attractive when interpretability and classical feature reasoning matter, at the cost of higher per-block compute.
- Any “WB performance claim” must be stated carefully due to the small sample count in the current validation dataset.
