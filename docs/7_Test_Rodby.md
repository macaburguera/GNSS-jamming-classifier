# 7. Test on Roadtest Dataset (Rodby Highway)

## 7.1 Purpose of the Test Phase

This document describes the **test phase** performed on the *Rodby highway roadtest dataset*.  
In contrast to validation, the test phase is conducted on **unlabelled, in-the-wild data**, with the objective of assessing:

- Model robustness under real operational conditions
- Long-term behavior over extended recordings
- Qualitative detection consistency
- Realistic interference patterns not present in controlled datasets

The test phase is **not used to compute accuracy metrics**, but to evaluate **practical usability and behavior**.

---

## 7.2 Definition: Validation vs Test

Throughout this project, the following distinction is applied:

- **Validation**  
  Controlled evaluation using labelled data with measurable accuracy, confusion matrices, and timing metrics.

- **Test**  
  Deployment-like execution on unseen, unlabelled data, focusing on:
  - Detection plausibility
  - Temporal persistence
  - Interference patterns
  - False positive behavior

The Rodby roadtest belongs strictly to the **test** category.

---

## 7.3 Roadtest Dataset Description

### 7.3.1 Recording Setup

The roadtest dataset was collected by placing a **Septentrio receiver** in a vehicle operating on the **Rodby highway in Denmark**.

Key characteristics:

- Recording duration: **~15 days**
- Environment: open highway
- Receiver output: raw baseband (SBF)
- Multiple GNSS bands recorded
- No active jammer under experimenter control

This dataset represents **true in-the-wild conditions**, including:

- Vehicle motion
- Changing RF environments
- Infrastructure-related interference
- Receiver-internal artifacts

---

### 7.3.2 Data Characteristics

Unlike Jammertest data:

- No ground truth labels are available
- Interference events are sporadic and uncontrolled
- Signal conditions vary continuously over time
- Long stretches of nominal operation are present

This makes the dataset unsuitable for validation, but ideal for **stress-testing detection pipelines**.

---

## 7.4 Test Pipeline Overview

The test pipeline follows the same architectural structure as the validation pipeline, with key differences:

1. Sequential block-wise inference over full recordings
2. No label comparison or accuracy computation
3. Logging of:
   - Timestamp
   - Frequency band
   - Predicted class
   - Model confidence (where available)
4. Optional generation of diagnostic plots for selected events

The pipeline is designed to scale to **multi-day recordings**.

---

## 7.5 Model Used for Testing

The test phase uses the **retrained deep learning spectrogram model**, selected for:

- Superior computational efficiency
- Robust WB detection after retraining
- Suitability for long-duration scanning

Feature-based models were not used for full roadtest scanning due to their higher computational cost.

---

## 7.6 Detection Behavior Observed

### 7.6.1 Chirp Interference

- Chirp detections appear as **isolated, short-duration events**
- Often correlated with transient RF activity
- Rarely persistent over long time intervals

This behavior is consistent with expected real-world sources.

---

### 7.6.2 Narrowband Interference

A key observation from the roadtest is the presence of **persistent narrowband interference**, particularly:

- Concentrated around the **L1 band**
- Appearing as stable spectral lines
- Persisting over extended time intervals

These detections are consistent across multiple days.

---

### 7.6.3 Wideband Interference

- WB detections are rare
- When present, they tend to be short-lived
- No evidence of sustained wideband jamming was observed

This aligns with expectations for a public highway environment.

---

## 7.7 Interpretation of Narrowband Detections

The persistent narrowband interference observed around L1 is likely attributable to:

- Infrastructure-related emissions
- Vehicle electronics
- External RF sources near the road environment

Although the exact origin cannot be determined without ground truth, the detections are:

- Spectrally coherent
- Temporally stable
- Consistent across days

This strongly suggests **real interference rather than model artifacts**.

---

## 7.8 Model Robustness and False Positives

Throughout the roadtest:

- False positive rates appear low
- No systematic misclassification patterns were observed
- No runaway detection behavior occurred during long clean intervals

The model remains stable over **multi-day continuous operation**.

---

## 7.9 Computational Performance in Test Mode

The DL spectrogram model enables:

- Block-level inference in ~1.8 ms
- Continuous scanning of multi-day recordings
- Real-time or faster-than-real-time processing on commodity hardware

This confirms its suitability for deployment-like scenarios.

---

## 7.10 Limitations of the Test Phase

- No ground truth labels are available
- Quantitative accuracy cannot be computed
- Interpretation relies on signal plausibility and consistency

Despite these limitations, the test phase provides critical insight into real-world behavior.

---

## 7.11 Test Conclusions

The Rodby roadtest demonstrates that:

- The detection pipeline operates reliably under real conditions
- Retrained DL models generalize well beyond controlled datasets
- Persistent narrowband interference is detectable and stable
- No excessive false positives occur during nominal operation

These findings confirm the **practical applicability** of the proposed approach for long-term GNSS interference monitoring.
