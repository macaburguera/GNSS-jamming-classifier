# 5. DL Spectrogram

## Introduction

This document describes the deep learning–based interference detection approach implemented in this repository. The model operates on **time–frequency representations (spectrograms)** derived from raw GNSS baseband recordings and uses a convolutional neural network (CNN) to classify interference types at the block level.

The document is structured to first introduce the theoretical foundations of spectrogram-based analysis, followed by a detailed explanation of the concrete design choices, parameters, and training strategy used in this project.

---

## Motivation for a Spectrogram-Based Approach

Raw baseband IQ samples contain rich information about interference phenomena, but this information is not always easily separable in the time domain. Many GNSS interference types are more naturally characterized by their **spectral structure and temporal evolution**, such as:

- Narrowband tones appearing as persistent spectral lines
- Chirp interference appearing as slanted frequency tracks
- Wideband interference appearing as broadband energy increases

Time–frequency representations make these patterns explicit and visually separable, which motivates the use of spectrograms as model inputs.

---

## Short-Time Fourier Transform (STFT)

### Definition

The spectrogram used in this project is computed via the **Short-Time Fourier Transform (STFT)**. Given a discrete-time signal $x[n]$, the STFT is defined as:

$$
X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \, w[n - m] \, e^{-j \omega n}
$$

where:
- $w[n]$ is a finite-length window function,
- $m$ indexes the time frame,
- $\omega$ is the angular frequency.

The **spectrogram** is obtained as the squared magnitude:

$$
S(m, \omega) = |X(m, \omega)|^2
$$

---

### Interpretation

The STFT trades frequency resolution for time resolution. By sliding a window across the signal, it captures how spectral content evolves over time. This is particularly well suited for interference detection, where temporal dynamics are informative.

---

## Spectrogram Parameters Used

The following parameters are used consistently across training, validation, and inference.

### FFT Length (`nfft`)

- Defines the number of frequency bins.
- Controls frequency resolution:
  
$$
\Delta f = \frac{f_s}{N_{\text{FFT}}}
$$

A fixed FFT length is used to maintain consistent input dimensions.

---

### Window Length (`win_length`)

- Determines the temporal extent of each STFT frame.
- Short windows improve time resolution but reduce frequency resolution.
- Longer windows improve frequency resolution but smear fast transients.

A moderate window length is chosen to balance chirp detectability and narrowband resolution.

---

### Hop Size (`hop_length`)

- Defines the step between consecutive windows.
- Overlapping windows are used:

$$
\text{overlap} = 1 - \frac{\text{hop}}{\text{win}}
$$

Overlap improves temporal continuity and robustness to alignment effects.

---

### Window Function

A standard tapering window (e.g. Hann) is applied to reduce spectral leakage. This is particularly important for narrowband interference, where leakage could blur spectral lines.

---

### Log-Power Scaling

Raw spectrogram magnitudes span several orders of magnitude. To stabilize learning, the following transformation is applied:

$$
S_{\text{log}} = \log(S + \epsilon)
$$

where $\epsilon$ is a small constant to avoid numerical issues.

Log-power scaling:
- Compresses dynamic range
- Emphasizes weak but structured interference
- Improves numerical conditioning for neural networks

---

### Frequency Axis Handling

The spectrogram is **FFT-shifted**, centering DC in the frequency axis. This produces symmetric representations and avoids artificial edge effects in convolutional filters.

---

### Normalization

Per-spectrogram normalization is applied (z-score):

$$
S_{\text{norm}} = \frac{S - \mu}{\sigma}
$$

This removes absolute power dependence and forces the model to focus on relative structure.

---

## Input Representation

Each baseband block is transformed into a **2D spectrogram tensor**, which is treated as a single-channel image:

- Height: frequency bins
- Width: time frames
- Channels: 1

This representation allows direct reuse of standard CNN architectures.

---

## Neural Network Architecture

The deep learning model is a convolutional neural network composed of:

- Convolutional layers for local pattern extraction
- Non-linear activations
- Pooling layers for spatial invariance
- Fully connected layers for classification

The architecture is intentionally compact to:

- Reduce inference latency
- Avoid overfitting
- Support large-scale scanning of recordings

---

## Training Strategy

### Dataset Composition

Training relies primarily on **synthetic spectrograms** generated from the synthetic interference generator. Real labelled data is introduced during retraining to reduce domain mismatch.

---

### Loss Function

A categorical cross-entropy loss is used:

$$
\mathcal{L} = - \sum_{c} y_c \log(\hat{y}_c)
$$

where:
- $y_c$ is the true class label,
- $\hat{y}_c$ is the predicted class probability.

---

### Optimization

Standard stochastic gradient-based optimization is used, with:

- Fixed learning rate schedules
- Mini-batch training
- Early stopping based on validation performance

Exact optimizer parameters are stored alongside model checkpoints.

---

## Retraining and Domain Adaptation

Retraining introduces labelled real-world spectrograms to the training set while keeping:

- Network architecture fixed
- Spectrogram parameters fixed

This ensures that improvements stem from data diversity rather than architectural changes.

---

## Inference Pipeline

During inference:

1. Baseband blocks are loaded sequentially
2. Spectrograms are computed on-the-fly using the same parameters
3. Spectrograms are normalized and passed to the CNN
4. Per-block class probabilities are produced

Inference is optimized for throughput and low latency.

---

## Computational Characteristics

The DL pipeline exhibits:

- Very low inference time per block
- Predictable computational cost
- Reduced preprocessing overhead compared to feature-based pipelines

Timing measurements confirm that spectrogram generation and inference are both efficient.

---

## Strengths and Limitations

### Strengths
- Strong generalization to unseen interference patterns
- High scalability for long recordings
- Minimal manual feature engineering

### Limitations
- Reduced interpretability compared to feature-based models
- Dependence on spectrogram parameter choices
- Sensitivity to domain mismatch if retraining is omitted

---

## Summary

The spectrogram-based deep learning approach provides a scalable and computationally efficient solution for GNSS interference detection. By carefully selecting STFT parameters and enforcing consistency across training and inference, the model effectively captures the spectral-temporal structure of common interference types while remaining suitable for large-scale real-world analysis.
