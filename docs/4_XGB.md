# 4. XGB

## Introduction

This document describes the feature-based machine learning approach used in this repository, based on **XGBoost (Extreme Gradient Boosting)**. It covers the theoretical foundations of the model, the rationale for its selection in the context of GNSS interference detection, and the concrete way it has been instantiated and used in the project.

The XGBoost models constitute one of the two main detection branches implemented in the pipeline, alongside the deep learning spectrogram-based approach.

---

## XGBoost: Conceptual Overview

### Gradient Boosting in Brief

Gradient boosting is an ensemble learning technique in which a strong predictor is built as a weighted sum of weak learners, typically decision trees.

Given a dataset $\{(x_i, y_i)\}_{i=1}^N$, the model prediction is expressed as:

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F}
$$

where:
- $f_k$ are regression trees,
- $K$ is the number of boosting rounds,
- $\mathcal{F}$ is the space of possible trees.

Each new tree is trained to minimize the residual error of the ensemble built so far, using gradient-based optimization of a differentiable loss function.

---

### XGBoost Specifics

XGBoost extends classical gradient boosting by introducing:

- Second-order (Hessian-based) optimization
- Explicit regularization terms
- Shrinkage and subsampling
- Efficient handling of sparse features

The objective function minimized at each iteration can be written as:

$$
\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
$$

with the regularization term:

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2
$$

where:
- $T$ is the number of leaves in a tree,
- $w_j$ are leaf weights,
- $\gamma$ and $\lambda$ control model complexity.

---

## Rationale for Using XGBoost

XGBoost is well-suited for GNSS interference detection for several reasons:

1. **Structured, low-dimensional features**  
   The extracted signal features are explicit, heterogeneous, and interpretable.

2. **Robustness to feature scaling and correlations**  
   Decision trees do not require strict feature normalization.

3. **Strong performance with limited real data**  
   XGBoost performs well even when real labelled datasets are relatively small.

4. **Interpretability**  
   Feature importance, permutation tests, and ablation studies are straightforward.

5. **Fast inference**  
   Once features are computed, inference latency is low and predictable.

---

## Feature-Based Input

The XGBoost models operate exclusively on **engineered features** extracted from baseband blocks. These features summarize:

- Spectral shape and concentration
- Temporal behavior and variability
- Envelope statistics
- Chirp-like modulation patterns

Two feature configurations are used:

- **Full feature set (78 features)**
- **Minimal feature set (10 features)**

The detailed definition of each feature is provided in the corresponding features document.

---

## Model Variants Used in the Project

Four main XGBoost variants are implemented and evaluated:

1. Full feature model (78 features), initial training
2. Full feature model (78 features), retrained with additional real data
3. Minimal feature model (10 features)
4. Minimal feature model (10 features), retrained

This structure allows systematic evaluation of performance versus feature dimensionality.

---

## Training Configuration

Across scripts, the XGBoost classifiers are instantiated with a consistent configuration philosophy.

Typical parameters include:

- `objective = multi:softprob`  
  Multi-class probabilistic classification.

- `num_class = 4`  
  Corresponding to the interference taxonomy.

- `eval_metric = mlogloss`  
  Multi-class logarithmic loss.

- `max_depth`  
  Controls tree complexity. Moderate values are used to limit overfitting.

- `learning_rate` ($\eta$)  
  Controls the contribution of each tree.

- `n_estimators`  
  Number of boosting rounds.

- `subsample`, `colsample_bytree`  
  Introduce stochasticity to improve generalization.

Exact numerical values are kept consistent within each experiment and are stored alongside trained models in the `artifacts/` directory.

---

## Retraining Strategy

Retraining is performed to reduce domain mismatch between synthetic and real-world data.

Key principles:

- Synthetic data remains the dominant training component
- Real labelled data is introduced incrementally
- Hyperparameters are held fixed

This isolates the impact of data diversity from that of model tuning.

---

## Inference Workflow

During inference, the following steps are executed:

1. Baseband blocks are loaded from SBF recordings
2. Feature extraction is performed block-wise
3. Feature vectors $x_i$ are passed to the trained model
4. Class probabilities $P(y \mid x_i)$ and predicted labels are produced

Inference is deterministic and reproducible for a given model and dataset.

---

## Computational Considerations

The total runtime of the XGBoost pipeline is dominated by:

- Feature extraction
- Disk I/O during data loading

Model inference itself accounts for a relatively small fraction of total computational cost.

This behavior is quantified in the validation document.

---

## Strengths and Limitations

### Strengths
- High accuracy under controlled and semi-controlled conditions
- Strong interpretability through feature importance analysis
- Stable behavior with limited labelled data

### Limitations
- Feature extraction cost scales linearly with data volume
- Performance depends on the quality of engineered features
- Limited flexibility for interference patterns outside the design space

These limitations motivate the complementary deep learning approach implemented in the repository.

---

## Summary

XGBoost provides a robust, interpretable, and efficient solution for GNSS interference classification when paired with carefully designed signal features. Within this project, it serves both as a high-performance detector and as a reference baseline for comparison with deep learning models.
