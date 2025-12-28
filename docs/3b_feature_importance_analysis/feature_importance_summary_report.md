## Summary report: methods, rationale, results, and main findings

### Context and objective
The objective is to understand, with statistical discipline, **which engineered features carry class-discriminative information** for jammer classification and **which features the trained XGBoost model actually relies on** to achieve performance on held-out data. The analysis is explicitly multi-class and focuses on balanced performance across classes.

---

## Methods used

### 1) Mutual Information (MI) and normalized Mutual Information (nMI)

**Purpose**
- Quantify **feature–label dependency** in the data, independently of any model.
- Provide a data-side audit: “Is the signal present in the dataset?”

**Definition**
- Mutual information for feature $X_j$ and label $Y$:
$$
I(X_j;Y)=\sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}.
$$
- Label entropy:
$$
H(Y)=-\sum_y p(y)\log p(y).
$$
- Normalized MI:
$$
nMI_j=\frac{I(X_j;Y)}{H(Y)}.
$$

**Implementation choices**
- MI is estimated using a kNN-based estimator (`mutual_info_classif`), which captures **nonlinear** dependencies.
- Features are standardized before MI estimation (`StandardScaler`) because kNN distance computations are scale-sensitive.
- nMI is computed on **train or train+val** (recommended) to avoid test leakage and to stabilize the estimate.

**Interpretation**
- High nMI: feature carries substantial information about jammer class.
- Low nMI: weak direct dependence (may still contribute via interactions).

---

### 2) Permutation importance using macro-F1 drop

**Purpose**
- Quantify **model reliance**: “Does the trained model need this feature to perform well on held-out data?”

**Definition**
- Let $S$ be the baseline macro-F1 score on the evaluation split.
- For a feature $X_j$, permute its column across samples to break its association with $Y$, recompute the score $S^{(perm)}_j$.
- Permutation importance (drop):
$$
\Delta_j = S - S^{(perm)}_j.
$$

**Why macro-F1**
- Macro-F1 weights classes equally and penalizes models that perform well on dominant classes while neglecting minority/harder classes.

**Interpretation**
- Large positive $\Delta_j$: performance drops strongly → feature is **necessary** for the model on this split.
- Near-zero $\Delta_j$: feature is **unused** or redundant given other features.
- Negative values may occur due to noise in finite sampling; these are typically treated as “no evidence of usefulness”.

---

### 3) Model adequacy metrics (held-out evaluation)

**Purpose**
- Validate whether the pipeline is performing acceptably and identify where errors concentrate.

**Metrics**
- Accuracy
- Balanced accuracy
- Macro-F1
- Log loss (when `predict_proba` is usable and aligned)

**Diagnostics**
- Confusion matrix (counts) and row-normalized confusion matrix
- Per-class precision/recall/F1 report
- Confidence distributions (when probabilities are available)

**Interpretation**
- Adequacy metrics provide the baseline against which “importance” is meaningful.
- Confusion structure indicates **which class boundaries** need targeted improvement.

---

## Results (high-level)

### A) Feature reliance is sharply concentrated (permutation importance)
The permutation-importance distribution is highly concentrated:

- **`stft_centroid_std_Hz`** accounts for approximately **53.5%** of the *total positive* permutation-importance mass.
- The **top ~15** features explain approximately **~90%** of the positive importance mass.
- Around **~23** features explain approximately **~95%**.

**Implication**
- The trained model behaves as if a relatively small core set of features carries most of the predictive burden.
- The remaining features form a long tail of redundancy or marginal contributions.

---

### B) Dominant feature families align with expected signal physics
The most influential features are strongly spectral/STFT-based, consistent with jammer phenomenology:

- NB/WB differences are fundamentally **spectral occupancy/bandwidth** phenomena.
- Chirp-like behavior introduces strong **time–frequency structure**, increasing variability and shifting spectral centroid-related statistics.
- NoJam is expected to maintain baseline/noise-like spectral behavior relative to jammer classes.

**Implication**
- The ranking is not arbitrary: it is consistent with plausible physical discriminants.

---

### C) Cross-method agreement: nMI vs permutation importance
Broad alignment is observed between data-side nMI and model-side permutation importance:

- Many features with high nMI also exhibit high permutation drops.
- Disagreements provide actionable interpretation:

**Common patterns**
- **High nMI + low perm drop**
  - Indicates redundancy: the feature is informative in the dataset but not uniquely required given correlated features.
- **Low nMI + high perm drop**
  - Suggests importance via interactions or proxy behavior, or underestimation by the MI estimator. These cases merit scrutiny for brittleness.

---

## Main findings (actionable conclusions)

### 1) The feature set has a clear hierarchy
- A small subset of features dominates both model reliance and explanatory mass.
- A large portion of the feature set contributes little to held-out macro-F1 when considered individually.

### 2) Feature simplification is plausible without large performance loss
Given the steep cumulative-importance curve, a reduced set (e.g., top 20–30 features by permutation importance) is likely to preserve most of the macro-F1, subject to validation on VAL and final verification on TEST.

### 3) Heavy reliance on a single feature is both strength and risk
- Strength: one feature provides strong separability on the current data distribution.
- Risk: robustness depends heavily on the stability of that feature under:
  - preprocessing changes (STFT parameter changes, normalization changes),
  - domain shifts (different RF environments, front-end characteristics),
  - dataset composition changes.

### 4) Many low-ranked features are likely redundant rather than useless
- Low permutation importance does not imply “no information”, only “not necessary given other features”.
- This supports feature-family clustering: keep representative features per family rather than many highly correlated variants.

### 5) Confusion structure should guide the next iteration
- The most valuable next improvements come from the highest-frequency/high-confidence confusion pairs (e.g., NB↔WB, Chirp↔NB, depending on the run).
- Feature engineering should focus on boundaries where the confusion matrix indicates persistent mixing, rather than on already-separated class pairs.

---

## Recommendations for the next cycle (minimal and rigorous)

- Use **nMI computed on train+val** for stability and to avoid leakage.
- Use **permutation importance on val** while iterating; reserve **test** for final reporting.
- Run an ablation experiment:
  - train/evaluate with top $N$ features by permutation drop for $N\in\{10,15,20,30\}$,
  - compare macro-F1 and the confusion matrix to detect boundary-specific degradation.
- Treat `stft_centroid_std_Hz` as a **critical dependency** and verify its stability across:
  - different STFT parameter settings,
  - different recordings/devices (if applicable),
  - different SNR conditions.

If the top confusion pairs from `misclassification_pairs_test.csv` are provided, targeted feature suggestions can be derived for the specific boundaries that remain hardest.
