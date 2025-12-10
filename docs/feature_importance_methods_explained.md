
# Feature Ranking Methods Used in the GNSS Jamming Classifier

This document explains, in a self-contained way, the different **feature ranking / importance** methods used in the analysis of the 78 features of the GNSS jamming classifier:

1. Pearson correlation (or point-biserial correlation in the classification case)  
2. Mutual information (MI)  
3. MICC: a combined score (correlation + MI)  
4. Permutation importance  
5. SHAP values (SHapley Additive exPlanations)

The goal is to make each method understandable for a reader with **basic undergraduate statistics**.

---

## 1. Why Feature Ranking?

We have a classifier that receives **78 numerical features** extracted from GNSS+jammer IQ data and outputs a **class**:

- NoJam  
- Narrowband jammer (NB)  
- Wideband jammer (WB)  
- Chirp jammer  

The questions we want to answer are:

- *Which features are most strongly related to the class labels?*  
- *Which features does the trained model actually use to make predictions?*  
- *Which features are redundant or almost irrelevant?*

To do this, we use several **complementary metrics**. Each one answers the question “importance” from a slightly different angle:

- Data-only view: correlation and mutual information.  
- Model-based view: permutation importance and SHAP values.  
- A combined, normalized score: MICC.

---

## 2. Pearson Correlation (and Point-Biserial Correlation)

### 2.1. Basic idea

**Pearson correlation** measures the **strength and direction of a linear relationship** between two variables $X$ and $Y$.

For two continuous variables, the Pearson correlation coefficient $r$ is:

$$
r = \frac{\mathrm{cov}(X, Y)}{\sigma_X \, \sigma_Y}
$$

- $\mathrm{cov}(X, Y)$: covariance between $X$ and $Y$  
- $\sigma_X$, $\sigma_Y$: standard deviations of $X$ and $Y$

The coefficient $r$ lies between $-1$ and $+1$:

- $r = +1$: perfect positive linear relationship.  
- $r = -1$: perfect negative linear relationship.  
- $r = 0$: no linear relationship (though there may still be a non-linear relationship).

In our context:

- $X_j$: values of feature $j$ across all samples.  
- $Y$: encoded class labels (for example, $0 = \text{NoJam}$, $1 = \text{NB}$, $2 = \text{WB}$, $3 = \text{Chirp}$).

For classification, this is similar to a **point-biserial** correlation when the target is binary. For multiple classes, we encode labels as integers and still compute Pearson correlation as a simple indicator.

### 2.2. How we used correlation

For each feature $j$:

1. We encode the class labels as integers ($0, 1, 2, 3$).  
2. Compute the Pearson correlation between feature $X_j$ and the label vector $Y$.  
3. Take the **absolute value** $|r_j|$, because we care about the **strength**, not the sign.

High $|r_j|$ means **strong linear association** between feature $j$ and the class encoding.

### 2.3. Advantages and limitations

**Advantages**

- Very simple and fast to compute.
- Easy to interpret: how much the feature increases or decreases with the (encoded) label.

**Limitations**

- Only captures **linear** relationships.  
- With multiple classes encoded as $0,1,2,3$, the numeric encoding is somewhat arbitrary.  
- Does not consider the model; it only looks at feature–label pairs.

For these reasons, we complement correlation with **mutual information** and model-based methods.

---

## 3. Mutual Information (MI)

### 3.1. Intuitive idea

**Mutual information** comes from information theory. It measures:

> **How much knowing one variable reduces uncertainty about another.**

Formally, the mutual information between variables $X$ and $Y$ is:

$$
I(X; Y) = H(Y) - H(Y \mid X)
$$

- $H(Y)$: entropy of $Y$ (how uncertain $Y$ is overall).  
- $H(Y \mid X)$: remaining uncertainty in $Y$ after we know $X$.

If $I(X; Y) = 0$, then $X$ gives no information about $Y$.  
If $I(X; Y)$ is large, then knowing $X$ strongly reduces uncertainty about $Y$.

### 3.2. Why mutual information is useful here

Unlike correlation, MI:

- Captures **non-linear relationships**.
- Works for both:
  - **Classification**: $Y$ is a discrete class label.  
  - **Regression** (with appropriate estimators): $Y$ continuous.

In our setting (classification):

- For each feature $X_j$, we estimate $I(X_j; Y)$.
- Higher values mean that the distribution of that feature changes a lot across classes, making it informative for classification.

### 3.3. Practical computation

We use `sklearn.feature_selection.mutual_info_classif`, which:

- Approximates mutual information between a continuous feature and a discrete target.
- It is non-parametric (does not assume linearity or Gaussian distributions).

We obtain, for each feature $j$, a value $MI_j \ge 0$.

### 3.4. Advantages and limitations

**Advantages**

- Detects **non-linear** relationships.
- Independent of how we encode the label (beyond its categories).

**Limitations**

- More computationally expensive than correlation.
- Requires density / probability estimation and can be noisy with small samples.
- Still a **feature-by-feature** measure: it does not see interactions between features.

---

## 4. MICC – Combined Correlation / MI Score

### 4.1. Motivation

Correlation and mutual information often agree on which features are important, but:

- Correlation focuses on **linear** dependence.
- MI focuses on **general (possibly non-linear) dependence**.

To get a **single ranking score** that incorporates both, we define:

$$
\text{MICC}_j = \frac{1}{2}\left( \tilde{r}_j + \tilde{MI}_j \right)
$$

where:

- $\tilde{r}_j = \dfrac{|r_j|}{\max_k |r_k|}$ is the **normalized absolute correlation** of feature $j$.  
- $\tilde{MI}_j = \dfrac{MI_j}{\max_k MI_k}$ is the **normalized MI** of feature $j$.  

So both $\tilde{r}_j$ and $\tilde{MI}_j$ lie in $[0,1]$, and so does MICC.

### 4.2. Interpretation

- MICC close to **1**:
  - The feature is among the **most correlated** with the target and also among the **highest MI** features.
- MICC close to **0**:
  - The feature has weak correlation and weak mutual information with the target.

MICC is a **data-only** measure (it does not use the trained model), but it is richer than using correlation or MI alone.

---

## 5. Permutation Importance

### 5.1. Idea

Permutation importance is a **model-based** method:

> “If we randomly shuffle the values of feature $j$, how much does the model’s performance degrade?”

Why does this work?

- When we **shuffle** feature $j$, we break any real relationship between $X_j$ and the target $Y$, while leaving the distribution of $X_j$ itself intact.
- If the model relied on $X_j$ to make good predictions, its performance (e.g., accuracy) should get worse after shuffling.

### 5.2. Algorithm

For each feature $j$:

1. Compute the model’s baseline score on the dataset (e.g., accuracy on the test set): $\text{Score}_\text{baseline}$.  
2. Randomly **permute** the values of feature $j$ across all samples, leaving all other features as they are.  
3. Compute the model’s score again on this modified data: $\text{Score}_\text{permuted for } j$.  
4. The **drop in performance** (baseline minus permuted) is a measure of how important feature $j$ is.  
5. Repeat the permutation several times and average the results (to reduce randomness).

Mathematically, the importance for feature $j$ is:

$$
\text{Imp}_j = \mathbb{E}\big[\text{Score}_\text{baseline} - \text{Score}_\text{permuted for } j\big]
$$

We do this with `sklearn.inspection.permutation_importance` using:

- Accuracy (for classification) as the scoring function.

### 5.3. Advantages and limitations

**Advantages**

- Works with **any** model that has a `predict` or `predict_proba` method (tree-based, neural nets, etc.).
- Directly tied to the model’s **predictive performance** on the chosen dataset.
- Naturally captures:
  - Non-linear relationships.
  - Interactions between features (to some extent).

**Limitations**

- Can be computationally expensive (requires multiple model evaluations).
- Results depend on:
  - The dataset used (train, validation, or test).  
  - The chosen performance metric (accuracy, F1, etc.).
- If two features are strongly correlated, shuffling one may not hurt performance much because the model can still use the other.

---

## 6. SHAP Values (SHapley Additive ExPlanations)

### 6.1. Origin: Shapley Values from Game Theory

SHAP is based on **Shapley values**, a concept from cooperative game theory.

Imagine:

- We have a “game” where several players work together to produce a **payout**.  
- We want to fairly assign to each player how much they **contributed** to the total payout.

Shapley’s idea:

- Consider **all possible orders** in which players can join the game.
- For each order, see how much the payout increases when a given player joins the coalition.
- The Shapley value of a player is the **average marginal contribution** over all possible orders.

### 6.2. SHAP for features

In our case:

- The **players** are the features $X_1, \dots, X_{78}$.  
- The **payout** is the **model prediction** (for example, the probability of WB class).  
- SHAP values tell us, for each sample and each feature, how much that feature **pushes the prediction up or down** relative to some baseline.

For one sample $x$:

$$
f(x) = f(\text{baseline}) + \sum_{j=1}^{78} \phi_j(x)
$$

- $f(x)$: model prediction for sample $x$.  
- $f(\text{baseline})$: an expected output for a “reference” input.  
- $\phi_j(x)$: SHAP value for feature $j$ on sample $x$.  

Interpretation of $\phi_j(x)$:

- If $\phi_j(x) > 0$, feature $j$ **increases** the prediction for that sample (e.g., pushes towards “jammer”).  
- If $\phi_j(x) < 0$, feature $j$ **decreases** the prediction.

### 6.3. Local vs Global importance

SHAP is **local** by design:

- For each sample $x$, it tells us the contribution of each feature.

To get a **global importance ranking**, we aggregate:

$$
\text{GlobalImportance}_j = \mathbb{E}_x \big[ |\phi_j(x)| \big]
$$

i.e., for each feature $j$, we compute the **average absolute SHAP value** across many samples.

- A large value means that, on average, feature $j$ has a strong effect on the prediction (in any direction).
- A small value means the model basically ignores that feature.

### 6.4. Practical computation

For tree-based models (e.g., XGBoost, Random Forest), we can use:

- `shap.TreeExplainer(model)`

to compute SHAP values efficiently.

In our analysis:

1. We took a sample of the test set (e.g., 500 samples).  
2. Computed SHAP values for each sample and feature.  
3. Aggregated mean absolute SHAP per feature.  
4. Used this as **SHAP-based global feature importance**.

### 6.5. Advantages and limitations

**Advantages**

- Model-specific: directly reflects how the particular trained model uses each feature.
- Local + global:
  - Per-sample explanations.
  - Global ranking from aggregated absolute values.
- Handles complex models:
  - Non-linearities, interactions, etc.

**Limitations**

- More complex to compute and understand than correlation or MI.
- For non-tree models, exact SHAP is expensive and approximations are needed.
- Interpretations must consider that SHAP uses a specific baseline and approximations of “missing” features.

---

## 7. Why Use All These Methods Together?

Each method answers a slightly different question:

1. **Correlation $|r|$**  
   - “Is there a linear relationship between this feature and the target label (as encoded)?”

2. **Mutual information (MI)**  
   - “Does this feature reduce uncertainty about the target, even in a non-linear way?”

3. **MICC (combined score)**  
   - “Is this feature consistently relevant across both correlation and MI?”  
   - A unified, normalized **data-only ranking**.

4. **Permutation importance**  
   - “If I destroy the relationship between this feature and the target in the data, how much does the model’s test performance drop?”

5. **SHAP mean absolute values**  
   - “On average, how much does this feature contribute (up or down) to the model’s prediction, over all samples?”

By comparing them, we can:

- Identify features that are **statistically related** to the labels but **ignored by the model** (high MICC, low SHAP).  
- Identify features that the **model heavily relies on**, even if they are not the top ones by correlation alone.  
- Detect **redundant** or **nearly irrelevant** features where both MICC and SHAP are very small.

---

## 8. Summary

- **Correlation** and **mutual information** give us a **model-independent** view: how strongly each feature is related to the target in the data.
- **MICC** combines them into a single normalized score, highlighting features that are consistently strong in both metrics.
- **Permutation importance** and **SHAP** give us a **model-dependent** view: how much each feature affects the predictions of the **actual trained classifier**.
- Using all these together gives a **much more robust understanding** of feature importance than any single method alone, and allows us to:
  - Justify why certain features are considered core,
  - Motivate feature pruning (removing weak or redundant features),
  - And provide a clear explanation for how the classifier makes its decisions.

