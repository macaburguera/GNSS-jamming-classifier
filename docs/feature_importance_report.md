
# Feature Importance Analysis for GNSS Jamming Classifier (78 Features)

This report summarizes the feature importance analysis of your trained GNSS jamming classifier, using:

- **MICC** (combined Pearson correlation + mutual information) — data-level relevance.
- **Permutation importance** — model-based global importance.
- **Mean |SHAP|** — how much each feature actually contributes to the model’s predictions on the test set.

The analysis is based on the `feature_importance_results.csv` generated from your `test_features.npz` (78 features).

---

## 1. Big Picture – What the Model Cares About

Across all metrics, the model is *not* using the 78 features uniformly. Roughly:

- The **top ~10 features** account for the majority of the total mean |SHAP| importance.
- The model focuses mainly on:
  1. **Time-frequency (STFT) dynamics**.
  2. **Overall power & basic time-domain stats**.
  3. **Spectral shape / narrowband structure**.
  4. **Instantaneous frequency & envelope dynamics** (secondary).
  5. More exotic features (cyclostationary, chirp-specific, DME-specific) are **used little or almost not at all**.

In other words:

> The classifier is driven primarily by **how the spectrum evolves over time**, **how concentrated vs diffuse the frequency content is**, and **overall power level**.

---

## 2. Top-tier Features – The Ones Really Steering Decisions

### 2.1. `stft_centroid_std_Hz` – Time-Frequency Spread (Star Feature)

- **Group**: Time-frequency (STFT).
- **Importance**: #1 by SHAP, and very high MICC.
- **Definition**: Standard deviation over time of the **STFT spectral centroid** (the “center of mass” of the spectrum, frame by frame).
- **Intuition**:
  - **NoJam / GNSS-only**: spectral centroid relatively stable around the GNSS band.
  - **NB jammer**: centroid locked near the jammer tone(s), low variability.
  - **WB / Chirp**: centroid moves significantly or is broadly spread as the jammer sweeps / fills the band.

**Conclusion**: This is your single most informative feature. It captures **“how much the spectrum moves in time”**, which is exactly what separates chirps and WB from static NB or clean GNSS.

---

### 2.2. `ZCR_Q` – Zero-Crossing Rate of Quadrature

- **Group**: Time-domain / basic stats.
- **Importance**: Very high MICC and second by SHAP.
- **Definition**: Rate of sign changes in the quadrature (Q) component.
- **Why it matters**:
  - More high-frequency content or noisy phase → more zero crossings.
  - NB tones, WB noise, and structured GNSS have different ZCR patterns.
- `ZCR_I` also has high MICC, but `ZCR_Q` has clearly higher SHAP. The model finds Q’s sign-change structure particularly informative.

**Conclusion**: ZCR in Q is a key indicator of the “fine” high-frequency behavior of the signal and is actively used by the model.

---

### 2.3. `pre_rms` and `psd_power` – Overall Power Level

- **Group**: Power / energy.
- `pre_rms`: Among the highest SHAP features.
- `psd_power`: Also very high SHAP.
- **Definition**:
  - `pre_rms`: RMS of the **raw** complex IQ, before normalization.
  - `psd_power`: Integral of the raw PSD.
- **Role**:
  - Separate **NoJam / low-JSR** from **strong jammer** scenarios.
  - Provide a direct measure of how “energized” the sample is.

**Conclusion**: The classifier heavily uses overall power as a first-order indicator of jamming presence/intensity, and then refines the class using shape features.

---

### 2.4. Narrowband Structure: `spec_peak_power`, `nb_peak_salience`, `nb_peak_count`

- **Group**: Spectral shape / narrowband structure.
- `spec_peak_power`: High SHAP and good MICC.
- `nb_peak_salience`: High SHAP, MICC around 0.6.
- `nb_peak_count`: Extremely high MICC (top-2 globally), moderate SHAP.
- **Intuition**:
  - These describe **how strong and how many spectral peaks** exist:
    - NB interference → a few sharp tones → high peak power, high salience, limited count.
    - WB noise → flatter spectrum → lower salience.
    - Multi-tone NB → more peaks, structured spacing.
- **Conclusion**: This trio is central for distinguishing **NB vs WB vs NoJam**.

---

### 2.5. `strong_bins_mean` – Occupancy in the Time-Frequency Plane

- **Group**: STFT dynamics.
- **Importance**: High SHAP, good MICC.
- **Definition**: Average fraction of STFT time-frequency bins above 50% of the per-frame maximum.
  - WB → many bins above the threshold across time and frequency.
  - NB → a narrow stripe of high-energy bins.
- **Conclusion**: Encodes **how “filled” the TF plane is**, making it a very natural discriminator between **wideband noise vs narrowband tone(s)**.

---

### 2.6. `tkeo_env_mean` – Envelope Nonlinearity / Impulsiveness

- **Group**: Envelope-based features.
- **Importance**: High SHAP and MICC.
- **Definition**: Mean Teager–Kaiser energy of the magnitude envelope, normalized.
- **Role**:
  - Captures **rapid energy changes / bursts** in the envelope.
  - Distinguishes more impulsive or strongly amplitude-modulated signals from smoother ones.
- **Conclusion**: The model uses it to refine decisions for signals with **pulsed or highly modulated structure**.

---

### 2.7. `spec_gini` – Inequality of Spectral Energy

- **Group**: Spectral shape.
- **Importance**: #1 by MICC globally, and also high SHAP.
- **Definition**: Gini coefficient over the PSD, measuring how concentrated the power is in a few frequencies.
  - High Gini → energy is concentrated in a few frequencies (typical NB).
  - Low Gini → energy spread over many frequencies (typical WB or many overlapping components).
- **Conclusion**: An extremely discriminative NB vs WB feature that the model strongly exploits.

---

### 2.8. Instantaneous Frequency: `instf_kurtosis` (and `instf_std_Hz`)

- **Group**: Instantaneous frequency stats.
- `instf_kurtosis`: High MICC and strong SHAP.
- `instf_std_Hz`: High MICC, moderate SHAP.
- **Role**:
  - Characterize the **distribution** of instantaneous frequency:
    - Chirps → broader IF distribution, often with smoother tails.
    - NB → more concentrated IF.
    - Complex modulations → heavier tails.
- **Conclusion**: IF stats provide an additional viewpoint on **how frequency content moves**, complementing STFT centroid statistics.

---

### 2.9. Bandpowers in Specific Regions – `bandpower_4`, `bandpower_5`, `bandpower_6`, `bandpower_7`

- **Group**: Spectral shape (coarse).
- Several of these appear with high MICC and non-trivial SHAP.
- **Definition**: Relative power in 8 equal frequency subbands (these focus on the upper bands).
- **Role**:
  - Indicate **where** in the band the jammer/energy is concentrated.
  - They likely interact with the GNSS E5/E5b dual-miniband structure vs WB vs NB tones.
- **Conclusion**: The model pays attention to energy distribution across the band, especially in the upper-middle and high subbands.

---

### 2.10. Other Supporting Features with Non-trivial Importance

Some additional features with meaningful SHAP:

- **STFT group**:
  - `stft_centroid_zcr_per_s` – how erratically the centroid moves (sign changes of derivative).
  - `stft_centroid_absderiv_med_Hzps` – median speed of spectral centroid movement.
- **Spectral shape**:
  - `spec_entropy` – global spectral spread (more uniform vs peaky).
  - `spec_spread_Hz`, `spec_rolloff95_Hz` – overall bandwidth.
- **Time-domain / amplitude**:
  - `mag_mean`, `mag_std`, `stdQ` – help anchor overall amplitude structure.

These features are not the top-1 or top-2, but together contribute significant additional information to refine class boundaries.

---

## 3. Feature Groups – Who Is Doing the Heavy Lifting?

### 3.1. Time-Frequency (STFT) Group – **Most Used by the Model**

Key features:

- `stft_centroid_std_Hz`
- `stft_centroid_absderiv_med_Hzps`
- `stft_centroid_zcr_per_s`
- `strong_bins_mean`
- `fh_hop_rate_per_s` (smaller importance, but conceptually part of this group)

**Interpretation**:

- This group captures **“how the spectrum lives in time”**.
- Crucial for:
  - Distinguishing **Chirp** vs static NB.
  - Distinguishing **WB** (diffuse, stable spread) from structured or transient patterns.
- If you ever prune features, **this group must be preserved**.

---

### 3.2. Spectral Shape & Narrowband Structure – **Core for NB vs WB vs NoJam**

Important features:

- `spec_gini`, `spec_flatness`, `spec_entropy`, `spec_spread_Hz`, `spec_rolloff95_Hz`.
- `spec_peak_power`.
- `bandpower_2`, `bandpower_3`, `bandpower_4`, `bandpower_5`, `bandpower_6`, `bandpower_7`.
- `nb_peak_count`, `nb_peak_salience`.
- `oob_ratio`.
- `spec_kurtosis_mean` (modest importance), `spec_kurtosis_max` (almost unused).

**Role**:

- Measure **concentration vs diffusion** of power and **where** it sits in frequency.
- Distinguish:
  - **NB tones** (peaky, high Gini, high salience, limited bandpower zones).
  - **WB jammers** (flatter, broader, more uniform bandpowers).
  - **NoJam/GNSS-only** (GNSS lobe structure, specific bandpower pattern and moderate Gini).

**Conclusion**: Together with STFT dynamics, this group forms the **core discriminative backbone** of the classifier.

---

### 3.3. Basic Time-Domain / Amplitude Stats – Strong but Not Dominant

Key features:

- `pre_rms`, `psd_power`.
- `mag_mean`, `mag_std`, `stdQ`.
- `ZCR_Q` (very important), `ZCR_I` (less but still relevant).

These features provide:

- Global **power scale**.
- Rough indicators of **high-frequency content** and **noise level**.

Less important time-domain stats:

- `meanI`, `corrIQ`, `skewI`, `skewQ`:
  - Low MICC and low SHAP.
  - Signal is roughly zero-mean and circular enough that these do not robustly separate classes.

**Conclusion**: The model uses time-domain stats primarily for **power and rough “texture”**, not for fine I/Q asymmetry.

---

### 3.4. Envelope / AM Features – High MICC but Partly Redundant

Interesting/useful ones:

- `env_ac_peak` – periodicity in the envelope; decent MICC and SHAP.
- `tkeo_env_mean` – already discussed as an important feature.
- `env_gini`, `env_p95_over_p50` – inequality and tail heaviness of the envelope; moderate importance.

Near-dead envelope features:

- `cep_peak_env` – almost zero MICC and SHAP.
- `env_dom_freq_Hz` – unused.
- `env_dom_peak_norm` – unused.
- `dme_ipi_std_s` – almost no contribution.

**Interpretation**:

- Some envelope features are genuinely useful (TKEO, autocorrelation), especially for more impulsive structures.
- Others are redundant: their information is already captured by spectral / TF features or simply not very discriminative in your data.

---

### 3.5. Instantaneous Frequency & Cyclostationarity – Auxiliary, Not Central

- `instf_std_Hz`, `instf_kurtosis`:
  - Provide meaningful information about how IF behaves.
  - Moderately used by the model.
- `instf_slope_Hzps`:
  - Low MICC and low SHAP; barely used.
- Cyclo features:
  - `cyclo_2chip_corr`, `cyclo_5chip_corr` – some SHAP, but small.
  - `cyclo_chip_corr`, `cyclo_halfchip_corr` – even less.

**Conclusion**:  
IF and cyclostationarity contribute *extra* robustness, but the model relies more on simpler spectral/TF patterns than on explicit GNSS chip-period structure.

---

### 3.6. Chirp-Specific and DME-Specific Proxies – Almost Irrelevant

- **Chirp** features:
  - `chirp_slope_Hzps`, `chirp_r2`, `chirp_curvature_Hzps2`:
    - MICC small-to-moderate, SHAP tiny (especially `chirp_r2`).
- **DME** features:
  - `dme_pulse_count`, `dme_duty`, `dme_ipi_med_s`, `dme_ipi_std_s`:
    - Very low SHAP, small MICC.

**Interpretation**:

- Generic spectral and TF features are already powerful enough to recognise chirp-like and pulse-like behavior.
- The explicit chirp/DME proxies do not add much beyond what `stft_*`, `bandpower_*`, `tkeo_env_mean`, and `nb_*` already provide.

---

## 4. Clearly Weak / Near-Irrelevant Features

Features with **low MICC and near-zero SHAP** are candidates for pruning:

- Envelope-cepstral / AM frequency features:
  - `cep_peak_env`
  - `env_dom_freq_Hz`
  - `env_dom_peak_norm`
- Basic stats:
  - `corrIQ`
  - `meanI`
  - `skewI`, `skewQ`
- High-order / niche features:
  - `cumulant_c40_mag`
  - `spec_kurtosis_max`
  - `fh_hop_rate_per_s`
  - `dme_ipi_std_s`
- Chirp:
  - `chirp_r2`

These add little to no information on top of the others and can be safely considered for removal in a **reduced feature set**.

---

## 5. Practical Implications

### 5.1. Feature Set Reduction

If you want to simplify the feature set (e.g., for efficiency, robustness or interpretability):

- **Definitely keep**:
  - All **STFT time-frequency features**: `stft_centroid_*`, `strong_bins_mean`.
  - Core **spectral shape**: `spec_gini`, `spec_flatness`, `spec_entropy`,
    `spec_spread_Hz`, `spec_rolloff95_Hz`, `spec_peak_power`,
    key `bandpower_*` (especially 2–7), `nb_peak_count`, `nb_peak_salience`, `oob_ratio`.
  - Core **power / time-domain**: `pre_rms`, `psd_power`, `ZCR_I`, `ZCR_Q`, `mag_mean`, `mag_std`, `stdQ`.
  - Select **envelope & IF**: `tkeo_env_mean`, `env_ac_peak`, `instf_std_Hz`, `instf_kurtosis`.

- **Candidates to drop or de-prioritise**:
  - `cep_peak_env`, `env_dom_freq_Hz`, `env_dom_peak_norm`,
  - `corrIQ`, `meanI`, `skewI`, `skewQ`,
  - `spec_kurtosis_max`, `cumulant_c40_mag`,
  - `chirp_r2`, `dme_ipi_std_s`, and similar very low-impact features.

This would give you a compact subset that preserves almost all the model’s predictive power.

---

### 5.2. Interpretation / Paper-Level Narrative

You can honestly describe your classifier as relying mainly on:

1. **Time-frequency centroid dynamics and TF occupancy**  
   → Distinguish **Chirp/WB** from **NB/NoJam**.

2. **Spectral inequality and narrowband peak structure**  
   → Separate **NB tones** from **WB noise** and **clean GNSS**.

3. **Overall power and envelope nonlinearity**  
   → Separate **jammed vs non-jammed regimes** and detect impulsive or strongly modulated interference.

Cyclostationary, chirp-specific and DME-specific features are used only marginally, suggesting that **simpler, more generic statistics are sufficient** for this particular task and dataset.

---

### 5.3. Debugging WB vs E5/E5b Confusion

If you want to understand / mitigate confusion between **WB jamming** and **E5/E5b GNSS spectrum**:

- Focus on features that the model actually uses to differentiate these cases:
  - `spec_gini`, `spec_flatness`, `spec_spread_Hz`, `spec_entropy`.
  - `bandpower_4`, `bandpower_5`, `bandpower_6`, `bandpower_7`.
  - `stft_centroid_std_Hz`, `strong_bins_mean`.
- If synthetic WB data looks too similar to real E5/E5b in these statistics, the model will naturally confuse them. Fixing this likely requires:
  - Adjusting how synthetic WB is generated, or
  - Adding training samples that explicitly represent E5/E5b as **non-jammed** examples.

---

## 6. Summary

- The model is heavily driven by **time-frequency centroid dynamics**, **spectral inequality / NB peaks**, and **overall power**.
- A relatively small subset of features contributes the majority of predictive power.
- Many high-complexity or highly specific features (chirp proxies, DME IPIs, some cyclo stats) are effectively **unused**.
- You have a clear path to:
  - Build a **reduced feature set** (~20–30 features),
  - And a clean narrative for explaining in a report or paper **how** the classifier decides between NoJam / NB / WB / Chirp.

