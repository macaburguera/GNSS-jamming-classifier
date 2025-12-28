# Feature guide (78 features) — explained for jammer classification

This document explains the **78 engineered features** produced by `extract_features(iq, fs)` and how they relate to **jamming classes** such as **NoJam**, **Narrowband (NB)**, **Wideband (WB)**, and **Chirp / sweep-like** interference.

The intent is not only to list formulas, but to answer three practical questions:

- **What does each feature try to capture physically?**
- **Why would that help distinguish jammer types?**
- **What should be expected (qualitatively) to increase or decrease?**

A key empirical finding from the latest feature-importance study is that **time–frequency dynamics** features—especially `stft_centroid_std_Hz`—can dominate model reliance. This is consistent with the idea that many real jammers are **non-stationary**: their spectral “center of mass” moves over time.

---

## 0) Big picture: what the feature extractor is doing

The extractor starts from complex baseband IQ samples:

- Complex samples: $x[n] \in \mathbb{C}$, where $x[n] = I[n] + j\,Q[n]$
- Sampling rate: $f_s$ (Hz)

From this, it computes **two families of measurements**:

1) **Amplitude-aware (power) measurements**  
   These preserve absolute level: e.g., `pre_rms`, `psd_power`, `oob_ratio`.

2) **Shape-aware measurements after RMS normalization**  
   These focus on **shape** (spectral shape, modulation, non-Gaussianity) while reducing sensitivity to gain/SNR changes.  
   Define the RMS:
   $$
   \mathrm{RMS}(x) = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} |x[n]|^2}
   $$
   Then normalize:
   $$
   z[n] = \frac{x[n]}{\mathrm{RMS}(x) + \epsilon}
   $$

This separation matters: if a jammer appears “stronger” just because the receiver gain changed, we do not want the classifier to confuse that with a genuinely different jammer type.

---

## 1) Glossary of terms (signal processing + statistics)

This section defines the terms that appear repeatedly.

### 1.1 Envelope
The **envelope** is the instantaneous magnitude of the complex signal:
$$
\mathrm{env}[n] = |x[n]| = \sqrt{I[n]^2 + Q[n]^2}.
$$
**Intuition:** envelope describes how “strong” the signal is over time. Pulsed jammers and AM-like interference show clear envelope structure.

### 1.2 PSD (Power Spectral Density) and Welch’s method
The **PSD** estimates how signal power is distributed over frequency. Welch’s method averages periodograms of overlapping windows to reduce variance.

**Intuition:** NB vs WB is mainly a PSD question:
- NB: power concentrated in a narrow region
- WB: power spread across a wider band

### 1.3 STFT and spectrogram
The **Short-Time Fourier Transform (STFT)** computes spectra on short sliding windows. A **spectrogram** is the magnitude/PSD of the STFT over time.

**Intuition:** chirps and hopping jammers are obvious in a spectrogram:
- Chirp: ridge moving steadily across frequency
- FH: sudden jumps of energy to new frequencies
- Stationary NB: a fixed narrow ridge

### 1.4 Cepstrum / cepstral peak
The **cepstrum** is obtained by taking the Fourier transform of the log-spectrum (or log-magnitude). In this project, “cepstral peak of the envelope” is used as a proxy for **periodic structure** in the envelope.

**Intuition:** if the envelope repeats with a regular period (pulses or strong AM), the cepstrum shows a peak at the corresponding “quefrency” (time-like axis).

### 1.5 Kurtosis
**Kurtosis** measures how heavy the tails of a distribution are relative to a Gaussian.

For a zero-mean variable $u$:
$$
\mathrm{kurt}(u) = \frac{\mathbb{E}[u^4]}{\left(\mathbb{E}[u^2]\right)^2}.
$$
**Intuition:** high kurtosis suggests rare large spikes (impulsiveness). Pulsed interference, occasional bursts, or strong spectral peaks can increase kurtosis.

### 1.6 Skewness
**Skewness** measures asymmetry:
$$
\mathrm{skew}(u)=\frac{\mathbb{E}[u^3]}{\left(\mathbb{E}[u^2]\right)^{3/2}}.
$$

### 1.7 Entropy (spectral entropy)
Entropy quantifies “disorder” or “spread”:
$$
H = -\sum_k p_k \log p_k
$$
where $p_k$ is a normalized spectrum ($\sum_k p_k=1$).  
**Intuition:** flat, noise-like spectra have higher entropy; peaky NB spectra have lower entropy.

### 1.8 Flatness
**Spectral flatness** compares geometric mean to arithmetic mean:
$$
\mathrm{flatness}=\frac{\exp\left(\frac{1}{K}\sum_k \log P_k\right)}{\frac{1}{K}\sum_k P_k}.
$$
**Intuition:** flatness near 1 → noise-like; near 0 → tone/peaky.

### 1.9 ZCR (zero-crossing rate)
ZCR is the fraction of sign changes per sample (scaled to per-second when needed).  
**Intuition:** higher ZCR typically indicates faster oscillations / higher-frequency content in time domain.

### 1.10 Gini coefficient (inequality)
Gini measures concentration/inequality of a nonnegative distribution.  
**Intuition:** a spectrum where a few bins dominate has high inequality; uniform spectra have low inequality.

### 1.11 Circularity (improperness) in complex signals
Proper complex Gaussian noise has certain symmetry properties. **Circularity** measures deviation from that (e.g., I/Q imbalance or certain modulations).

---

## 2) Feature groups redesigned around “what they capture”

The original implementation lists features in 11 blocks. Here the groups are reorganized to match **interpretation** and the **observed model findings**:

1) **Level & time-amplitude shape** (power, PAPR, envelope shape)  
2) **I/Q geometry & higher-order statistics**  
3) **Global spectrum location/width** (centroid, spread, rolloff)  
4) **Spectrum concentration & distribution** (bandpowers, entropy, peakiness, inequality)  
5) **Modulation & periodic structure** (envelope ACF, cepstrum, AM)  
6) **Frequency dynamics** (instantaneous frequency + chirp proxies)  
7) **Time–frequency dynamics (STFT)** (centroid motion, hop rate)  
8) **Pulse / DME-like timing** (pulses, duty, IPIs)  
9) **Cyclostationarity & higher-order “structure detectors”** (cyclo, cumulants, spectral kurtosis, TKEO)

All 78 features are still present; only the explanations are reorganized.

---

# A) Level & time-amplitude shape (time-domain + power)

These features capture **how big** the signal is and how its amplitude behaves over time.

## A1) Absolute power and out-of-band energy

### `pre_rms`
- **What it is:** RMS of raw IQ, before normalization.
- **What it captures:** absolute energy level.
- **Why it helps:** strong jammers often raise overall power; also useful for detecting “NoJam” vs “something is present”.
- **Caution:** can be sensitive to AGC, front-end gain, and recording conditions.

### `psd_power`
- **What it is:** total integrated PSD power estimate.
- **What it captures:** absolute power in frequency domain.
- **Why it helps:** corroborates `pre_rms`; can be more stable when time-domain has spikes.

### `oob_ratio`
- **What it is:** fraction of power outside an in-band region (implementation-defined).
- **What it captures:** “spectral spill” and wide occupancy.
- **Expected:** higher for WB; lower for NB.

## A2) Peakiness and impulsiveness in time

### `PAPR_dB`
Peak-to-average power ratio:
$$
\mathrm{PAPR}= \frac{\max_n |z[n]|^2}{\frac{1}{N}\sum_n |z[n]|^2},\quad \mathrm{PAPR}_{dB}=10\log_{10}(\mathrm{PAPR}).
$$
- **Captures:** bursts/spikes relative to average.
- **Expected:** pulsed/bursty interference → higher PAPR.

### `crest_env`
Crest factor of envelope:
$$
\mathrm{crest}=\frac{\max_n \mathrm{env}[n]}{\mathrm{RMS}(\mathrm{env})}.
$$
- **Captures:** spiky amplitude peaks.

### `kurt_env`
Kurtosis of envelope.
- **Captures:** heavy tails / impulsiveness.

### `env_p95_over_p50`
- **Captures:** tail heaviness using robust percentiles.
- **Useful:** more stable than raw max-based measures.

---

# B) I/Q geometry & higher-order statistics

These features describe the complex-plane statistics of the samples and distribution shape.

## B1) First- and second-order I/Q geometry

### `meanI`, `meanQ`
- **Captures:** DC offsets in I and Q.

### `stdI`, `stdQ`
- **Captures:** spread/variance in each component.

### `corrIQ`
- **Captures:** correlation between I and Q.
- **Why it helps:** ideal circular noise has low I/Q correlation; some interference or imbalance can increase correlation.

### `mag_mean`, `mag_std`
- **Captures:** magnitude statistics after normalization.

## B2) Higher-order I/Q shape

### `skewI`, `skewQ`
- **Captures:** asymmetry in I/Q distributions.

### `kurtI`, `kurtQ`
- **Captures:** tail heaviness / non-Gaussianity in I/Q.

### `circularity_mag`, `circularity_phase_rad`
- **Captures:** improperness / departure from rotational symmetry of complex samples.

---

# C) Global spectrum location & width (PSD-based)

These features treat the spectrum like a distribution and measure its “center” and “spread”.

### `spec_centroid_Hz`
Centroid (center of mass):
$$
\mu_f = \sum_k f_k\,P_k
$$
(with $P_k$ normalized so $\sum_k P_k = 1$).
- **Expected:** a jammer offset from DC shifts centroid.

### `spec_spread_Hz`
Second moment (spread):
$$
\sigma_f = \sqrt{\sum_k (f_k-\mu_f)^2 P_k}
$$
- **Expected:** WB → larger spread; NB → smaller spread.

### `spec_rolloff95_Hz`
- **Captures:** frequency such that 95% of energy lies below (in magnitude ordering sense).
- **Expected:** WB → larger rolloff.

### `spec_peak_freq_Hz`, `spec_peak_power`
- **Captures:** dominant frequency location and strength.
- **Expected:** NB tones → strong peaks; WB noise → weaker peaks.

### `spec_flatness`
- **Captures:** tone-like vs noise-like.
- **Expected:** NB tones → low flatness; WB noise → higher.

---

# D) Spectrum distribution & concentration

This block is central for NB vs WB discrimination.

## D1) Band power histogram (8 features)

### `bandpower_0` … `bandpower_7`
- **What it is:** total power in each of 8 equal-width frequency bands across $[-f_s/2, +f_s/2]$.
- **What it captures:** coarse spectral occupancy pattern.
- **Expected:**
  - NB: one or two bins dominate
  - WB: many bins non-negligible

## D2) Concentration / inequality / disorder

### `spec_entropy`
- **Captures:** how spread or concentrated the PSD is.
- **Expected:** WB → higher entropy; NB → lower.

### `spec_gini`
- **Captures:** inequality of PSD bins.
- **Expected:** NB peaks → higher inequality; WB → lower.

### `spec_peakiness_ratio`
- **Captures:** dominance of top bins over the remainder.
- **Expected:** NB tones → high.

### `nb_peak_salience`
- **Captures:** whether a few peaks dominate the PSD strongly.
- **Expected:** high for NB, lower for WB.

### `dc_notch_ratio`
- **Captures:** how suppressed the DC region is relative to surrounding bands (implementation-defined).
- **Useful:** some interference types create a notch or avoid DC.

### `spec_symmetry_index`
- **Captures:** how symmetric the spectrum is around DC.
- **Useful:** symmetric wideband noise tends to be more symmetric than offset narrowband tones.

---

# E) Modulation & periodic structure (envelope ACF, cepstrum, AM)

These features attempt to capture regular amplitude patterns.

## E1) Envelope autocorrelation

### `env_ac_peak`
- **What it is:** maximum non-zero-lag autocorrelation of the envelope.
- **Captures:** periodicity / repeated structure.

### `env_ac_lag_s`
- **What it is:** lag (seconds) where that peak occurs.
- **Captures:** estimated repetition period.

## E2) Cepstral envelope periodicity

### `cep_peak_env`
- **Captures:** strong periodicity in envelope; useful for pulsed or AM-like jammers.
- **Term recap:** “cepstral peak” means a peak in the cepstrum, which indicates periodic structure.

## E3) AM-specific envelope features

### `env_mod_index`
- **Captures:** amplitude modulation depth (how strongly the envelope varies relative to its mean).

### `env_dom_freq_Hz`
- **Captures:** dominant modulation frequency.

### `env_dom_peak_norm`
- **Captures:** strength of that modulation peak (normalized).

---

# F) Frequency dynamics (instantaneous frequency + chirp proxies)

These features are aimed at chirps, sweeps, and frequency instability.

## F1) Instantaneous frequency statistics (phase derivative)

Let $\phi[n] = \arg(z[n])$. Instantaneous frequency is related to $\Delta \phi$ (after unwrapping).

### `instf_mean_Hz`
- **Captures:** average frequency offset.

### `instf_std_Hz`
- **Captures:** frequency jitter.

### `instf_slope_Hzps`
- **Captures:** linear drift rate.

### `instf_kurtosis`
- **Captures:** whether frequency occasionally “jumps” (heavy-tailed instf).

### `instf_dZCR_per_s`
- **Captures:** sign-change rate of the frequency derivative; proxy for rapid alternation.

## F2) Chirp slope and linearity

### `chirp_slope_Hzps`
- **Captures:** estimated sweep rate (Hz/s) from time–frequency behavior.

### `chirp_r2`
- **Captures:** how well a line fits the sweep (chirp-like linearity).

### `chirp_curvature_Hzps2`
- **Captures:** second-order curvature of frequency trajectory; distinguishes linear chirps from more complex sweeps.

---

# G) Time–frequency dynamics (STFT features)

These features measure “how the spectrum moves over time.” They often matter a lot in practice.

### `stft_centroid_std_Hz`
- **Captures:** how much the spectral centroid wanders over time.
- **Why it matters:** moving interferers (chirp/FH) cause large centroid variability; stationary NB causes small variability.
- **Empirical note:** this feature can dominate permutation importance because it directly captures non-stationarity.

### `stft_centroid_absderiv_med_Hzps`
- **Captures:** typical speed of centroid movement.

### `stft_centroid_zcr_per_s`
- **Captures:** how often centroid direction changes (back-and-forth motion).

### `fh_hop_rate_per_s`
- **Captures:** rate of large centroid jumps; proxy for frequency hopping.

### `strong_bins_mean`
- **Captures:** how concentrated “strong bins” are over time; ridge-like energy vs diffuse.

---

# H) Pulses and DME-like timing

These features describe pulse trains and their timing.

### `dme_pulse_count`
- **Captures:** number of detected pulse-like events (proxy).

### `dme_duty`
- **Captures:** fraction of time the signal is in “pulse on” state.

### `dme_ipi_med_s`, `dme_ipi_std_s`
- **Captures:** median and variability of inter-pulse intervals.
- **Useful:** stable pulse repetition yields low IPI std; irregular pulses yield higher.

---

# I) Cyclostationarity & higher-order structure detectors

These features try to detect structured periodic components and non-Gaussian behavior.

## I1) Cyclostationarity proxies

### `cyclo_chip_corr`, `cyclo_2chip_corr`
- **Captures:** correlation at delays related to a “chip period” model (proxy).
- **Useful:** structured spread-spectrum signals have cyclostationary signatures.

### `cyclo_halfchip_corr`, `cyclo_5chip_corr`
- **Captures:** additional cyclo checks at other delays; can help distinguish certain structured signals or interference artifacts.

## I2) Higher-order cumulants

### `cumulant_c40_mag`, `cumulant_c42_mag`
- **Captures:** 4th-order structure beyond variance; useful for non-Gaussianity and modulation structure.

## I3) Spectral kurtosis

### `spec_kurtosis_mean`, `spec_kurtosis_max`
- **Captures:** impulsiveness/burstiness in the spectral domain (per-frequency-bin kurtosis, summarized).

## I4) Teager–Kaiser energy operator (TKEO)

### `tkeo_env_mean`
TKEO (on envelope) highlights rapid energy changes:
$$
\Psi(u[n]) = u[n]^2 - u[n-1]u[n+1].
$$
- **Captures:** rapid amplitude transitions; useful for bursts and pulses.

---

## Appendix: Full feature list (78)

### Group 1 (18)
`meanI`, `meanQ`, `stdI`, `stdQ`, `corrIQ`, `mag_mean`, `mag_std`, `ZCR_I`, `ZCR_Q`, `PAPR_dB`, `env_ac_peak`, `env_ac_lag_s`, `pre_rms`, `psd_power`, `oob_ratio`, `crest_env`, `kurt_env`, `spec_entropy`

### Group 2 (6)
`spec_centroid_Hz`, `spec_spread_Hz`, `spec_flatness`, `spec_rolloff95_Hz`, `spec_peak_freq_Hz`, `spec_peak_power`

### Group 3 (8)
`bandpower_0`, `bandpower_1`, `bandpower_2`, `bandpower_3`, `bandpower_4`, `bandpower_5`, `bandpower_6`, `bandpower_7`

### Group 4 (5)
`instf_mean_Hz`, `instf_std_Hz`, `instf_slope_Hzps`, `instf_kurtosis`, `instf_dZCR_per_s`

### Group 5 (4)
`cep_peak_env`, `dme_pulse_count`, `dme_duty`, `nb_peak_salience`

### Group 6 (8)
`nb_peak_count`, `nb_spacing_med_Hz`, `nb_spacing_std_Hz`, `env_mod_index`, `env_dom_freq_Hz`, `env_dom_peak_norm`, `chirp_slope_Hzps`, `chirp_r2`

### Group 7 (7)
`cyclo_chip_corr`, `cyclo_2chip_corr`, `cumulant_c40_mag`, `cumulant_c42_mag`, `spec_kurtosis_mean`, `spec_kurtosis_max`, `tkeo_env_mean`

### Group 8 (6)
`skewI`, `skewQ`, `kurtI`, `kurtQ`, `circularity_mag`, `circularity_phase_rad`

### Group 9 (6)
`spec_gini`, `env_gini`, `env_p95_over_p50`, `spec_symmetry_index`, `dc_notch_ratio`, `spec_peakiness_ratio`

### Group 10 (5)
`stft_centroid_std_Hz`, `stft_centroid_absderiv_med_Hzps`, `stft_centroid_zcr_per_s`, `fh_hop_rate_per_s`, `strong_bins_mean`

### Group 11 (5)
`cyclo_halfchip_corr`, `cyclo_5chip_corr`, `chirp_curvature_Hzps2`, `dme_ipi_med_s`, `dme_ipi_std_s`
