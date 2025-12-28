# Feature guide — original groups (78 features)

This document follows the **original 11 feature groups** used in the feature-importance report, so you can match group-by-group discussions directly.

Notation:
- $I[n]$, $Q[n]$: in-phase and quadrature samples.
- Envelope: $|z[n]|$ for $z[n]=I[n]+jQ[n]$.
- PSD: power spectral density (two-sided).

## Group 1: Basic Time-Domain & Power Features (18)

These features are computed directly from the complex IQ samples (or their envelope). They mainly describe **overall power level**, **DC offsets**, **burstiness**, and basic signal “geometry” (e.g., whether I and Q are correlated).

### `meanI`
- **Captures:** DC offsets in I and Q.

### `meanQ`
- **Captures:** DC offsets in I and Q.

### `stdI`
- **Captures:** spread/variance in each component.

### `stdQ`
- **Captures:** spread/variance in each component.

### `corrIQ`
- **Captures:** correlation between I and Q. - **Why it helps:** ideal circular noise has low I/Q correlation; some interference or imbalance can increase correlation.

### `mag_mean`
- **Captures:** magnitude statistics after normalization.

### `mag_std`
- **Captures:** magnitude statistics after normalization.

### `ZCR_I`
- **Captures:** zero-crossing rate of the I component (sign-change rate). Noise-like signals tend to have higher ZCR; steady tones / strongly smoothed signals tend to have lower ZCR.

### `ZCR_Q`
- **Captures:** zero-crossing rate of the Q component (sign-change rate). Interference that collapses the waveform into a more regular pattern can reduce ZCR; noise-like mixtures increase it.

### `PAPR_dB`
- **Captures:** bursts/spikes relative to average. - **Expected:** pulsed/bursty interference → higher PAPR.

### `env_ac_peak`
- **What it is:** maximum non-zero-lag autocorrelation of the envelope. - **Captures:** periodicity / repeated structure.

### `env_ac_lag_s`
- **What it is:** lag (seconds) where that peak occurs. - **Captures:** estimated repetition period.

### `pre_rms`
- **What it is:** RMS of raw IQ, before normalization. - **What it captures:** absolute energy level. - **Why it helps:** strong jammers often raise overall power; also useful for detecting “NoJam” vs “something is present”.

### `psd_power`
- **What it is:** total integrated PSD power estimate. - **What it captures:** absolute power in frequency domain. - **Why it helps:** corroborates `pre_rms`; can be more stable when time-domain has spikes.

### `oob_ratio`
- **What it is:** fraction of power outside an in-band region (implementation-defined). - **What it captures:** “spectral spill” and wide occupancy. - **Expected:** higher for WB; lower for NB.

### `crest_env`
- **Captures:** spiky amplitude peaks.

### `kurt_env`
- **Captures:** heavy tails / impulsiveness.

### `spec_entropy`
- **Captures:** how spread or concentrated the PSD is. - **Expected:** WB → higher entropy; NB → lower.

## Group 2: Global Spectral Shape Features (6)

These summarize the **global shape** of the two-sided power spectral density (PSD): where the energy sits, how wide it is, and whether the spectrum looks tone-like or noise-like.

### `spec_centroid_Hz`
- **Expected:** a jammer offset from DC shifts centroid.

### `spec_spread_Hz`
- **Expected:** WB → larger spread; NB → smaller spread.

### `spec_flatness`
- **Captures:** tone-like vs noise-like. - **Expected:** NB tones → low flatness; WB noise → higher.

### `spec_rolloff95_Hz`
- **Captures:** frequency such that 95% of energy lies below (in magnitude ordering sense). - **Expected:** WB → larger rolloff.

### `spec_peak_freq_Hz`
- **Captures:** dominant frequency location and strength. - **Expected:** NB tones → strong peaks; WB noise → weaker peaks.

### `spec_peak_power`
- **Captures:** dominant frequency location and strength. - **Expected:** NB tones → strong peaks; WB noise → weaker peaks.

## Group 3: Band Power Distribution (8)

A coarse “histogram” of spectral energy: the PSD is split into 8 equal-width frequency bands over $[-f_s/2, f_s/2]$, and we integrate the (normalized) PSD in each band.

**Band indexing note:** `bandpower_0` is the lowest-frequency band (most negative frequencies); `bandpower_7` is the highest-frequency band.

### `bandpower_0`
- **What it is:** total power in each of 8 equal-width frequency bands across $[-f_s/2, +f_s/2]$. - **What it captures:** coarse spectral occupancy pattern. - **Expected:**

### `bandpower_1`
- **Captures:** relative PSD energy in sub-band 1 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_2`
- **Captures:** relative PSD energy in sub-band 2 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_3`
- **Captures:** relative PSD energy in sub-band 3 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_4`
- **Captures:** relative PSD energy in sub-band 4 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_5`
- **Captures:** relative PSD energy in sub-band 5 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_6`
- **Captures:** relative PSD energy in sub-band 6 of 8 equal-width frequency bands across the Nyquist range.

### `bandpower_7`
- **What it is:** total power in each of 8 equal-width frequency bands across $[-f_s/2, +f_s/2]$. - **What it captures:** coarse spectral occupancy pattern. - **Expected:**

## Group 4: Instantaneous Frequency Features (5)

These features approximate the **instantaneous frequency (IF)** of the signal and quantify what that IF does over time: average, spread, slope, and how irregular/jumpy it is.

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

## Group 5: Envelope, Cepstrum, Pulse & Narrowband Salience (4)

These focus on **envelope structure** and **pulse-like behavior**. They are meant to react to short bursts, periodic pulses (DME-like), and a strong narrowband component visible through envelope/cepstral cues.

### `cep_peak_env`
- **Captures:** strong periodicity in envelope; useful for pulsed or AM-like jammers.

### `dme_pulse_count`
- **Captures:** number of detected pulse-like events (proxy).

### `dme_duty`
- **Captures:** fraction of time the signal is in “pulse on” state.

### `nb_peak_salience`
- **Captures:** whether a few peaks dominate the PSD strongly. - **Expected:** high for NB, lower for WB.

## Group 6: Narrowband Peaks, AM & Chirp Features (8)

Dedicated detectors for **narrowband** and **chirp-like** interference: peak counting/spacing in the PSD, envelope AM metrics, and simple chirp slope / goodness-of-fit summaries.

### `nb_peak_count`
- **Captures:** number of prominent narrowband peaks detected in the PSD (proxy for single-tone vs multi-tone narrowband interference).

### `nb_spacing_med_Hz`
- **Captures:** median spacing (Hz) between detected narrowband peaks; helps identify multi-tone comb-like structures.

### `nb_spacing_std_Hz`
- **Captures:** variability of spacing (Hz) between detected narrowband peaks; distinguishes regular combs (low std) from irregular multi-tone patterns (high std).

### `env_mod_index`
- **Captures:** amplitude modulation depth (how strongly the envelope varies relative to its mean).

### `env_dom_freq_Hz`
- **Captures:** dominant modulation frequency.

### `env_dom_peak_norm`
- **Captures:** strength of that modulation peak (normalized).

### `chirp_slope_Hzps`
- **Captures:** estimated sweep rate (Hz/s) from time–frequency behavior.

### `chirp_r2`
- **Captures:** how well a line fits the sweep (chirp-like linearity).

## Group 7: Cyclostationarity, Cumulants, Spectral Kurtosis, TKEO (7)

Higher-order structure detectors: cyclostationarity proxies, 4th-order cumulants, spectral kurtosis summaries, and the Teager–Kaiser energy operator (TKEO). These aim to capture **non-Gaussianity**, **periodic structure**, and **impulsiveness** beyond second-order statistics.

### `cyclo_chip_corr`
- **Captures:** correlation at delays related to a “chip period” model (proxy). - **Useful:** structured spread-spectrum signals have cyclostationary signatures.

### `cyclo_2chip_corr`
- **Captures:** correlation at delays related to a “chip period” model (proxy). - **Useful:** structured spread-spectrum signals have cyclostationary signatures.

### `cumulant_c40_mag`
- **Captures:** 4th-order structure beyond variance; useful for non-Gaussianity and modulation structure.

### `cumulant_c42_mag`
- **Captures:** 4th-order structure beyond variance; useful for non-Gaussianity and modulation structure.

### `spec_kurtosis_mean`
- **Captures:** impulsiveness/burstiness in the spectral domain (per-frequency-bin kurtosis, summarized).

### `spec_kurtosis_max`
- **Captures:** impulsiveness/burstiness in the spectral domain (per-frequency-bin kurtosis, summarized).

### `tkeo_env_mean`
- **Captures:** rapid amplitude transitions; useful for bursts and pulses.

## Group 8: Higher-order I/Q Stats & Circularity (6)

Higher-order moments of I and Q (skewness/kurtosis) plus measures of **complex circularity** (how close the IQ cloud is to rotationally symmetric). Useful for detecting “improper” complex statistics from certain interference types or receiver impairments.

### `skewI`
- **Captures:** asymmetry in I/Q distributions.

### `skewQ`
- **Captures:** asymmetry in I/Q distributions.

### `kurtI`
- **Captures:** tail heaviness / non-Gaussianity in I/Q.

### `kurtQ`
- **Captures:** tail heaviness / non-Gaussianity in I/Q.

### `circularity_mag`
- **Captures:** improperness / departure from rotational symmetry of complex samples.

### `circularity_phase_rad`
- **Captures:** improperness / departure from rotational symmetry of complex samples.

## Group 9: Inequality, Symmetry, DC Notch & Peakiness (6)

Features that quantify **inequality**, **spectral symmetry**, **DC notch behavior**, and “peakiness”. These often separate narrowband tones (peaky/inequal) from wideband/noise-like spectra (flatter/more equal).

### `spec_gini`
- **Captures:** inequality of PSD bins. - **Expected:** NB peaks → higher inequality; WB → lower.

### `env_gini`
- **Captures:** inequality of the envelope magnitude distribution (Gini coefficient). High values suggest bursty/impulsive energy; low values suggest more uniform amplitude over time.

### `env_p95_over_p50`
- **Captures:** tail heaviness using robust percentiles. - **Useful:** more stable than raw max-based measures.

### `spec_symmetry_index`
- **Captures:** how symmetric the spectrum is around DC. - **Useful:** symmetric wideband noise tends to be more symmetric than offset narrowband tones.

### `dc_notch_ratio`
- **Captures:** how suppressed the DC region is relative to surrounding bands (implementation-defined). - **Useful:** some interference types create a notch or avoid DC.

### `spec_peakiness_ratio`
- **Captures:** dominance of top bins over the remainder. - **Expected:** NB tones → high.

## Group 10: STFT-based Time–Frequency Dynamics (5)

Short-time Fourier transform (STFT) dynamics: rather than a single PSD, these measure **how the spectrum moves over time** (centroid wandering, jumps, ridge concentration). Especially relevant for sweeping/chirp interference.

### `stft_centroid_std_Hz`
- **Captures:** how much the spectral centroid wanders over time.

### `stft_centroid_absderiv_med_Hzps`
- **Captures:** typical speed of centroid movement.

### `stft_centroid_zcr_per_s`
- **Captures:** how often centroid direction changes (back-and-forth motion).

### `fh_hop_rate_per_s`
- **Captures:** rate of large centroid jumps; proxy for frequency hopping.

### `strong_bins_mean`
- **Captures:** how concentrated “strong bins” are over time; ridge-like energy vs diffuse.

## Group 11: Extra Cyclo Lags, Chirp Curvature, DME IPIs (5)

Extra specialized cues: additional cyclostationary lag checks, a curvature term for non-linear chirps, and DME-like inter-pulse interval (IPI) statistics.

### `cyclo_halfchip_corr`
- **Captures:** additional cyclo checks at other delays; can help distinguish certain structured signals or interference artifacts.

### `cyclo_5chip_corr`
- **Captures:** additional cyclo checks at other delays; can help distinguish certain structured signals or interference artifacts.

### `chirp_curvature_Hzps2`
- **Captures:** second-order curvature of frequency trajectory; distinguishes linear chirps from more complex sweeps.

### `dme_ipi_med_s`
- **Captures:** median and variability of inter-pulse intervals. - **Useful:** stable pulse repetition yields low IPI std; irregular pulses yield higher.

### `dme_ipi_std_s`
- **Captures:** median and variability of inter-pulse intervals. - **Useful:** stable pulse repetition yields low IPI std; irregular pulses yield higher.
