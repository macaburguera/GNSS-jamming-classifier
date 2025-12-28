# Feature Documentation for `extract_features(iq, fs)`

This document describes **all 78 features** produced by `extract_features(iq, fs)`.

Goal:  
Starting from a complex GNSS+jammer IQ vector, we turn it into a **single feature vector** that describes:

- How strong the signal is (power),
- How its spectrum looks (flat vs peaky, in-band vs out-of-band),
- How its **frequency evolves in time** (chirps, hops, jitter),
- How the **amplitude behaves** (AM, pulses, DME-like),
- How **GNSS-like** the signal is (chip-period cyclostationarity),
- How **non-Gaussian / non-circular** it is.

The idea is that a classifier (RF, XGBoost, MLP, …) can work on this feature vector instead of raw IQ.

Each feature below has:

- **Intuition**: what it tells you, in undergrad language.
- **Formula**: how it is computed.

---

## 0. Common Definitions

Let

- $x[n] \in \mathbb{C}$, $n = 0,\dots,N-1$ be the **raw IQ samples** (`iq`).
- Raw envelope: $\text{env}_\text{raw}[n] = |x[n]|$.
- Sampling frequency: $f_s$ (Hz).
- Small constant: $\varepsilon = 10^{-20}$ to avoid division by zero and $\log(0)$.

### 0.1. RMS and normalization

We first compute the raw RMS power and normalize the signal:

- **Raw RMS power**

$$
  \text{pre} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2}.
$$

- **Normalized complex samples**

$$
  z[n] =
    \begin{cases}
      \dfrac{x[n]}{\text{pre}}, & \text{if } \text{pre} > 0 \\
      x[n], & \text{otherwise}
    \end{cases}
$$

- **Normalized I/Q and magnitude**

$$
  I[n] = \Re\{z[n]\}, \quad Q[n] = \Im\{z[n]\}, \quad \text{mag}[n] = |z[n]|.
$$

Most “shape” features use $z[n]$ so they don’t depend on absolute power.

### 0.2. Zero-crossing rate (ZCR)

For a real sequence $x[n]$,

- Code: `zero_crossing_rate(x) = mean(abs(diff(signbit(x))))`.
- Approximate definition:

$$
  \text{ZCR}(x) \approx
  \frac{\left|\{n: x[n]\;\text{and}\;x[n+1]\;\text{have opposite signs}\}\right|}{N-1}.
$$

**Intuition**: high ZCR → the signal oscillates fast (more “high-frequency stuff” in time domain).

### 0.3. Welch PSD

We use SciPy’s Welch method to estimate a power spectral density (PSD):

$$
\text{WelchPSD}(x) \Rightarrow (f_k, P_{xx}[k]),\quad k=0,\dots,K-1
$$

with Hann window, `nperseg = WELCH_NPERSEG`, `noverlap = WELCH_OVERLAP`, `return_onesided=False`, `scaling="density"`.

We often normalise to get a probability distribution over frequency:

$$
P^\text{norm}_{xx}[k] = \frac{\max(P_{xx}[k], \varepsilon)}{\sum_j \max(P_{xx}[j], \varepsilon)}.
$$

---

## 1. Basic Time-Domain & Power Features (18)

**Category intuition**

These describe **simple statistics in time**:

- DC offsets and spread of I/Q,
- How often the signal crosses zero,
- Peak-to-average ratio and envelope periodicity,
- Overall power, out-of-band power, and “spikiness” of the amplitude.

They are cheap to compute and give a quick fingerprint of the signal.

---

### 1.1 I/Q means and spreads

1. **`meanI`**

   - **Intuition**: Average value of I; if not ≈0, the I channel has a DC offset.
   - **Formula**:

$$
     \text{meanI} = \frac{1}{N} \sum_{n=0}^{N-1} I[n].
$$

2. **`meanQ`**

   - **Intuition**: Same as above but for Q.
   - **Formula**:

$$
     \text{meanQ} = \frac{1}{N} \sum_{n=0}^{N-1} Q[n].
$$

3. **`stdI`**

   - **Intuition**: How much I varies around its mean (spread / energy in I).
   - **Formula**:

$$
     \text{stdI} = \sqrt{\frac{1}{N} \sum_n \big(I[n] - \text{meanI}\big)^2}.
$$

4. **`stdQ`**

   - **Intuition**: Same idea, but for Q.
   - **Formula**:

$$
     \text{stdQ} = \sqrt{\frac{1}{N} \sum_n \big(Q[n] - \text{meanQ}\big)^2}.
$$

5. **`corrIQ`**

   - **Intuition**: How linearly related I and Q are.  
     - GNSS-like proper noise → low correlation.  
     - Certain modulations → strong correlation.
   - **Formula**:

$$
     \text{corrIQ} =
       \frac{\sum_n (I[n] - \text{meanI})(Q[n] - \text{meanQ})}{
             \sqrt{\sum_n (I[n] - \text{meanI})^2}\;
             \sqrt{\sum_n (Q[n] - \text{meanQ})^2} }.
$$

6. **`mag_mean`**

   - **Intuition**: Average normalized magnitude; around 1 for sane signals because of the normalization.
   - **Formula**:

$$
     \text{mag} = \frac{1}{N} \sum_n |z[n]|.
$$

7. **`mag_std`**

   - **Intuition**: How much the magnitude fluctuates.  
     - Constant amplitude carrier → low.  
     - Pulsed or heavily AM signal → higher.
   - **Formula**:

$$
     \text{mag} = \sqrt{\frac{1}{N} \sum_n (|z[n]| - \text{mag})^2}.
$$

### 1.2 Zero crossings and PAPR

8. **`ZCR_I`**

   - **Intuition**: How rapidly I changes sign → higher for high-frequency content.
   - **Formula**:

$$
     \text{ZCR} = \text{ZCR}(I[n]).
$$

9. **`ZCR_Q`**

   - **Intuition**: Same for Q.
   - **Formula**:

$$
     \text{ZCR} = \text{ZCR}(Q[n]).
$$

10. **`PAPR_dB`**

    - **Intuition**: Measures how “peaky” the amplitude is.  
      - OFDM-like or pulsed signals → high PAPR.  
      - Smooth constant-envelope signals → low PAPR.
    - **Formula**:

$$
      \text{PAPR} =
        20 \log_{10}
        \left(
          \frac{\max_n |z[n]| + \varepsilon}{\frac{1}{N} \sum_n |z[n]| + \varepsilon}
        \right).
$$

### 1.3 Envelope autocorrelation

Let $\text{env}[n] = |z[n]|$ and $\tilde{\text{env}}[n] = \text{env}[n] - \overline{\text{env}}$.

We compute its autocorrelation efficiently using FFT:

$$
\text{AC}[k] = \text{IFFT}\big(|\text{FFT}(\tilde{\text{env}})|^2\big),
$$

then normalise by $\text{AC}[0]$:

$$
r[k] = \frac{\text{AC}[k]}{\text{AC}[0]}.
$$

We only look at lags up to $k_\text{max} \approx \text{MAX LAG} \cdot f_s$.

11. **`env_ac_peak`**

   - **Intuition**: Strength of the most pronounced *periodicity* in the envelope (excluding lag 0).  
     Useful for repeatedly pulsed or AM signals.
   - **Formula**:

$$
     \text{env ac} = \max_{1 \le k \le k_\text{max}} r[k].
$$

12. **`env_ac_lag_s`**

   - **Intuition**: Time between those repeating patterns (period where the peak happens).
   - **Formula**:

$$
     k^* = \arg\max_{1 \le k \le k_\text{max}} r[k],\quad
     \text{env ac lag} = \frac{k^*}{f_s}.
$$

### 1.4 Raw power, out-of-band, crest, kurtosis, entropy

13. **`pre_rms`**

   - **Intuition**: Absolute power of the raw IQ chunk, as seen by the receiver.
   - **Formula**:

$$
     \text{pre} = \sqrt{\frac{1}{N} \sum_n |x[n]|^2}.
$$

14. **`psd_power`**

   - **Intuition**: Total energy in the PSD estimate (basically the same information as `pre_rms` but in the frequency domain).
   - **Formula**:

$$
     \text{psd} = \sum_k P_{xx0}[k],
$$

     where $P_{xx0}$ is the Welch PSD of $x[n]$.

15. **`oob_ratio`**

   - **Intuition**: How much power lives **outside** the “interesting” in-band GNSS window (e.g. wideband jammer).
   - **Formula** (with in-band half-bandwidth $B=\text{INB BW}$):

$$
     \mathcal{I} = \{k : |f_k| \le B\}, \quad
     \mathcal{O} = \{k : |f_k| > B\},
$$

$$
     \text{oob} =
       \frac{\sum_{k \in \mathcal{O}} P_{xx0}[k]}{\sum_{k \in \mathcal{I}} P_{xx0}[k] + \varepsilon}.
$$

16. **`crest_env`**

   - **Intuition**: Another impulsiveness metric: how tall the biggest amplitude peak is compared to the average amplitude.
   - **Formula**:

$$
     \text{crest} =
       \frac{\max_n \text{env}_\text{raw}[n]}{\frac{1}{N} \sum_n \text{env}_\text{raw}[n] + \varepsilon}.
$$

17. **`kurt_env`**

   - **Intuition**: How heavy-tailed the amplitude distribution is.  
     - Gaussianish noise → kurtosis ≈ 3.  
     - Occasional huge pulses → much larger.
   - **Formula** (population kurtosis):

$$
     \text{kurt} =
       \frac{\mathbb{E}\left[(\text{env}_\text{raw} - \mu)^4\right]}{\big(\mathbb{E}[(\text{env}_\text{raw} - \mu)^2]\big)^2},
$$

     where $\mu = \mathbb{E}[\text{env}_\text{raw}]$ (with safe defaults for short / constant sequences).

18. **`spec_entropy`**

   - **Intuition**: How “spread” vs “concentrated” the raw spectrum is.  
     - White noise → high entropy.  
     - One or a few strong tones → lower entropy.
   - **Formula** using normalised PSD of $x[n]$:

$$
     P^\text{prob}_{xx0}[k] =
       \frac{\max(P_{xx0}[k], \varepsilon)}{\sum_j \max(P_{xx0}[j], \varepsilon)},
$$

$$
     \text{spec} = -\sum_k P^\text{prob}_{xx0}[k] \,\log\big(P^\text{prob}_{xx0}[k]\big).
$$

---

## 2. Global Spectral Shape Features (6)

**Category intuition**

These describe the **overall shape of the spectrum** of the normalized signal:

- Where it is centered,
- How wide it is,
- How flat vs peaky it is,
- Where the main peak is.

All use the normalized PSD of $z[n]$.

---

Let $(f_k, P_{xx}[k])$ be Welch PSD of $z[n]$ and

$$
P^\text{norm}_{xx}[k] = \frac{\max(P_{xx}[k], \varepsilon)}{\sum_j \max(P_{xx}[j], \varepsilon)}.
$$

19. **`spec_centroid_Hz`**

   - **Intuition**: “Center of mass” of the spectrum; indicates which side (positive/negative) holds more energy.
   - **Formula**:

$$
     \text{spec centroid} =
       \sum_k f_k \, P^\text{norm}_{xx}[k].
$$

20. **`spec_spread_Hz`**

   - **Intuition**: Effective bandwidth: how far the energy spreads around the centroid.
   - **Formula**:

$$
     \text{spec spread} =
       \sqrt{
         \sum_k (f_k - \text{spec centroid})^2 \, P^\text{norm}_{xx}[k]
       }.
$$

21. **`spec_flatness`**

   - **Intuition**: 1 for perfectly flat spectrum, near 0 for spectra with strong peaks.  
     Good to distinguish narrowband tones from wideband noise.
   - **Formula** (Wiener flatness):

$$
     \text{spec} =
       \frac{
         \exp\left( \frac{1}{K} \sum_k \ln(P^\text{norm}_{xx}[k]) \right)
       }{
         \frac{1}{K} \sum_k P^\text{norm}_{xx}[k]
       }.
$$

22. **`spec_rolloff95_Hz`**

   - **Intuition**: Frequency such that 95% of total spectral energy lies below it → “edge” of effective band.
   - **Formula**:

$$
     C[m] = \sum_{k \le m} P^\text{norm}_{xx}[k],
$$

     find smallest $m$ with $C[m] \ge 0.95$, then

$$
     \text{spec rolloff95} = f_m.
$$

23. **`spec_peak_freq_Hz`**

   - **Intuition**: Frequency of the strongest spectral component (e.g. a CW jammer).
   - **Formula**:

$$
     k^* = \arg\max_k P^\text{norm}_{xx}[k], \quad
     \text{spec peak freq} = f_{k^*}.
$$

24. **`spec_peak_power`**

   - **Intuition**: How strong that main peak is relative to total power (because PSD is normalized).
   - **Formula**:

$$
     \text{spec peak} = P^\text{norm}_{xx}[k^*].
$$

---

## 3. Band Power Distribution (8)

**Category intuition**

Here we split the whole band into **8 equal slices** from $-f_s/2$ to $+f_s/2$ and measure how much normalized power lies in each. This is a coarse “spectral histogram”.

---

Edges:

$$
e_i = -\frac{f_s}{2} + i\cdot\frac{f_s}{8}, \quad i=0,\dots,8.
$$

For band $i$:

$$
\mathcal{B}_i = \{k : e_i \le f_k < e_{i+1}\},
$$

$$
B_i = \sum_{k \in \mathcal{B}_i} P^\text{norm}_{xx}[k],\quad
\text{bandpower}_i = \frac{B_i}{\sum_{j=0}^7 B_j + \varepsilon}.
$$

Features:

25–32. **`bandpower_0` … `bandpower_7`**

- **Intuition**: Fraction of energy in each of the 8 sub-bands.  
  Together they roughly sum to 1 and describe where the spectrum lives.
- **Formula**: as above.

---

## 4. Instantaneous Frequency Features (5)

**Category intuition**

These look at the **instantaneous frequency over time** (from the IQ phase). They are good for detecting:

- Frequency drift and chirps,
- Jitter in the carrier,
- Rapid frequency fluctuations (possibly FH-like behaviour).

---

We first compute:

- Unwrapped phase:

$$
  \phi[n] = \text{unwrap}(\arg z[n]).
$$

- Instantaneous frequency:

$$
  f_\text{inst}[k] = \frac{f_s}{2\pi}\,(\phi[k+1] - \phi[k]),\quad k=0,\dots,N-2,
$$

then clip out extreme percentiles for robustness.

33. **`instf_mean_Hz`**

- **Intuition**: Average carrier offset of the chunk. Non-zero means the centre frequency is shifted.
- **Formula** (with $M=N-1$):

$$
  \text{instf mean} = \frac{1}{M} \sum_{k=0}^{M-1} f_\text{inst}[k].
$$

34. **`instf_std_Hz`**

- **Intuition**: How much the instantaneous frequency wiggles around its mean (frequency jitter).
- **Formula**:

$$
  \text{instf std} = \sqrt{\frac{1}{M} \sum_k (f_\text{inst}[k] - \text{instf mean})^2}.
$$

35. **`instf_slope_Hzps`**

- **Intuition**: Linear trend of frequency vs time, i.e. chirp slope.  
  - Positive → frequency ramps up.  
  - Negative → ramps down.  
  - Near zero → stationary carrier.
- **Formula**: least-squares fit $f_\text{inst}[k] \approx a t_k + b$ with $t_k = k/f_s$:

$$
  \text{instf slope} = a.
$$

36. **`instf_kurtosis`**

- **Intuition**: Whether the inst. frequency has occasional big jumps (heavy tails) vs more Gaussian noise.
- **Formula** (population kurtosis):

$$
  \text{instf} =
    \frac{\mathbb{E}\big[(f_\text{inst} - \mu_f)^4\big]}
         {\big(\mathbb{E}[(f_\text{inst}-\mu_f)^2]\big)^2}.
$$

37. **`instf_dZCR_per_s`**

- **Intuition**: How often the *change* in inst. frequency flips sign per second → how “zig-zaggy” the frequency evolution is.
- **Formula**:

$$
  d_f[k] = f_\text{inst}[k+1] - f_\text{inst}[k],
$$

$$
  \text{instf dZCR per} = \text{ZCR}(d_f)\cdot f_s.
$$

---

## 5. Envelope, Cepstrum, Pulse & Narrowband Salience (4)

**Category intuition**

These features look at the **amplitude envelope** and at how much the spectrum is dominated by a few peaks. They catch things like:

- Periodic amplitude modulation (AM),
- DME-style pulses,
- Tones that dominate the spectrum.

---

38. **`cep_peak_env`**

- **Intuition**: Strength of a **periodic pattern** in the envelope (e.g. regularly spaced pulses) in a given quefrency range (here ≈0.2–5 ms).
- **Formula** (simplified):

  - Envelope $e[n] = |z[n]| - \overline{|z|}$, window $w[n]$.
  - Spectrum:

$$
    S[k] = \text{FFT}(e[n]w[n]).
$$

  - Log magnitude:

$$
    L[k] = \log(|S[k]| + \varepsilon).
$$

  - Real cepstrum:

$$
    c[q] = \text{IFFT}(L[k]).
$$

  - With quefrency $q/f_s$ in $[q_\min,q_\max]$ (e.g. $[2\cdot10^{-4},5\cdot10^{-3}]$ s):

$$
    \text{cep peak} = \max_{q \in [q_\min,q_\max]} c[q].
$$

39. **`dme_pulse_count`**

- **Intuition**: Rough count of strong pulses in the envelope (designed with DME-like bursts in mind).
- **Formula idea**:

  - Smooth $\text{env}_\text{raw}$ with moving average of length $\approx 0.5\,\mu$s to get $\text{env}_s[n]$.
  - Threshold $T = \mathbb{E}[\text{env}_s] + 3 \cdot \text{std}(\text{env}_s)$.
  - Boolean above threshold: $a[n] = 1$ if $\text{env}_s[n]>T$, else 0.
  - Rising edges $r[n] = 1$ when $a[n]=1$ and $a[n-1]=0$.

$$
  \text{dme pulse} = \sum_n r[n].
$$

40. **`dme_duty`**

- **Intuition**: Fraction of time where the smoothed envelope is “high” → how busy the pulsed interference is.
- **Formula**:

$$
  \text{dme} = \frac{1}{N}\sum_n a[n].
$$

41. **`nb_peak_salience`**

- **Intuition**: How much of the spectral energy is concentrated in the **top 5 peaks** vs the rest.  
  - Large value → strong tones.  
  - Small value → more spread / noise-like.
- **Formula**: using normalized PSD $P^\text{norm}_{xx}[k]$,

  - Let $\mathcal{T}$ be indices of 5 largest bins.
  - Top power: $P_\text{top} = \sum_{k\in \mathcal{T}} P^\text{norm}_{xx}[k]$.
  - Remaining: $P_\text{rest} = 1 - P_\text{top}$.

$$
  \text{nb peak} = \frac{P_\text{top}}{P_\text{rest} + \varepsilon}.
$$

---

## 6. Narrowband Peaks, AM & Chirp Features (8)

**Category intuition**

This group refines the view on **narrowband tones**, **AM behaviour**, and **chirp-like sweeps**:

- How many peaks,
- How regularly spaced they are,
- How strongly the amplitude is modulated,
- Whether the signal behaves like a clean chirp.

---

### 6.1. Narrowband peaks and spacing

42. **`nb_peak_count`**

- **Intuition**: Number of significant spectral peaks above a prominence threshold.  
  Multi-tone jammers → more peaks; single CW → 1 strong peak.
- **Formula**:

  - $P_{xx}[k]$: PSD (unnormalised) on $z$.
  - $P_\max = \max_k P_{xx}[k]$, threshold $\text{prom} = 0.03 P_\max$.
  - Use `find_peaks` (SciPy) with this prominence to get peak index set $\mathcal{P}$.

$$
  \text{nb peak} = |\mathcal{P}|.
$$

43. **`nb_spacing_med_Hz`**

44. **`nb_spacing_std_Hz`**

- **Intuition**: If there are several peaks, these metrics tell you how **regularly spaced** they are in frequency.  
  - E.g. a comb of tones → fairly constant spacing.
- **Formula** (if $|\mathcal{P}|\ge2$):

  - Peak frequencies $f_i$ (sorted),
  - Spacings $s_j = f_{j+1} - f_j$.

$$
  \text{nb spacing med} = \text{median}(s_j),
  \quad
  \text{nb spacing std} = \text{std}(s_j).
$$

  If fewer than 2 peaks, both are 0.

### 6.2. AM envelope features

Let $\text{env}[n]=|z[n]|$, mean $\mu$, zero-mean $e[n]=\text{env}[n]-\mu$.

45. **`env_mod_index`**

- **Intuition**: How strongly the amplitude is modulated.  
  - Constant envelope → near 0.  
  - Strong AM → larger.
- **Formula**:

$$
  \text{env mod} =
    \frac{\mathbb{E}[(\text{env} - \mu)^2]}{\mu^2 + \varepsilon}.
$$

46. **`env_dom_freq_Hz`**

47. **`env_dom_peak_norm`**

- **Intuition**:
  - `env_dom_freq_Hz` → dominant **modulation frequency** of the envelope (in 30–7000 Hz band).  
  - `env_dom_peak_norm` → how dominant that modulation is relative to all others.
- **Formula**:

  - FFT: $E[k] = \text{FFT}(e[n]w[n])$, envelope power $P_e[k] = |E[k]|^2$.
  - Frequencies $f^{(e)}_k$.
  - Band mask $\mathcal{B}_e = \{k : f_\min \le f^{(e)}_k \le f_\max\}$.

$$
  k^* = \arg\max_{k\in \mathcal{B}_e} P_e[k],\quad
  \text{env dom freq} = f^{(e)}_{k^*},
$$

$$
  \text{env dom peak} =
    \frac{P_e[k^*]}{\sum_{k\in \mathcal{B}_e} P_e[k] + \varepsilon}.
$$

### 6.3. Chirp slope and linearity

We split $z[n]$ into $S$ equal segments and per segment compute the spectral centroid $c_s$ and its center time $t_s$.

48. **`chirp_slope_Hzps`**

- **Intuition**: Average frequency change rate across the whole chunk, estimated from these centroids.  
  Basically another chirp slope (complementing `instf_slope_Hzps`).
- **Formula**: Fit $c_s \approx a t_s + b$ by least squares and take

$$
  \text{chirp slope} = a.
$$

49. **`chirp_r2`**

- **Intuition**: How well a **linear** model explains the centroid evolution.  
  - Close to 1 → clean linear chirp.  
  - Small → behaviour is messy or non-chirp.
- **Formula**:

$$
  SS_\text{res} = \sum_s (c_s - \hat{c}_s)^2,\quad \hat{c}_s = a t_s + b,
$$

$$
  SS_\text{tot} = \sum_s (c_s - \overline{c})^2 + \varepsilon,
$$

$$
  \text{chirp} = 1 - \frac{SS_\text{res}}{SS_\text{tot}}.
$$

---

## 7. Cyclostationarity, Cumulants, Spectral Kurtosis, TKEO (7)

**Category intuition**

GNSS signals are **cyclostationary** at chip rate and have specific higher-order statistics. This category tries to capture:

- GNSS-like chip periodicity,
- Modulation type / non-Gaussianity,
- Time-frequency “burstiness”,
- Rapid energy changes.

---

### 7.1. Cyclostationary proxies

We define:

$$
\text{cyclo lag}(z,L) =
  \frac{\left|\sum_{n=0}^{N-L-1} z[n+L]\; \overline{z[n]}\right|}
       {\sqrt{\left(\sum_{n} |z[n+L]|^2\right)\left(\sum_{n} |z[n]|^2\right)} + \varepsilon}.
$$

50. **`cyclo_chip_corr`**

- **Intuition**: Correlation between the signal and a copy shifted by **1 chip period** (approx).  
  GNSS-like signals should have non-zero structure here; pure noise or generic jammers less so.
- **Formula**: lag

$$
  L_1 = \text{round}\big(f_s / 1.023\,\text{MHz}\big),
$$

$$
  \text{cyclo chip} = \text{cyclo lag}(z, L_1).
$$

51. **`cyclo_2chip_corr`**

- **Intuition**: Same idea but at 2 chip periods.
- **Formula**:

$$
  L_2 = \text{round}\big(f_s / 2.046\,\text{MHz}\big),
$$

$$
  \text{cyclo 2chip} = \text{cyclo lag}(z, L_2).
$$

### 7.2. Higher-order cumulants

We normalise $z$ to unit average power and compute 4th-order cumulants.

Let

$$
z_c[n] = z[n] - \overline{z},\quad
p = \mathbb{E}[|z_c|^2] + \varepsilon,\quad
z_n[n] = \frac{z_c[n]}{\sqrt{p}}.
$$

Moments:

$$
m_{20} = \mathbb{E}[z_n^2],\quad
m_{40} = \mathbb{E}[z_n^4],\quad
m_{42} = \mathbb{E}[|z_n|^2 z_n^2].
$$

Cumulants:

$$
c_{40} = m_{40} - 3 m_{20}^2,\quad
c_{42} = m_{42} - |m_{20}|^2 - 2.
$$

52. **`cumulant_c40_mag`**

- **Intuition**: Magnitude of 4th-order cumulant $C_{40}$; sensitive to modulation format and non-Gaussianity.
- **Formula**:

$$
  \text{cumulant c40} = |c_{40}|.
$$

53. **`cumulant_c42_mag`**

- **Intuition**: Similar to above for $C_{42}$; together, $C_{40}$ and $C_{42}$ help discriminate between modulations (BPSK, QPSK, etc.) and noise.
- **Formula**:

$$
  \text{cumulant c42} = |c_{42}|.
$$

### 7.3. Spectral kurtosis

We compute a spectrogram $S_{xx}[i,j]$ (frequency $i$, time $j$) of $z[n]$ (PSD mode). For each frequency bin $i$ we look at how its power varies across time.

Per-bin kurtosis:

$$
\text{kurt}_i =
  \frac{\mathbb{E}_j[(S_{xx}[i,j] - \mu_i)^4]}
       {(\mathbb{E}_j[(S_{xx}[i,j] - \mu_i)^2])^2},
\quad \mu_i = \mathbb{E}_j[S_{xx}[i,j]].
$$

54. **`spec_kurtosis_mean`**

- **Intuition**: Average “burstiness” across all frequencies.  
  If many frequencies are sometimes very loud and sometimes quiet, this rises.
- **Formula**:

$$
  \text{spec kurtosis} =
    \frac{1}{I} \sum_{i=1}^I \text{kurt}_i.
$$

55. **`spec_kurtosis_max`**

- **Intuition**: Maximal burstiness at any frequency.  
  Good for detecting a single frequency that occasionally spikes.
- **Formula**:

$$
  \text{spec kurtosis} = \max_i \text{kurt}_i.
$$

### 7.4. Teager–Kaiser on envelope

56. **`tkeo_env_mean`**

- **Intuition**: Sensitive to **instantaneous energy changes** in the envelope (like a second derivative adapted for energy).  
  Higher for signals with rapid local changes.
- **Formula**:

  Let $e[n] = |z[n]|$ and for $n=1,\dots,N-2$:

$$
  \psi[n] = e[n]^2 - e[n-1] e[n+1],
$$

  then clamp $\psi[n]\ge0$ and

$$
  \text{tkeo env} =
    \frac{\mathbb{E}[\psi[n]]}{\mathbb{E}[e[n]]^2 + \varepsilon}.
$$

---

## 8. Higher-order I/Q Stats & Circularity (6)

**Category intuition**

Proper complex Gaussian noise has:

- zero skewness,
- kurtosis ≈ 3,
- and is **circular** (no preferred axis in I/Q plane).

These features measure how far we are from that ideal, giving clues about modulation and interference structure.

---

57. **`skewI`**, 58. **`skewQ`**

- **Intuition**: Asymmetry of I and Q histograms.  
  Heavy skew can indicate offset or one-sided modulation.
- **Formula**:

$$
  \text{skewI} =
    \frac{\mathbb{E}[(I - \mu_I)^3]}{\sigma_I^3 + \varepsilon},
  \quad
  \text{skewQ} =
    \frac{\mathbb{E}[(Q - \mu_Q)^3]}{\sigma_Q^3 + \varepsilon}.
$$

59. **`kurtI`**, 60. **`kurtQ`**

- **Intuition**: Tail heaviness of I and Q distributions.  
  Pulses or outliers increase these.
- **Formula**:

$$
  \text{kurtI} =
    \frac{\mathbb{E}[(I - \mu_I)^4]}{\big(\mathbb{E}[(I - \mu_I)^2]\big)^2},
$$

$$
  \text{kurtQ} =
    \frac{\mathbb{E}[(Q - \mu_Q)^4]}{\big(\mathbb{E}[(Q - \mu_Q)^2]\big)^2}.
$$

61. **`circularity_mag`**, 62. **`circularity_phase_rad`**

- **Intuition**:
  - Proper circular complex noise → $E[z^2] \approx 0$, so circularity magnitude ≈ 0.  
  - Strong modulation confined to I or Q → large magnitude, phase capturing orientation.
- **Formula**:

$$
  d = \mathbb{E}[|z|^2] + \varepsilon, \quad
  \rho = \frac{\mathbb{E}[z^2]}{d},
$$

$$
  \text{circularity} = |\rho|,\quad
  \text{circularity phase} = \arg(\rho).
$$

---

## 9. Inequality, Symmetry, DC Notch & Peakiness (6)

**Category intuition**

These features summarise:

- How unevenly power is distributed across frequencies and amplitudes (Gini, peakiness),
- How symmetric the spectrum is around DC,
- Whether we have a notch or spike near DC.

---

63. **`spec_gini`**

- **Intuition**: Gini coefficient of the normalized PSD.  
  - 0 → perfectly equal power per bin.  
  - 1 → all power in a single bin.  
  Another “peakiness” measure.
- **Formula**:

  Let $x_k = P^\text{norm}_{xx}[k]$ sorted ascending, $S = \sum_k x_k$:

$$
  G = \frac{2 \sum_{k=1}^K k x_k}{K S} - \frac{K+1}{K},
$$

  clipped to $[0,1]$.

64. **`env_gini`**

- **Intuition**: Same concept but applied to the envelope samples.  
  High value → only a few samples carry most amplitude (strong pulses).
- **Formula**:

  Let

$$
  x_n = \frac{\max(\text{env}_\text{raw}[n], 0)}{\sum_n \max(\text{env}_\text{raw}[n], 0) + \varepsilon},
$$

  sort and apply the same Gini formula.

65. **`env_p95_over_p50`**

- **Intuition**: “How much bigger are the large amplitudes than the median?”.  
  Pulsed signals → 95th percentile much larger than median.
- **Formula**:

$$
  p_{95} = \text{percentile}_{95}(\text{env}_\text{raw}),\quad
  p_{50} = \text{percentile}_{50}(\text{env}_\text{raw}),
$$

$$
  \text{env p95 over} = \frac{p_{95}}{p_{50} + \varepsilon}.
$$

66. **`spec_symmetry_index`**

- **Intuition**: Whether positive and negative frequencies carry similar power.  
  LO offsets or IF design can make things asymmetric.
- **Formula**:

$$
  P_+ = \sum_{k: f_k > 0} P_{xx}[k], \quad
  P_- = \sum_{k: f_k < 0} P_{xx}[k],
$$

$$
  \text{spec symmetry} = \frac{P_+ - P_-}{P_+ + P_- + \varepsilon}.
$$

67. **`dc_notch_ratio`**

- **Intuition**: Power near DC compared to power in a wider central band.  
  - Notch filter at DC → low ratio.  
  - Strong DC spike → high ratio.
- **Formula**:

$$
  \mathcal{D} = \{k: |f_k| \le 0.5\,\text{MHz}\},\quad
  \mathcal{R} = \{k: |f_k| \le 5\,\text{MHz}\},
$$

$$
  \text{dc notch} = \frac{\sum_{k \in \mathcal{D}} P_{xx}[k]}{\sum_{k \in \mathcal{R}} P_{xx}[k] + \varepsilon}.
$$

68. **`spec_peakiness_ratio`**

- **Intuition**: Simple ratio of max PSD to median PSD.  
  - A few tones over noise → very large.  
  - Flat noise → closer to 1.
- **Formula**:

$$
  \text{spec peakiness} =
    \frac{\max_k P_{xx}[k]}{\text{median}_k(P_{xx}[k]) + \varepsilon}.
$$

---

## 10. STFT-based Time–Frequency Dynamics (5)

**Category intuition**

These use an STFT (short-time Fourier transform) to follow the spectrum over time. They capture:

- How the **spectral centroid** moves,
- Whether it jumps (FH-like),
- Whether at each time we see a broad or narrow ridge of power.

---

We compute a spectrogram of $z[n]$:

$$
(f_i, t_j, S_{xx}[i,j]) = \text{spectrogram}(z, f_s)
$$

(Hann window, `STFT_NPERSEG`, `STFT_NOVERLAP`, `STFT_NFFT`, PSD mode).

Time hop:

$$
\Delta t = \frac{\text{STFT} - \text{STFT}}{f_s}.
$$

Normalise each time column:

$$
S^\text{norm}_{xx}[i,j] =
  \frac{S_{xx}[i,j]}{\sum_l S_{xx}[l,j] + \varepsilon}.
$$

Define spectral centroid per frame:

$$
c_j = \sum_i f_i S^\text{norm}_{xx}[i,j].
$$

69. **`stft_centroid_std_Hz`**

- **Intuition**: How far the centroid moves around its average position.  
  Useful for spotting very mobile interference (e.g. FH).
- **Formula**:

$$
  \overline{c} = \frac{1}{J}\sum_j c_j,
$$

$$
  \text{stft centroid std} =
    \sqrt{\frac{1}{J}\sum_j (c_j - \overline{c})^2}.
$$

70. **`stft_centroid_absderiv_med_Hzps`**

- **Intuition**: Typical speed (in Hz/s) at which the centroid moves.
- **Formula**:

$$
  d_c[j] = c_{j+1} - c_j,
$$

$$
  \text{stft centroid absderiv med} =
    \text{median}_j\left(\left|\frac{d_c[j]}{\Delta t}\right|\right).
$$

71. **`stft_centroid_zcr_per_s`**

- **Intuition**: How often the centroid’s velocity changes sign per second (back-and-forth movement).
- **Formula**:

$$
  \text{ZCR}(d_c) = \frac{\left|\{j: d_c[j]\cdot d_c[j+1] < 0\}\right|}{J-1},
$$

$$
  \text{stft centroid zcr per} =
    \frac{\text{ZCR}(d_c)}{\Delta t}.
$$

72. **`fh_hop_rate_per_s`**

- **Intuition**: Approximate **hop rate** of strong frequency jumps → aimed at FH jammers.
- **Formula**:

$$
  \text{mad} = \text{median}_j(|d_c[j] - \text{median}(d_c)|) + 10^{-6},
$$

$$
  T_\text{hop} = \max(5\cdot10^5, 6 \cdot \text{mad}),
$$

$$
  N_\text{hops} = \left|\{j : |d_c[j]| > T_\text{hop}\}\right|,
  \quad
  T_\text{dur} = (J-1)\Delta t,
$$

$$
  \text{fh hop rate per} = \frac{N_\text{hops}}{T_\text{dur} + \varepsilon}.
$$

73. **`strong_bins_mean`**

- **Intuition**: On average, how many time-frequency bins are “strong” (above half of the max at each time).  
  - Wideband jammers → many strong bins.  
  - Narrow tones → very few.
- **Formula**:

$$
  M_j = \max_i S^\text{norm}_{xx}[i,j],
$$

$$
  \text{strong mask}_{i,j} =
    \begin{cases}
    1, & S^\text{norm}_{xx}[i,j] > 0.5 M_j\\
    0, & \text{otherwise}
    \end{cases},
$$

$$
  \text{strong bins} =
    \frac{1}{I J} \sum_{i,j} \text{strong mask}_{i,j}.
$$

---

## 11. Extra Cyclo Lags, Chirp Curvature, DME IPIs (5)

**Category intuition**

Finally we have:

- Extra cyclostationary lags around the chip period,
- A curvature term to detect non-linear chirps,
- Detailed interpulse-interval stats for DME-like interference.

---

### 11.1. Extra cyclostationarity

74. **`cyclo_halfchip_corr`**

- **Intuition**: Cyclo correlation at **half a chip**. Gives extra granularity on how chip-like the structure is.
- **Formula**:

$$
  L_{\frac12} = \text{round}\left(\frac{f_s}{2 \cdot 1.023\,\text{MHz}}\right),
$$

$$
  \text{cyclo halfchip} = \text{cyclo lag}(z, L_{\frac12}).
$$

75. **`cyclo_5chip_corr`**

- **Intuition**: Cyclo correlation at **5 chips**. Checks for longer-range chip periodicity.
- **Formula**:

$$
  L_5 = \text{round}\left(5 \cdot \frac{f_s}{1.023\,\text{MHz}}\right),
$$

$$
  \text{cyclo 5chip} = \text{cyclo lag}(z, L_5).
$$

### 11.2. Chirp curvature from STFT

We again use the STFT centroid sequence $c_j$ and times $t_j = j\Delta t$.

76. **`chirp_curvature_Hzps2`**

- **Intuition**: Measures **curvature** of the frequency trajectory.  
  - Linear chirp → curvature ≈ 0.  
  - Non-linear sweep → non-zero curvature.
- **Formula**: Fit

$$
  c_j \approx a t_j^2 + b t_j + c
$$

  and take

$$
  \text{chirp curvature} = 2a.
$$

### 11.3. DME interpulse intervals (IPIs)

For DME-like pulsed interference we look at the time between pulses.

- Smooth $\text{env}_\text{raw}$ with a shorter window ($\approx 0.3\,\mu$s) to get $\text{env}_s[n]$.
- Threshold as before: $T = \mathbb{E}[\text{env}_s] + 3\cdot\text{std}(\text{env}_s)$.
- Detect peaks $p_0,\dots,p_{K-1}$ (at least $\approx 0.2\,\mu$s apart).

Interpulse intervals:

$$
\text{IPI}_k = \frac{p_{k+1} - p_k}{f_s},\quad k=0,\dots,K-2.
$$

77. **`dme_ipi_med_s`**

- **Intuition**: Typical spacing between pulses (seconds).  
  Useful to recognise specific pulsed systems like DME.
- **Formula**:

$$
  \text{dme ipi med} = \text{median}_k(\text{IPI}_k).
$$

78. **`dme_ipi_std_s`**

- **Intuition**: How regular those IPIs are.  
  - Very periodic pulses → low std.  
  - Irregular bursts → higher std.
- **Formula**:

$$
  \text{dme ipi std} = \text{std}_k(\text{IPI}_k).
$$

---

## 12. Big-picture summary

- The feature vector combines:
  - **Time-domain shape** (DC, variance, ZCR, PAPR, envelope stats),
  - **Global and local spectral shape** (centroid, spread, band powers, peaks),
  - **Instantaneous frequency dynamics** (drift, slope, jitter, hops),
  - **Envelope modulation and pulses** (AM, DME-style, cepstrum),
  - **Cyclostationarity & higher-order structure** (GNSS chip periodicity, cumulants),
  - **Non-Gaussian / non-circular behaviour** (skew, kurtosis, circularity),
  - **Time–frequency evolution** via STFT features.

- Together they form a **strict, interpretable fingerprint** of a GNSS+jammer IQ chunk, suitable for supervised learning and for qualitative inspection by humans.