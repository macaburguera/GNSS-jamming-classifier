# sbf_morning_prn20w_predict_every_Xsec_cfg_multiband.py
#
# Sample IQ from a Septentrio SBF file every X seconds, for EACH LO/band
# logged in the BBSamples, label using the 9/10 high-power PRN (20 W)
# schedule, run an XGBoost jammer model on each sample, and save
# spectrograms with prediction + LO/band in the title.

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import stft, welch, spectrogram, find_peaks
from scipy import ndimage

from joblib import load as joblib_load
from datetime import datetime, timedelta, timezone

from sbf_parser import SbfParser

# ============================ CONFIG ============================

# Paths
SBF_PATH   = r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\Geopositioning and Navigation - Jammertest 2023\23.09.19 - Jammertest 2023 - Day 2\alt02 - reference during kraken test at location 2 for smartphone comparison.sbf"
OUT_DIR    = r"D:\datasets\Jammertest2026_Day2\alt02-sbf-morning\out_alt02_hp20w_10s_multiband"
MODEL_PATH = r"..\artifacts\jammertest_sim\xgb_run_20251117_182853\xgb_20251117_182911\xgb_trainval.joblib"

# Sampling policy: one IQ snap & plot every X seconds (UTC) *per LO/band*
SAMPLE_PERIOD_SEC = 30.0     # e.g. 10, 15, 30...
MAX_SAMPLES       = 600      # max total number of samples / plots (all bands combined)

# Optional decimation of IQ before features + spectrogram
DECIM         = 1            # 1 = no decimation
NPERSEG_STFT  = 128          # STFT window for plots
NOVERLAP_STFT = 96
DPI_FIG       = 120
VMIN_DB       = -80
VMAX_DB       = -20
REMOVE_DC     = True
CHUNK_BYTES   = 1_000_000    # SBF read chunk size

# Testplan / local time assumptions
LOCAL_UTC_OFFSET = 2.0       # LOCAL minus UTC (e.g. CEST=+2)
MORNING_START_H  = 10        # hour (LOCAL time) of test 1 ON start
MORNING_START_M  = 35        # minute (LOCAL time) of test 1 ON start

# ===============================================================

EPS = 1e-20
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (stable since 2017-01-01)

MODEL_CLASS_NAMES = ["NoJam", "Chirp", "NB", "CW", "WB", "FH"]
CANON = {c.lower(): c for c in MODEL_CLASS_NAMES}

def canon(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    return CANON.get(s.lower(), s)

# -------------------- time helpers --------------------

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600)
    m = int((tsec % 3600) // 60)
    s = tsec - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)

def extract_time_labels(infos):
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0))
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

# -------------------- SBF IQ decode --------------------

def decode_bbsamples_iq(infos):
    """
    Decode BBSamples into complex IQ + fs + LO frequency (Hz).
    """
    if "Samples" not in infos or "N" not in infos:
        return None, None, None
    buf = infos["Samples"]; N = int(infos.get("N", 0))
    arr = np.frombuffer(buf, dtype=np.int8)
    if arr.size != 2 * N:
        return None, None, None
    I = arr[0::2].astype(np.float32) / 128.0
    Q = arr[1::2].astype(np.float32) / 128.0
    x = I + 1j * Q
    fs = float(infos.get("SampleFreq", 1.0))
    flo = float(infos.get("LOFreq", 0.0))  # <<< NEW: LO in Hz
    return x, fs, flo

# -------------------- band naming helper --------------------  # <<< NEW

def band_name_from_lo(flo_hz: float) -> str:
    """
    Roughly classify LO into a band label, based on what we saw in alt02.
    """
    if flo_hz <= 0:
        return "Unknown"
    if 1.57e9 <= flo_hz <= 1.60e9:
        return "L1_E1"
    if 1.20e9 <= flo_hz <= 1.26e9:
        return "L2_E5b"
    if 1.16e9 <= flo_hz <= 1.20e9:
        return "L5_E5ab"
    # fallback: generic Hz label
    return f"LO_{flo_hz/1e6:.1f}MHz"

# ======================= 9/10 TESTPLAN LABELS =======================

N_TESTS = 15
TEST_ON_SEC = 3 * 60
BREAK_SEC = 2 * 60
FINAL_BREAK_SEC = 15 * 60

TOTAL_SCENARIO_SEC = (
    N_TESTS * TEST_ON_SEC +
    (N_TESTS - 1) * BREAK_SEC +
    FINAL_BREAK_SEC
)

def label_from_testplan(
    utc_dt: datetime,
    local_utc_offset_hours: float,
    morning_start_h: int,
    morning_start_m: int,
) -> Tuple[str, str, Optional[int]]:
    """
    Map utc_dt to (log_label, gt_label, test_idx) according to the 9./10.
    high-power PRN (20 W) schedule.
    """
    local_dt = utc_dt + timedelta(hours=float(local_utc_offset_hours))
    base_local = local_dt.replace(
        hour=morning_start_h,
        minute=morning_start_m,
        second=0,
        microsecond=0,
    )
    offset_sec = (local_dt - base_local).total_seconds()

    if offset_sec < 0 or offset_sec >= TOTAL_SCENARIO_SEC:
        return "Outside_HP_PRN_scenario", "NoJam", None

    t = offset_sec
    for i in range(N_TESTS):
        on_start = i * (TEST_ON_SEC + BREAK_SEC)
        on_end = on_start + TEST_ON_SEC
        if on_start <= t < on_end:
            test_idx = i + 1
            log_label = f"TC9/10_HP_PRN_T{test_idx:02d}_ON"
            gt = "WB"
            return log_label, gt, test_idx

        if i < N_TESTS - 1:
            break_end = on_end + BREAK_SEC
        else:
            break_end = on_end + FINAL_BREAK_SEC

        if on_end <= t < break_end:
            test_idx = i + 1
            log_label = f"TC9/10_break_after_T{test_idx:02d}"
            gt = "NoJam"
            return log_label, gt, test_idx

    return "Outside_HP_PRN_scenario", "NoJam", None

# ======================= FEATURE EXTRACTOR =======================

WELCH_NPERSEG = 256
WELCH_OVERLAP = 128
MAX_LAG_S     = 200e-6
INB_BW_HZ     = 2_000_000.0
ENV_FFT_BAND  = (30.0, 7_000.0)
CHIRP_TARGET_SLICE_S = 0.125e-3
CHIRP_MIN_SLICES     = 6
CHIRP_MAX_SLICES     = 24
CYC_ALPHA1_HZ = 1.023e6
CYC_ALPHA2_HZ = 2.046e6
STFT_NPERSEG  = 128
STFT_NOVERLAP = 96
STFT_NFFT     = 128

def safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s <= 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))

def safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 3.0
    m = x.mean()
    v = np.mean((x - m) ** 2)
    if v <= 0:
        return 3.0
    m4 = np.mean((x - m) ** 4)
    return float(m4 / (v ** 2))

def fast_autocorr_env(z: np.ndarray, fs: float, max_lag_s: float):
    env = np.abs(z).astype(np.float64)
    env -= env.mean()
    n = int(1 << int(np.ceil(np.log2(max(2, 2 * env.size - 1)))))
    E = np.fft.rfft(env, n=n)
    ac = np.fft.irfft(np.abs(E) ** 2, n=n)
    ac = ac[:env.size]
    if ac.size == 0 or ac[0] <= 0:
        return 0.0, 0.0
    ac /= ac[0]
    max_lag = int(min(env.size - 1, max(1, round(max_lag_s * fs))))
    if max_lag <= 1:
        return 0.0, 0.0
    k = 1 + int(np.argmax(ac[1:max_lag]))
    return float(ac[k]), float(k / fs)

def zero_crossing_rate(x):
    return float(np.mean(np.abs(np.diff(np.signbit(x))).astype(float))) if x.size > 1 else 0.0

def spectral_moments(f, Pxx):
    P = np.maximum(Pxx, EPS); w = P / P.sum()
    mu = float(np.sum(f * w))
    std = float(np.sqrt(np.sum(((f - mu) ** 2) * w)))
    return mu, std

def spectral_rolloff(f, Pxx, roll=0.95):
    P = np.maximum(Pxx, EPS); c = np.cumsum(P)
    idx = int(np.searchsorted(c, roll * c[-1])); idx = min(idx, len(f)-1)
    return float(f[idx])

def spectral_flatness(Pxx):
    P = np.maximum(Pxx, EPS)
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))

def bandpowers(f, Pxx, bands):
    P = np.maximum(Pxx, EPS)
    out = []
    for f1, f2 in bands:
        mask = (f >= f1) & (f < f2)
        out.append(P[mask].sum())
    out = np.array(out, float)
    s = out.sum()
    return (out / (s + EPS)).astype(float)

def inst_freq_stats(z: np.ndarray, fs: float):
    if len(z) < 3:
        return 0.0, 0.0, 0.0, 3.0, 0.0
    phi = np.unwrap(np.angle(z))
    inst_f = np.diff(phi) * fs / (2*np.pi)
    if inst_f.size > 8:
        lo, hi = np.percentile(inst_f, [1, 99])
        inst_f = np.clip(inst_f, lo, hi)
    t = np.arange(inst_f.size, dtype=np.float64) / fs
    slope = float(np.polyfit(t, inst_f, 1)[0]) if inst_f.size >= 2 else 0.0
    kurt = safe_kurtosis(inst_f)
    dinst = np.diff(inst_f)
    if dinst.size < 2:
        dzcr_per_s = 0.0
    else:
        frac = zero_crossing_rate(dinst)
        dzcr_per_s = float(frac * fs)
    return float(np.mean(inst_f)), float(np.std(inst_f)), slope, kurt, dzcr_per_s

def cepstral_peak_env(z: np.ndarray, fs: float, qmin=2e-4, qmax=5e-3):
    env = np.abs(z).astype(np.float64)
    env = env - env.mean()
    if env.size < 8:
        return 0.0
    w = np.hanning(env.size).astype(np.float64)
    S = np.fft.rfft(env * w)
    log_mag = np.log(np.abs(S) + 1e-12)
    ceps = np.fft.irfft(log_mag)
    q = np.arange(ceps.size, dtype=np.float64) / fs
    mask = (q >= qmin) & (q <= qmax)
    return float(np.max(ceps[mask])) if np.any(mask) else 0.0

def dme_pulse_proxy(z: np.ndarray, fs: float):
    env = np.abs(z).astype(np.float64)
    win = max(3, int(round(0.5e-6 * fs)))
    env_s = ndimage.uniform_filter1d(env, size=win, mode="nearest")
    thr = env_s.mean() + 3.0 * env_s.std()
    above = env_s > thr
    rising = (above[1:] & (~above[:-1])).astype(np.int32)
    duty = float(above.mean())
    return float(rising.sum()), duty

def nb_peak_salience(f: np.ndarray, Pxx: np.ndarray, top_k=5):
    if Pxx.size < top_k:
        return 0.0
    idx = np.argpartition(Pxx, -top_k)[-top_k:]
    top = float(Pxx[idx].sum())
    rest = float(max(EPS, 1.0 - top))
    return float(top / (rest + EPS))

def nb_peaks_and_spacing(f: np.ndarray, Pxx: np.ndarray):
    maxp = float(Pxx.max())
    if maxp <= 0:
        return 0.0, 0.0, 0.0
    prom = 0.03 * maxp
    idx, _ = find_peaks(Pxx, prominence=prom)
    if idx.size < 2:
        return float(idx.size), 0.0, 0.0
    freqs = f[idx]
    spac = np.diff(np.sort(freqs))
    return float(idx.size), float(np.median(spac)), float(np.std(spac))

def am_envelope_features(z: np.ndarray, fs: float, fmin=ENV_FFT_BAND[0], fmax=ENV_FFT_BAND[1]):
    env = np.abs(z).astype(np.float64)
    mu = env.mean()
    mod_index = float(np.var(env) / (mu*mu + EPS))
    e = env - mu
    n = int(1 << int(np.ceil(np.log2(max(8, e.size)))))
    E = np.fft.rfft(np.hanning(e.size) * e.astype(np.float64), n=n)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    P = np.abs(E)**2
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return mod_index, 0.0, 0.0
    fb, Pb = f[band], P[band]
    k = int(np.argmax(Pb))
    peak_f = float(fb[k])
    peak_pow_norm = float(Pb[k] / (Pb.sum() + EPS))
    return mod_index, peak_f, peak_pow_norm

def choose_chirp_slices(N: int, fs: float) -> int:
    total_t = N / float(fs)
    target = CHIRP_TARGET_SLICE_S
    s = int(round(max(CHIRP_MIN_SLICES, min(CHIRP_MAX_SLICES, total_t / target))))
    return max(CHIRP_MIN_SLICES, min(CHIRP_MAX_SLICES, s))

def chirp_slope_proxy(z: np.ndarray, fs: float, slices: int):
    N = len(z)
    if slices < 2 or N < slices * 8:
        return 0.0, 0.0
    seg = np.array_split(z, slices)
    cents, times = [], []
    for i, s in enumerate(seg):
        f1, Pxx = welch(s.astype(np.complex64), fs=fs, window="hann",
                        nperseg=min(WELCH_NPERSEG, max(16, len(s)//2)),
                        noverlap=0, return_onesided=False, scaling="density")
        order = np.argsort(f1)
        f1, Pxx = f1[order], np.maximum(Pxx[order], EPS)
        Pxx /= (Pxx.sum() + EPS)
        mu, _ = spectral_moments(f1, Pxx)
        cents.append(mu)
        times.append((i + 0.5) * (N/fs) / slices)
    cents = np.array(cents, float); times = np.array(times, float)
    if cents.size < 2:
        return 0.0, 0.0
    p = np.polyfit(times, cents, 1)
    slope = float(p[0])
    yhat = np.polyval(p, times)
    ss_res = float(np.sum((cents - yhat)**2))
    ss_tot = float(np.sum((cents - cents.mean())**2) + EPS)
    r2 = float(1.0 - ss_res/ss_tot)
    return slope, r2

def cyclo_lag_corr(z: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(z):
        return 0.0
    a = z[lag:]; b = z[:-lag]
    num = np.vdot(b, a)
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real) + EPS
    return float(np.abs(num) / den)

def cyclo_proxies(z: np.ndarray, fs: float):
    L1 = int(round(fs / CYC_ALPHA1_HZ)) if CYC_ALPHA1_HZ > 0 else 0
    L2 = int(round(fs / CYC_ALPHA2_HZ)) if CYC_ALPHA2_HZ > 0 else 0
    return cyclo_lag_corr(z, L1), cyclo_lag_corr(z, L2)

def cumulants_c40_c42(z: np.ndarray):
    if z.size < 8:
        return 0.0, 0.0
    zc = z - np.mean(z)
    p = np.mean(np.abs(zc)**2) + EPS
    zn = zc / np.sqrt(p)
    m20 = np.mean(zn**2)
    m40 = np.mean(zn**4)
    m42 = np.mean((np.abs(zn)**2) * (zn**2))
    c40 = m40 - 3.0 * (m20**2)
    c42 = m42 - (np.abs(m20)**2) - 2.0
    return float(np.abs(c40)), float(np.abs(c42))

def spectral_kurtosis_stats(z: np.ndarray, fs: float):
    try:
        f, t, Sxx = spectrogram(z, fs=fs, window="hann",
                                nperseg=256, noverlap=128,
                                detrend=False, return_onesided=False,
                                scaling="density", mode="psd")
        if Sxx.ndim != 2 or Sxx.shape[1] < 4:
            return 0.0, 0.0
        def kurt_pop_safe(v):
            v = np.asarray(v, float)
            if v.size < 4:
                return 3.0
            m = v.mean()
            s2 = np.mean((v - m) ** 2)
            if s2 <= 0:
                return 3.0
            m4 = np.mean((v - m) ** 4)
            return float(m4 / (s2 ** 2))
        sk = np.array([kurt_pop_safe(Sxx[i, :]) for i in range(Sxx.shape[0])], dtype=float)
        sk = np.clip(sk, 0.0, 1e6)
        return float(np.mean(sk)), float(np.max(sk))
    except Exception:
        return 0.0, 0.0

def tkeo_env_mean(z: np.ndarray):
    e = np.abs(z).astype(np.float64)
    if e.size < 3:
        return 0.0
    psi = e[1:-1]**2 - e[:-2]*e[2:]
    psi = np.maximum(psi, 0.0)
    denom = (np.mean(e)**2 + EPS)
    return float(np.mean(psi) / denom)

def gini_coefficient(x):
    x = np.asarray(x, float).ravel()
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.sum((np.arange(1, n+1, dtype=float)) * x)
    G = (2.0 * cum) / (n * s) - (n + 1.0) / n
    return float(np.clip(G, 0.0, 1.0))

def stft_timefreq_dynamics(z: np.ndarray, fs: float):
    f, t, Sxx = spectrogram(z, fs=fs, window="hann",
                            nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                            nfft=STFT_NFFT, detrend=False,
                            return_onesided=False, scaling="density", mode="psd")
    if Sxx.ndim != 2 or Sxx.shape[1] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    order = np.argsort(f)
    f = f[order]
    Sxx = np.maximum(Sxx[order, :], EPS)
    Sxxn = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + EPS)
    cent = np.sum(f[:, None] * Sxxn, axis=0)
    if cent.size < 2:
        strong_bins_mean = float((Sxxn > 0.5*np.max(Sxxn, axis=0, keepdims=True)).mean())
        return 0.0, 0.0, 0.0, 0.0, strong_bins_mean
    dt = (STFT_NPERSEG - STFT_NOVERLAP) / fs
    dcent = np.diff(cent)
    dcent_dt = dcent / max(dt, 1/fs)
    mad = np.median(np.abs(dcent - np.median(dcent))) + 1e-6
    hop_thr = max(5e5, 6.0 * mad)
    hops = float(np.sum(np.abs(dcent) > hop_thr))
    dur = float((cent.size - 1) * dt) if cent.size > 1 else 0.0
    hop_rate = float(hops / dur) if dur > 0 else 0.0
    zcr_frac = zero_crossing_rate(dcent)
    zcr_per_s = float(zcr_frac / max(dt, 1/fs))
    strong_bins_mean = float((Sxxn > 0.5*np.max(Sxxn, axis=0, keepdims=True)).mean())
    return float(np.std(cent)), float(np.median(np.abs(dcent_dt))), zcr_per_s, hop_rate, strong_bins_mean

def symmetry_dc_peakiness(f, Pxx, fs):
    P = np.maximum(Pxx, EPS)
    pos = P[f > 0].sum()
    neg = P[f < 0].sum()
    symmetry_idx = float((pos - neg) / (pos + neg + EPS))
    dc_band = 0.5e6
    ref_band = 5e6
    m_dc = (np.abs(f) <= dc_band)
    m_ref = (np.abs(f) <= ref_band)
    dc_notch_ratio = float(P[m_dc].sum() / (P[m_ref].sum() + EPS))
    peakiness_ratio = float(P.max() / (np.median(P) + EPS))
    return symmetry_idx, dc_notch_ratio, peakiness_ratio

def dme_ipi_stats(z: np.ndarray, fs: float):
    env = np.abs(z).astype(np.float64)
    if env.size < 8:
        return 0.0, 0.0
    win = max(3, int(round(0.3e-6 * fs)))
    env_s = ndimage.uniform_filter1d(env, size=win, mode="nearest")
    thr = env_s.mean() + 3.0 * env_s.std()
    pk, _ = find_peaks(env_s, height=thr, distance=max(1, int(0.2e-6*fs)))
    if pk.size < 2:
        return 0.0, 0.0
    ipi = np.diff(pk) / fs
    return float(np.median(ipi)), float(np.std(ipi))

FEATURE_NAMES = (
    ["meanI","meanQ","stdI","stdQ","corrIQ","mag_mean","mag_std",
     "ZCR_I","ZCR_Q","PAPR_dB","env_ac_peak","env_ac_lag_s",
     "pre_rms","psd_power","oob_ratio","crest_env","kurt_env","spec_entropy"]
    + ["spec_centroid_Hz","spec_spread_Hz","spec_flatness",
       "spec_rolloff95_Hz","spec_peak_freq_Hz","spec_peak_power"]
    + [f"bandpower_{i}" for i in range(8)]
    + ["instf_mean_Hz","instf_std_Hz","instf_slope_Hzps","instf_kurtosis","instf_dZCR_per_s",
       "cep_peak_env","dme_pulse_count","dme_duty","nb_peak_salience"]
    + ["nb_peak_count","nb_spacing_med_Hz","nb_spacing_std_Hz",
       "env_mod_index","env_dom_freq_Hz","env_dom_peak_norm",
       "chirp_slope_Hzps","chirp_r2"]
    + ["cyclo_chip_corr","cyclo_2chip_corr",
       "cumulant_c40_mag","cumulant_c42_mag",
       "spec_kurtosis_mean","spec_kurtosis_max",
       "tkeo_env_mean"]
    + ["skewI","skewQ","kurtI","kurtQ","circularity_mag","circularity_phase_rad"]
    + ["spec_gini","env_gini","env_p95_over_p50","spec_symmetry_index","dc_notch_ratio","spec_peakiness_ratio"]
    + ["stft_centroid_std_Hz","stft_centroid_absderiv_med_Hzps",
       "stft_centroid_zcr_per_s","fh_hop_rate_per_s","strong_bins_mean"]
    + ["cyclo_halfchip_corr","cyclo_5chip_corr"]
    + ["chirp_curvature_Hzps2"]
    + ["dme_ipi_med_s","dme_ipi_std_s"]
)
PRE_RMS_IDX = FEATURE_NAMES.index("pre_rms")

def extract_features(iq: np.ndarray, fs: float) -> np.ndarray:
    iq = iq.astype(np.complex64, copy=False)

    env_raw = np.abs(iq).astype(np.float64)
    pre_rms = float(np.sqrt(np.mean(np.abs(iq)**2)) + EPS)
    crest_env = float(env_raw.max() / (env_raw.mean() + EPS))
    kurt_env = safe_kurtosis(env_raw)

    f0, Pxx0 = welch(iq, fs=fs, window="hann", nperseg=WELCH_NPERSEG,
                     noverlap=WELCH_OVERLAP, return_onesided=False, scaling="density")
    order0 = np.argsort(f0)
    f0, Pxx0 = f0[order0], np.maximum(Pxx0[order0], EPS)
    psd_power = float(Pxx0.sum())
    Pprob0 = Pxx0 / (Pxx0.sum() + EPS)
    spec_entropy = float(-np.sum(Pprob0 * np.log(Pprob0)))
    inb = (np.abs(f0) <= INB_BW_HZ)
    oob_ratio = float(Pxx0[~inb].sum() / (Pxx0[inb].sum() + EPS))

    z = (iq / (pre_rms if pre_rms > 0 else 1.0)).astype(np.complex64)
    I, Q = z.real.astype(float), z.imag.astype(float)
    mag  = np.abs(z).astype(float)

    corrIQ = float(np.corrcoef(I, Q)[0, 1]) if I.size > 1 else 0.0
    papr_db = float(20*np.log10((mag.max()+EPS)/(mag.mean()+EPS)))
    ac_peak, ac_lag = fast_autocorr_env(z, fs, MAX_LAG_S)

    feats = [
        I.mean(), Q.mean(), I.std(), Q.std(),
        corrIQ, mag.mean(), mag.std(),
        zero_crossing_rate(I), zero_crossing_rate(Q),
        papr_db, ac_peak, ac_lag,
        pre_rms, psd_power, oob_ratio, crest_env, kurt_env, spec_entropy
    ]

    f1, Pxx = welch(z, fs=fs, window="hann", nperseg=WELCH_NPERSEG,
                    noverlap=WELCH_OVERLAP, return_onesided=False, scaling="density")
    order = np.argsort(f1)
    f, Pxx = f1[order], np.maximum(Pxx[order], EPS)
    Pxx /= (Pxx.sum() + EPS)
    mu, std = spectral_moments(f, Pxx)
    flat = spectral_flatness(Pxx)
    roll = spectral_rolloff(f, Pxx, 0.95)
    kmax = int(np.argmax(Pxx)); fmax = float(f[kmax]); pmax = float(Pxx[kmax])
    feats += [mu, std, flat, roll, fmax, pmax]

    edges = np.linspace(-fs/2, fs/2, 9)
    feats += bandpowers(f, Pxx, list(zip(edges[:-1], edges[1:]))).tolist()

    f_mean, f_std, f_slope, f_kurt, instf_dzcr = inst_freq_stats(z, fs)
    feats += [f_mean, f_std, f_slope, f_kurt, instf_dzcr]

    cep_env = cepstral_peak_env(z, fs, 2e-4, 5e-3)
    pulse_cnt, duty = dme_pulse_proxy(z, fs)
    sal = nb_peak_salience(f, Pxx, top_k=5)
    feats += [cep_env, pulse_cnt, duty, sal]

    pk_cnt, sp_med, sp_std = nb_peaks_and_spacing(f, Pxx)
    mi, env_f, env_pk = am_envelope_features(z, fs)
    slices = choose_chirp_slices(len(z), fs)
    ch_slope, ch_r2 = chirp_slope_proxy(z, fs, slices)
    feats += [pk_cnt, sp_med, sp_std, mi, env_f, env_pk, ch_slope, ch_r2]

    c1, c2 = cyclo_proxies(z, fs)
    c40_mag, c42_mag = cumulants_c40_c42(z)
    sk_mean, sk_max = spectral_kurtosis_stats(z, fs)
    tkeo_mean = tkeo_env_mean(z)
    feats += [c1, c2, c40_mag, c42_mag, sk_mean, sk_max, tkeo_mean]

    skewI = safe_skew(I); skewQ = safe_skew(Q)
    kurtI = safe_kurtosis(I); kurtQ = safe_kurtosis(Q)
    den = float(np.mean(np.abs(z)**2) + EPS)
    rho = np.mean(z**2) / den
    circularity_mag = float(np.abs(rho))
    circularity_phase = float(np.angle(rho))
    feats += [skewI, skewQ, kurtI, kurtQ, circularity_mag, circularity_phase]

    spec_gini = gini_coefficient(Pxx)
    env_gini  = gini_coefficient(env_raw / (env_raw.sum() + EPS))
    p95 = float(np.percentile(env_raw, 95))
    p50 = float(np.percentile(env_raw, 50))
    env_p95_over_p50 = float(p95 / (p50 + EPS))
    sym_idx, dc_notch_ratio, peakiness_ratio = symmetry_dc_peakiness(f, Pxx, fs)
    feats += [spec_gini, env_gini, env_p95_over_p50, sym_idx, dc_notch_ratio, peakiness_ratio]

    stft_std, stft_absder_med, stft_zcr_ps, hop_rate_ps, strong_bins_mean = stft_timefreq_dynamics(z, fs)
    feats += [stft_std, stft_absder_med, stft_zcr_ps, hop_rate_ps, strong_bins_mean]

    L_half = int(round(fs / (2.0*CYC_ALPHA1_HZ))) if CYC_ALPHA1_HZ > 0 else 0
    L_5    = int(round(5.0*fs / CYC_ALPHA1_HZ))   if CYC_ALPHA1_HZ > 0 else 0
    feats += [cyclo_lag_corr(z, L_half), cyclo_lag_corr(z, L_5)]

    f_st, t_st, Sxx_st = spectrogram(z, fs=fs, window="hann",
                                     nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                                     nfft=STFT_NFFT, detrend=False,
                                     return_onesided=False, scaling="density", mode="psd")
    if Sxx_st.ndim == 2 and Sxx_st.shape[1] >= 3:
        order2 = np.argsort(f_st)
        f_st = f_st[order2]; Sxx_st = np.maximum(Sxx_st[order2, :], EPS)
        Sxxn = Sxx_st / (np.sum(Sxx_st, axis=0, keepdims=True) + EPS)
        cent = np.sum(f_st[:, None] * Sxxn, axis=0)
        tt = np.arange(cent.size, dtype=float) * ((STFT_NPERSEG - STFT_NOVERLAP) / fs)
        p2 = np.polyfit(tt, cent, 2)
        chirp_curvature = float(2.0 * p2[0])
    else:
        chirp_curvature = 0.0
    feats += [chirp_curvature]

    ipi_med, ipi_std = dme_ipi_stats(z, fs)
    feats += [ipi_med, ipi_std]

    v = np.asarray(feats, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0
    return v

# ======================= MODEL WRAPPER =======================

def normalize_model_labeler(model):
    classes_attr = getattr(model, "classes_", None)
    to_name = None
    if classes_attr is None:
        try:
            est = model.named_steps.get("clf") or model.named_steps.get("classifier")
            classes_attr = getattr(est, "classes_", None)
        except Exception:
            classes_attr = None

    if classes_attr is not None and len(classes_attr) > 0:
        arr = np.array(classes_attr)
        if np.issubdtype(arr.dtype, np.integer):
            idx_to_name = {int(i): MODEL_CLASS_NAMES[int(i)] for i in range(len(MODEL_CLASS_NAMES))}
            to_name = lambda y: idx_to_name.get(int(y), str(y))
        else:
            to_name = lambda y: canon(y)
    else:
        to_name = lambda y: MODEL_CLASS_NAMES[int(y)] if isinstance(y, (int, np.integer)) and 0 <= int(y) < len(MODEL_CLASS_NAMES) else str(y)

    def predict_fn(feat_vec: np.ndarray):
        try:
            yhat = model.predict([feat_vec])[0]
            name = canon(to_name(yhat))
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba([feat_vec])[0]
                    raw_classes = getattr(model, "classes_", None)
                    if raw_classes is not None:
                        if np.issubdtype(np.array(raw_classes).dtype, np.integer):
                            names = [MODEL_CLASS_NAMES[int(c)] for c in raw_classes]
                        else:
                            names = [canon(c) for c in raw_classes]
                    else:
                        names = MODEL_CLASS_NAMES[:len(probs)]
                    proba = {canon(n): float(p) for n, p in zip(names, probs)}
                except Exception:
                    proba = None
            return name, proba
        except Exception:
            return None, None

    return predict_fn

# ======================= PLOTTING =======================

def plot_and_save(
    sample_idx: int,
    block_idx: int,
    x: np.ndarray,
    fs: float,
    flo: float,          # <<< NEW
    band_name: str,      # <<< NEW
    wnc: int,
    tow_s: float,
    tow_hms: str,
    utc_hms: str,
    utc_iso: str,
    utc_dt: datetime,
    log_label: str,
    gt_label: str,
    pred_label: Optional[str],
    pred_proba: Optional[Dict[str,float]],
    out_dir: Path,
) -> Optional[Path]:

    if x.size < 8:
        return None
    xx = x - np.mean(x) if REMOVE_DC else x

    nperseg_eff = min(int(NPERSEG_STFT), len(xx))
    noverlap_eff = min(int(NOVERLAP_STFT), max(0, nperseg_eff - 1))

    f, t, Z = stft(xx, fs=fs, window="hann",
                   nperseg=nperseg_eff, noverlap=noverlap_eff,
                   return_onesided=False, boundary=None, padded=False)
    if t.size < 2:
        nperseg_eff = max(16, min(len(xx)//4, nperseg_eff))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = stft(xx, fs=fs, window="hann",
                       nperseg=nperseg_eff, noverlap=noverlap_eff,
                       return_onesided=False, boundary=None, padded=False)

    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    tt = np.arange(len(xx), dtype=np.float32) / fs
    I = np.real(xx); Q = np.imag(xx)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2,1,1)
    if t.size >= 2:
        pcm = ax1.pcolormesh(t, f, S_dB, shading="auto", vmin=VMIN_DB, vmax=VMAX_DB)
        plt.colorbar(pcm, ax=ax1, label="dB")
    else:
        im = ax1.imshow(S_dB, aspect="auto", origin="lower",
                        extent=[0.0, max(1.0/fs, nperseg_eff/fs), f[0], f[-1]],
                        vmin=VMIN_DB, vmax=VMAX_DB)
        plt.colorbar(im, ax=ax1, label="dB")

    jam_txt = f" | Plan: {log_label}"
    gt_txt  = f" | GT: {gt_label}"
    if pred_label is None:
        pred_txt = " | Pred: (model failed)"
    else:
        if isinstance(pred_proba, dict) and pred_label in pred_proba:
            pred_txt = f" | Pred: {pred_label} ({pred_proba[pred_label]:.2f})"
        else:
            pred_txt = f" | Pred: {pred_label}"

    band_name_label = band_name_from_lo(flo)

    lo_str = f"LO={flo/1e6:.3f} MHz ({band_name_label})" if flo and flo > 0 else "LO=Unknown"

    title = (f"Sample #{sample_idx} (BBSamples #{block_idx}) | GPS week {wnc} | "
             f"TOW {tow_s:.3f}s ({tow_hms}) | UTC {utc_hms} | {lo_str}"
             f"{jam_txt}{gt_txt}{pred_txt}\n"
             f"nperseg={nperseg_eff}, noverlap={noverlap_eff}")
    ax1.set_title(title)
    ax1.set_ylabel("Baseband freq [Hz]")

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(tt, I, linewidth=0.7, label="I")
    ax2.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.legend(loc="upper right")
    ax2.text(0.01, 0.02, utc_iso, transform=ax2.transAxes,
             fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    safe_log  = log_label.replace(" ", "_").replace("/", "-")
    safe_pred = (pred_label or "nopred").replace(" ", "_")
    safe_band = band_name.replace(" ", "_")
    out_path = out_dir / (
        f"spec_{utc_dt.strftime('%H%M%S')}"
        f"_S{sample_idx:05d}_{safe_band}_PLAN-{safe_log}_GT-{gt_label}_PRED-{safe_pred}.png"
    )
    fig.savefig(out_path, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ======================= MAIN =======================

def main():
    sbf_path = Path(SBF_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SBF file   : {sbf_path}")
    print(f"Output dir : {out_dir}")
    print(f"Model path : {MODEL_PATH}")
    print(f"Sampling every {SAMPLE_PERIOD_SEC} s (UTC) PER LO/band")
    print(f"Morning start (LOCAL) assumed at {MORNING_START_H:02d}:{MORNING_START_M:02d}, "
          f"UTC offset={LOCAL_UTC_OFFSET:+.1f} h")
    print(f"Total scenario length from plan: {TOTAL_SCENARIO_SEC/60:.1f} min")

    model = joblib_load(MODEL_PATH)
    predict_fn = normalize_model_labeler(model)

    parser = SbfParser()
    block_i = -1
    saved = 0

    # Separate gating per LO: flo_hz -> next_sample_datetime_UTC    # <<< NEW
    next_sample_t_by_lo: Dict[float, Optional[datetime]] = {}

    with open(sbf_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            for blk, infos in parser.parse(chunk):
                if blk != "BBSamples":
                    continue
                block_i += 1

                x, fs, flo = decode_bbsamples_iq(infos)
                if x is None or flo is None:
                    continue
                if DECIM > 1:
                    x = x[::DECIM]
                    fs = fs / DECIM

                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)

                # Init per-LO sampling schedule
                if flo not in next_sample_t_by_lo or next_sample_t_by_lo[flo] is None:
                    stride = int(SAMPLE_PERIOD_SEC)
                    stride = max(1, stride)
                    floor = utc_dt.replace(
                        second=(utc_dt.second // stride) * stride,
                        microsecond=0
                    )
                    if floor > utc_dt:
                        floor -= timedelta(seconds=stride)
                    next_sample_t_by_lo[flo] = floor + timedelta(seconds=stride)

                ns = next_sample_t_by_lo[flo]
                if ns is None:
                    ns = utc_dt
                    next_sample_t_by_lo[flo] = ns

                # Gate: only first block at/after each slot for THIS LO
                if utc_dt < ns:
                    continue
                while utc_dt >= ns + timedelta(seconds=SAMPLE_PERIOD_SEC):
                    ns += timedelta(seconds=SAMPLE_PERIOD_SEC)
                next_sample_t_by_lo[flo] = ns

                # This block is our sample for this LO at this slot
                log_label, gt_label, test_idx = label_from_testplan(
                    utc_dt,
                    LOCAL_UTC_OFFSET,
                    MORNING_START_H,
                    MORNING_START_M,
                )

                feats = extract_features(x, fs)
                pred_label, pred_proba = predict_fn(feats)
                pred_label = canon(pred_label)

                band_name = band_name_from_lo(flo)

                saved += 1
                out_path = plot_and_save(
                    sample_idx=saved,
                    block_idx=block_i,
                    x=x,
                    fs=fs,
                    flo=flo,
                    band_name=band_name,
                    wnc=wnc,
                    tow_s=tow_s,
                    tow_hms=tow_hms,
                    utc_hms=utc_hms,
                    utc_iso=utc_iso,
                    utc_dt=utc_dt,
                    log_label=log_label,
                    gt_label=gt_label,
                    pred_label=pred_label,
                    pred_proba=pred_proba,
                    out_dir=out_dir,
                )
                print(f"[{saved}] {utc_iso} | {band_name} | LO={flo/1e6:.3f} MHz | "
                      f"Plan={log_label} | GT={gt_label} | Pred={pred_label} -> {out_path}")

                # Advance this LO's next slot
                next_sample_t_by_lo[flo] = ns + timedelta(seconds=SAMPLE_PERIOD_SEC)

                if saved >= MAX_SAMPLES:
                    print("Reached MAX_SAMPLES; stopping.")
                    return

    print(f"Done. Saved {saved} spectrograms (all bands).")

if __name__ == "__main__":
    main()
