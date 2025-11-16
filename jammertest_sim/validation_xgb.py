# validation.py
"""
Validate the SIM-trained XGB model on real Jammertest SBF IQ data.

CHANGES (2025-03-31):
- Canonicalization: enforce a single, consistent label namespace end-to-end.
- Metrics: confusion matrix pinned to fixed class order; added GT/Pred counts.
- NoJam veto (optional): conservative override to recover likely NoJam cases
  when model confidence is low. Thresholds are tunable and logged.

Run:
  python sbf_iq_perblock_loglabel_predict_30s_new.py
"""

from pathlib import Path
import re, json, csv
from typing import List, Tuple, Optional, Dict
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from scipy.signal import stft, welch, spectrogram, find_peaks
from scipy import ndimage

from joblib import load as joblib_load
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from datetime import datetime, timedelta, timezone
from collections import Counter

# You must have a local parser for SBF BBSamples similar to your earlier script.
# It should yield tuples: (block_name, infos_dict), where infos contains:
#   - "Samples": raw interleaved int8 I/Q buffer (len = 2*N)
#   - "N": number of complex samples
#   - "SampleFreq": sampling frequency in Hz (float)
#   - "WNc": GPS week number (int)
#   - "TOW": GPS time-of-week seconds or milliseconds (float)
from sbf_parser import SbfParser

# ============================ USER VARIABLES ============================
SBF_PATH   = r"D:\datasets\Jammertest2023_Day1\Altus06 - 150m\alt06001.sbf"
OUT_DIR    = r"D:\datasets\Jammertest2023_Day1\plots\alt06001_predicted_30s_SIMXGB_modNB"
LOGBOOK_PATH = r"D:\datasets\Jammertest2023_Day1\Testlog 23.09.18.txt"

# ==> Point to YOUR new XGB model trained on 78 features & new classes:
MODEL_PATH = r"..\artifacts\jammertest_sim\xgb_run_20251116_213143\xgb_20251116_213206\xgb_trainval.joblib"

# Local test date/time (for logbook parsing)
LOCAL_DATE       = "2023-09-18"   # date of the test in LOCAL time
LOCAL_UTC_OFFSET = 2.0            # LOCAL - UTC (e.g., CEST=+2)

# Sampling policy: keep first block at/after each boundary (UTC)
SAVE_EVERY_SEC = 30.0             # 30-second cadence

# Optional decimation for features & plots (keep 1 for exact original fs)
DECIM = 1

# Spectrogram appearance (robust settings for short IQ bursts)
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20
DPI_FIG = 140

# Save options
CHUNK_BYTES = 1_000_000
SAVE_IMAGES = True
SAVE_PER_SAMPLE_CSV = True
SUMMARY_JSON = True
DEBUG_PRINT_SAMPLE_LABELS = False  # set True to verbose-print per-sample labels

# New model class names (order must match your training labels)
MODEL_CLASS_NAMES = ["NoJam", "Chirp", "NB", "CW", "WB", "FH"]

# Canonicalization dictionary (lowercase -> canonical)
CANON = {c.lower(): c for c in MODEL_CLASS_NAMES}

def canon(name: Optional[str]) -> Optional[str]:
    """Return canonical label if known; pass through unknowns as-is; None stays None."""
    if name is None:
        return None
    s = str(name).strip()
    return CANON.get(s.lower(), s)

# ----------------- Mapping: Jammertest logcode -> Model class -----------------
# NOTE: Mapping semantics kept per your note:
# - All listed jammer codes → "Chirp" EXCEPT "h1.1" → "NB"
# - "no jam" / "off" / "no jamming" → "NoJam"
# Keys are stored lowercase; parser text is normalized to lowercase before lookup.
LOG_TO_MODEL: Dict[str, Optional[str]] = {
    "no jam": "NoJam",
    "off": "NoJam",
    "no jamming": "NoJam",

    # Narrowband high-power case:
    "h1.1": "NB",

    # Chirp-like (all of these map to Chirp)
    "h1.2": "Chirp",
    "u1.1": "Chirp",
    "u1.2": "Chirp",
    "s1.2": "Chirp",
    "h3.1": "Chirp",
    "s2.1": "Chirp",

    # Unknown/confusion entries do not contribute GT
    "unknown/confusion": None,
    "unknown": None,
}
# ============================================================================

# ====================== TIME / LOGBOOK HELPERS ======================
EPS = 1e-20
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds

Interval = Tuple[datetime, datetime, str]  # (UTC start, UTC end, label)

def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600); m = int((tsec % 3600) // 60)
    s = tsec - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def extract_time_labels(infos):
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0))
    # Some SBFs encode TOW in ms; detect & convert
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

def parse_plaintext_logbook(path: str, local_date: str, local_utc_offset_hours: float) -> List[Interval]:
    """
    Lines example:
      '16:00 - Test was started - no jamming'
      '16:05 - Jammer u1.1 was turned on'
    Returns UTC intervals with labels: 'NO JAM', 'u1.1', 's1.2', etc. (raw text codes)
    """
    time_re = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*[-–—]\s*(.+)$')

    events = []
    base = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    offs = timedelta(hours=float(local_utc_offset_hours))

    def label_from_text(txt: str) -> str:
        # Normalize spaces and case for robust checks
        t_norm = " ".join(txt.strip().split())
        tl = t_norm.lower()

        # Clear "off" / "no jamming" cases first
        if ("no jamming" in tl) or (re.search(r"\bturned\s+off\b", tl) and "confusion" not in tl):
            return "NO JAM"
        if "confusion" in tl:
            return "UNKNOWN/CONFUSION"

        # Pattern: Jammer <CODE> [optional (...) blurb] (was)? turned on
        # - <CODE> allows letters/numbers/.- (e.g., s2.1, h1.1, u1-xyz)
        pat = re.compile(
            r"jammer\s+([A-Za-z0-9.\-]+)"
            r"(?:\s*\([^)]*\))?"
            r"\s+(?:was\s+)?turned\s+on\b",
            flags=re.IGNORECASE
        )
        m = pat.search(t_norm)
        if m:
            return m.group(1).lower()

        # Fallback: Jammer <CODE> ... on   (avoid misfiring on 'turned off')
        if "turned off" not in tl:
            m2 = re.search(r"jammer\s+([A-Za-z0-9.\-]+)\b.*\bon\b", t_norm, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).lower()

        return "UNKNOWN"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line)
            if not m: 
                continue
            hh, mm, rest = m.groups()
            local_dt = base.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            lbl = label_from_text(rest)
            events.append((local_dt, lbl))

    events.sort(key=lambda e: e[0])
    intervals: List[Interval] = []
    for i in range(len(events)):
        start_local, lbl = events[i]
        end_local = events[i+1][0] if i+1 < len(events) else (start_local + timedelta(minutes=60))
        intervals.append((start_local - offs, end_local - offs, lbl))
    return intervals

def label_for_time(intervals: List[Interval], t_utc: datetime) -> Optional[str]:
    for a,b,label in intervals:
        if a <= t_utc < b:
            return label
    return None

def map_log_to_model(log_label: Optional[str]) -> Optional[str]:
    """
    Map raw log label (e.g., 'NO JAM', 'u1.1', 'h1.1') to canonical model label.
    Preserves your mapping semantics. Unknown/None -> None (excluded from GT).
    """
    if log_label is None:
        return None
    key = log_label.strip().lower()
    mapped = LOG_TO_MODEL.get(key, None)
    return canon(mapped) if mapped is not None else None

# ====================== SBF IQ DECODING ======================
def decode_bbsamples_iq(infos):
    if "Samples" not in infos or "N" not in infos:
        return None, None
    buf = infos["Samples"]; N = int(infos["N"])
    arr = np.frombuffer(buf, dtype=np.int8)
    if arr.size != 2 * N:
        return None, None
    I = arr[0::2].astype(np.float32) / 128.0
    Q = arr[1::2].astype(np.float32) / 128.0
    x = I + 1j * Q
    fs = float(infos.get("SampleFreq", 1.0))
    return x, fs

# ====================== 78-FEATURE EXTRACTOR (1:1 with data_preparation.py) ======================
# Tunables (copied from your latest prep script)
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
    # Population kurtosis (Fisher=False). Returns 3.0 for constant/short arrays.
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
    inst_f = np.diff(phi) * fs / (2*np.pi)  # Hz
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
    rest = float(max(EPS, 1.0 - top))  # Pxx normalized
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
        # population kurtosis per frequency bin
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
    Sxx = Sxx[order, :]
    Sxx = np.maximum(Sxx, EPS)
    Sxxn = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + EPS)
    cent = np.sum(f[:, None] * Sxxn, axis=0)  # Hz
    if cent.size < 2:
        strong_bins_mean = float((Sxxn > 0.5*np.max(Sxxn, axis=0, keepdims=True)).mean())
        return 0.0, 0.0, 0.0, 0.0, strong_bins_mean
    dt = (STFT_NPERSEG - STFT_NOVERLAP) / fs
    dcent = np.diff(cent)
    dcent_dt = dcent / max(dt, 1/fs)
    mad = np.median(np.abs(dcent - np.median(dcent))) + 1e-6
    hop_thr = max(5e5, 6.0 * mad)  # Hz threshold for "hops" (~FH)
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
    # NEW 22
    + ["skewI","skewQ","kurtI","kurtQ","circularity_mag","circularity_phase_rad"]
    + ["spec_gini","env_gini","env_p95_over_p50","spec_symmetry_index","dc_notch_ratio","spec_peakiness_ratio"]
    + ["stft_centroid_std_Hz","stft_centroid_absderiv_med_Hzps",
       "stft_centroid_zcr_per_s","fh_hop_rate_per_s","strong_bins_mean"]
    + ["cyclo_halfchip_corr","cyclo_5chip_corr"]
    + ["chirp_curvature_Hzps2"]
    + ["dme_ipi_med_s","dme_ipi_std_s"]
)
PRE_RMS_IDX = FEATURE_NAMES.index("pre_rms")  # used by veto

def extract_features(iq: np.ndarray, fs: float) -> np.ndarray:
    iq = iq.astype(np.complex64, copy=False)

    # ---------- POWER-aware on raw ----------
    env_raw = np.abs(iq).astype(np.float64)
    pre_rms = float(np.sqrt(np.mean(np.abs(iq)**2)) + EPS)
    crest_env = float(env_raw.max() / (env_raw.mean() + EPS))
    env_std = float(env_raw.std() + EPS)
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

    # ---------- Normalize for shape ----------
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

    # ---------- Spectral shape (normalized) ----------
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

    edges = np.linspace(-fs/2, fs/2, 9)   # 8 equal bands
    feats += bandpowers(f, Pxx, list(zip(edges[:-1], edges[1:]))).tolist()

    # ---------- Instantaneous frequency ----------
    f_mean, f_std, f_slope, f_kurt, instf_dzcr = inst_freq_stats(z, fs)
    feats += [f_mean, f_std, f_slope, f_kurt, instf_dzcr]

    # ---------- Envelope/cepstrum/pulses & NB salience ----------
    cep_env = cepstral_peak_env(z, fs, 2e-4, 5e-3)
    pulse_cnt, duty = dme_pulse_proxy(z, fs)
    sal = nb_peak_salience(f, Pxx, top_k=5)
    feats += [cep_env, pulse_cnt, duty, sal]

    # ---------- Physics-driven ----------
    pk_cnt, sp_med, sp_std = nb_peaks_and_spacing(f, Pxx)
    mi, env_f, env_pk = am_envelope_features(z, fs)
    slices = choose_chirp_slices(len(z), fs)
    ch_slope, ch_r2 = chirp_slope_proxy(z, fs, slices)
    feats += [pk_cnt, sp_med, sp_std, mi, env_f, env_pk, ch_slope, ch_r2]

    # ---------- Cyclo / cumulants / SK / TKEO ----------
    c1, c2 = cyclo_proxies(z, fs)
    c40_mag, c42_mag = cumulants_c40_c42(z)
    sk_mean, sk_max = spectral_kurtosis_stats(z, fs)
    tkeo_mean = tkeo_env_mean(z)
    feats += [c1, c2, c40_mag, c42_mag, sk_mean, sk_max, tkeo_mean]

    # ---------- NEW 22 ----------
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
        chirp_curvature = float(2.0 * p2[0])  # Hz/s^2
    else:
        chirp_curvature = 0.0
    feats += [chirp_curvature]

    ipi_med, ipi_std = dme_ipi_stats(z, fs)
    feats += [ipi_med, ipi_std]

    v = np.asarray(feats, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0
    return v

# ====================== PLOTTING ======================
def plot_and_save(block_idx, x, fs, wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt,
                  log_label, gt_label, pred_label, pred_proba, out_dir,
                  nperseg=NPERSEG, noverlap=NOVERLAP, dpi=DPI_FIG,
                  remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB):
    if x.size < 8:
        return None
    xx = x - np.mean(x) if remove_dc else x

    nperseg_eff = min(int(nperseg), len(xx))
    noverlap_eff = min(int(noverlap), max(0, nperseg_eff - 1))

    f, t, Z = stft(xx, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                   return_onesided=False, boundary=None, padded=False)
    if t.size < 2:
        nperseg_eff = max(16, min(len(xx)//4, nperseg_eff))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = stft(xx, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                       return_onesided=False, boundary=None, padded=False)

    Z = np.fft.fftshift(Z, axes=0); f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    tt = np.arange(len(xx), dtype=np.float32) / fs
    I = np.real(xx); Q = np.imag(xx)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2,1,1)
    if t.size >= 2:
        pcm = ax1.pcolormesh(t, f, S_dB, shading="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax1, label="dB")
    else:
        im = ax1.imshow(S_dB, aspect="auto", origin="lower",
                        extent=[0.0, max(1.0/fs, nperseg_eff/fs), f[0], f[-1]],
                        vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax1, label="dB")

    jam_txt = f" | Jammer: {log_label}" if log_label else " | Jammer: (none)"
    gt_txt  = f" | GT: {gt_label}" if gt_label is not None else " | GT: Unknown"
    if pred_label is None:
        pred_txt = " | Pred: (model failed)"
    else:
        if isinstance(pred_proba, dict) and pred_label in pred_proba:
            pred_txt = f" | Pred: {pred_label} ({pred_proba[pred_label]:.2f})"
        else:
            pred_txt = f" | Pred: {pred_label}"

    title = (f"Spectrogram (BBSamples #{block_idx})  |  GPS week {wnc}  |  "
             f"TOW {tow_s:.3f}s ({tow_hms})  |  UTC {utc_hms}{jam_txt}{gt_txt}{pred_txt}\n"
             f"nperseg={nperseg_eff}, noverlap={noverlap_eff}")
    ax1.set_title(title)
    ax1.set_ylabel("Frequency [Hz]")

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(tt, I, linewidth=0.7, label="I")
    ax2.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.legend(loc="upper right")
    ax2.text(0.01, 0.02, utc_iso, transform=ax2.transAxes, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fname_log  = (log_label or "nolabel").replace(" ", "_").replace("/", "-")
    fname_pred = (pred_label or "nopred").replace(" ", "_")
    out_path = Path(out_dir) / f"spec_{utc_dt.strftime('%H%M%S')}_{fname_log}_GT-{gt_label or 'Unknown'}_PRED-{fname_pred}_blk{block_idx:06d}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ====================== MODEL NORMALIZATION ======================
def normalize_model_labeler(model):
    """
    Returns (predict_fn, classes_order, to_name):
      - predict_fn(x)-> (pred_name, proba_dict)
      - classes_order: list[str] (names used in CM/report)
      - to_name: function converting raw yhat to str name
    """
    classes_attr = getattr(model, "classes_", None)
    to_name = None
    if classes_attr is None:
        try:
            est = model.named_steps.get("clf") or model.named_steps.get("classifier")
            classes_attr = getattr(est, "classes_", None)
        except Exception:
            classes_attr = None

    if classes_attr is not None and len(classes_attr) > 0:
        # If stored labels are ints 0..5, map via MODEL_CLASS_NAMES
        arr = np.array(classes_attr)
        if np.issubdtype(arr.dtype, np.integer):
            idx_to_name = {int(i): MODEL_CLASS_NAMES[int(i)] for i in range(len(MODEL_CLASS_NAMES))}
            to_name = lambda y: idx_to_name.get(int(y), str(y))
            classes_order = [idx_to_name.get(int(i), str(i)) for i in arr]
        else:
            # Already strings
            to_name = lambda y: canon(y)
            classes_order = [canon(c) for c in classes_attr]
    else:
        # Fallback: assume estimator returns ints
        to_name = lambda y: MODEL_CLASS_NAMES[int(y)] if isinstance(y, (int, np.integer)) and 0 <= int(y) < len(MODEL_CLASS_NAMES) else str(y)
        classes_order = MODEL_CLASS_NAMES[:]

    def predict_fn(feat_vec: np.ndarray):
        try:
            yhat = model.predict([feat_vec])[0]
            name = canon(to_name(yhat))
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba([feat_vec])[0]
                    raw_classes = getattr(model, "classes_", None)
                    names = None
                    if raw_classes is not None:
                        if np.issubdtype(np.array(raw_classes).dtype, np.integer):
                            names = [MODEL_CLASS_NAMES[int(c)] for c in raw_classes]
                        else:
                            names = [canon(c) for c in raw_classes]
                    if names is None:
                        names = MODEL_CLASS_NAMES[:len(probs)]
                    proba = {canon(n): float(p) for n, p in zip(names, probs)}
                except Exception:
                    proba = None
            return name, proba
        except Exception:
            return None, None

    return predict_fn, classes_order, to_name

# ====================== NOJAM VETO (OPTIONAL) ======================
# These thresholds are conservative defaults; tune on your validation data.
USE_NOJAM_VETO = True
P_TOP_MIN       = 0.45   # if max class prob is below this, prediction is "uncertain"
P_NOJAM_MIN     = 0.45   # if NoJam prob is at/above this and we're uncertain, call NoJam
ENERGY_RMS_MAX  = 0.12   # if absolute power is very low and NoJam prob is non-trivial, call NoJam
P_NOJAM_LOWPOW  = 0.30   # NoJam prob needed in the low-power case

def apply_nojam_veto(pred_label: Optional[str],
                     pred_proba: Optional[Dict[str, float]],
                     feats_vec: np.ndarray) -> (Optional[str], bool, Dict[str, float]):
    """
    Optionally override the model prediction to 'NoJam' in two scenarios:
      (A) Uncertain prediction: top prob < P_TOP_MIN and p(NoJam) >= P_NOJAM_MIN
      (B) Very low power: pre_rms <= ENERGY_RMS_MAX and p(NoJam) >= P_NOJAM_LOWPOW
    Returns: (possibly-updated label, veto_applied_flag, proba_dict or {})
    """
    if not USE_NOJAM_VETO or pred_label is None or pred_proba is None:
        return pred_label, False, pred_proba or {}

    p_top = max(pred_proba.values()) if pred_proba else 0.0
    p_nj  = float(pred_proba.get("NoJam", 0.0))
    pre_rms_val = float(feats_vec[PRE_RMS_IDX]) if PRE_RMS_IDX is not None else 0.0

    veto = False
    # Scenario A: uncertain overall, decent NoJam probability
    if (p_top < P_TOP_MIN) and (p_nj >= P_NOJAM_MIN):
        pred_label = "NoJam"; veto = True
    # Scenario B: very low energy, modest NoJam probability
    elif (pre_rms_val <= ENERGY_RMS_MAX) and (p_nj >= P_NOJAM_LOWPOW):
        pred_label = "NoJam"; veto = True

    return pred_label, veto, {"p_top": p_top, "p_nojam": p_nj, "pre_rms": pre_rms_val, **pred_proba}

# ====================== MAIN ======================
def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = joblib_load(MODEL_PATH)
    predict_fn, model_class_names, _ = normalize_model_labeler(model)
    print("Model class names (from estimator):", model_class_names)
    print("Canonical class order used for metrics:", MODEL_CLASS_NAMES)

    # Logbook → intervals
    intervals = parse_plaintext_logbook(LOGBOOK_PATH, LOCAL_DATE, LOCAL_UTC_OFFSET)
    print(f"Loaded {len(intervals)} intervals from logbook.")
    for a,b,lbl in intervals:
        print(f"  {a.strftime('%H:%M:%S')}Z → {b.strftime('%H:%M:%S')}Z : {lbl}")

    # Metrics accumulators (strict: only mapped GT)
    y_true: List[str] = []
    y_pred: List[str] = []
    rows: List[dict] = []

    parser = SbfParser()
    block_i = -1
    saved = 0
    next_save_t: Optional[datetime] = None

    with open(SBF_PATH, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk: break
            for blk, infos in parser.parse(chunk):
                if blk != "BBSamples":
                    continue
                block_i += 1

                x, fs = decode_bbsamples_iq(infos)
                if x is None: 
                    continue
                if DECIM and DECIM > 1:
                    x = x[::DECIM]; fs = fs / DECIM

                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)

                # 30 s gating (UTC)
                if next_save_t is None:
                    stride = int(SAVE_EVERY_SEC)
                    floor = utc_dt.replace(second=(utc_dt.second // stride) * stride, microsecond=0)
                    if floor > utc_dt:
                        floor -= timedelta(seconds=stride)
                    next_save_t = floor + timedelta(seconds=stride)

                if utc_dt < next_save_t:
                    continue
                while utc_dt >= next_save_t + timedelta(seconds=SAVE_EVERY_SEC):
                    next_save_t += timedelta(seconds=SAVE_EVERY_SEC)

                # Labels (GT from logbook; canonicalized)
                log_label_raw = label_for_time(intervals, utc_dt)
                gt_label  = map_log_to_model(log_label_raw)  # canonical or None

                # Feature extraction + prediction
                feats = extract_features(x, fs)
                pred_label, pred_proba = predict_fn(feats)
                pred_label = canon(pred_label)

                # Optional NoJam veto
                veto_applied = False
                veto_meta = {}
                if pred_label is not None:
                    pred_label, veto_applied, veto_meta = apply_nojam_veto(pred_label, pred_proba, feats)

                # accumulate metrics when GT is known (strict)
                if gt_label is not None and pred_label is not None:
                    y_true.append(gt_label)
                    y_pred.append(pred_label)

                if DEBUG_PRINT_SAMPLE_LABELS:
                    print(f"[{utc_iso}] GT={gt_label} | Pred={pred_label} | Veto={veto_applied}")

                # save plot
                if SAVE_IMAGES:
                    _ = plot_and_save(
                        block_idx=block_i, x=x, fs=fs,
                        wnc=wnc, tow_s=tow_s, tow_hms=tow_hms,
                        utc_hms=utc_hms, utc_iso=utc_iso, utc_dt=utc_dt,
                        log_label=log_label_raw, gt_label=gt_label,
                        pred_label=pred_label, pred_proba=pred_proba,
                        out_dir=out_dir,
                        nperseg=NPERSEG, noverlap=NOVERLAP,
                        remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB
                    )
                    saved += 1
                    if saved % 200 == 0:
                        print(f"Saved {saved} figures...")

                # per-sample row (add veto/proba diagnostics)
                row = {
                    "block_idx": block_i,
                    "utc_iso": utc_iso,
                    "gps_week": int(wnc),
                    "tow_s": float(tow_s),
                    "log_label_raw": log_label_raw,
                    "gt_label": gt_label,
                    "pred_label_raw": canon(pred_label),   # after veto, canonical
                    "veto_applied": bool(veto_applied),
                    "fs_hz": float(fs),
                }
                # lightweight proba diagnostics (if available)
                if isinstance(pred_proba, dict):
                    row.update({
                        "p_NoJam": float(pred_proba.get("NoJam", 0.0)),
                        "p_Chirp": float(pred_proba.get("Chirp", 0.0)),
                        "p_NB": float(pred_proba.get("NB", 0.0)),
                        "p_CW": float(pred_proba.get("CW", 0.0)),
                        "p_WB": float(pred_proba.get("WB", 0.0)),
                        "p_FH": float(pred_proba.get("FH", 0.0)),
                    })
                if veto_meta:
                    row.update({
                        "veto_p_top": float(veto_meta.get("p_top", 0.0)),
                        "veto_p_nojam": float(veto_meta.get("p_nojam", 0.0)),
                        "veto_pre_rms": float(veto_meta.get("pre_rms", 0.0)),
                    })
                # always include pre_rms from features for tuning
                row["pre_rms"] = float(feats[PRE_RMS_IDX])
                rows.append(row)

                next_save_t = next_save_t + timedelta(seconds=SAVE_EVERY_SEC)

    # ---- metrics (strict on known GT)
    print("\n=== METRICS (strict, only mapped GT) ===")
    if len(y_true) == 0:
        print("No blocks with mapped ground-truth. Refine LOG_TO_MODEL using the official plan.")
    else:
        # Sanity histograms
        ct_true = Counter(y_true)
        ct_pred = Counter(y_pred)
        print("GT counts:", dict(ct_true))
        print("Pred counts:", dict(ct_pred))

        # Fixed canonical order so missing classes are obvious
        labels_for_metrics = MODEL_CLASS_NAMES[:]  # always full, fixed order

        cm = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion matrix (labels order):", labels_for_metrics)
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=labels_for_metrics, zero_division=0))

        # Plot CM
        fig = plt.figure(figsize=(1.1*len(labels_for_metrics)+2, 1.0*len(labels_for_metrics)+2))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(labels_for_metrics))); ax.set_xticklabels(labels_for_metrics, rotation=45, ha="right")
        ax.set_yticks(range(len(labels_for_metrics))); ax.set_yticklabels(labels_for_metrics)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(OUT_DIR) / "confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # Write textual metrics
        with open(Path(OUT_DIR) / "metrics.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Accuracy: {acc:.6f}\n")
            fh.write(f"Labels (fixed order): {labels_for_metrics}\n")
            fh.write("GT counts:\n")
            for k in labels_for_metrics:
                fh.write(f"  {k}: {ct_true.get(k, 0)}\n")
            fh.write("Pred counts:\n")
            for k in labels_for_metrics:
                fh.write(f"  {k}: {ct_pred.get(k, 0)}\n")
            fh.write("Confusion matrix (rows=True, cols=Pred):\n")
            for r in cm:
                fh.write(",".join(map(str, r)) + "\n")
            fh.write("\nClassification report:\n")
            fh.write(classification_report(y_true, y_pred, labels=labels_for_metrics, zero_division=0))

    # per-sample CSV
    if SAVE_PER_SAMPLE_CSV and rows:
        csv_path = Path(OUT_DIR) / "samples_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            # ensure stable field order
            fieldnames = ["block_idx","utc_iso","gps_week","tow_s",
                          "log_label_raw","gt_label","pred_label_raw","veto_applied",
                          "fs_hz","pre_rms",
                          "p_NoJam","p_Chirp","p_NB","p_CW","p_WB","p_FH",
                          "veto_p_top","veto_p_nojam","veto_pre_rms"]
            # include only those that appear
            present = set().union(*[set(r.keys()) for r in rows])
            fieldnames = [f for f in fieldnames if f in present] + [f for f in rows[0].keys() if f not in fieldnames]
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote per-sample log to: {csv_path}")

    if SUMMARY_JSON:
        js = {
            "sbf_path": SBF_PATH,
            "model_path": MODEL_PATH,
            "model_classes": MODEL_CLASS_NAMES,
            "log_to_model_mapping_used": LOG_TO_MODEL,
            "n_images_saved": saved if SAVE_IMAGES else 0,
            "save_every_sec": SAVE_EVERY_SEC,
            "decim": DECIM,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "spectrogram_plot": {
                "nperseg": NPERSEG, "noverlap": NOVERLAP,
                "vmin_db": VMIN_DB, "vmax_db": VMAX_DB
            },
            "veto": {
                "enabled": USE_NOJAM_VETO,
                "P_TOP_MIN": P_TOP_MIN,
                "P_NOJAM_MIN": P_NOJAM_MIN,
                "ENERGY_RMS_MAX": ENERGY_RMS_MAX,
                "P_NOJAM_LOWPOW": P_NOJAM_LOWPOW
            }
        }
        with open(Path(OUT_DIR) / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(js, fh, indent=2)
        print("Wrote summary.json")

if __name__ == "__main__":
    main()
