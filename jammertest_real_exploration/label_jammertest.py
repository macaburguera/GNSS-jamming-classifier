# sbf_iq_perblock_loglabel_predict_30s.py
# Sample one IQ block every 30 s from a Septentrio SBF, label with jammertest code, extract the
# SAME features as data_preparation.py, predict with a .joblib model (class NAMES), and compute
# accuracy + confusion matrix on blocks with reliably mapped GT.
#
# >>> Edit USER VARIABLES below and run:  python sbf_iq_perblock_loglabel_predict_30s.py

from pathlib import Path
import re, json, csv, math
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, welch, spectrogram
from scipy import ndimage, stats
from scipy.stats import kurtosis, skew, entropy
from joblib import load as joblib_load
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sbf_parser import SbfParser
from datetime import datetime, timedelta, timezone

# ============================ USER VARIABLES ============================
SBF_PATH   = r"D:\datasets\Jammertest2023_Day1\Altus01 - 5m\alt01002.sbf"
OUT_DIR    = r"D:\datasets\Jammertest2023_Day1\plots\alt01002_predicted_30s"
LOGBOOK_PATH = r"D:\datasets\Jammertest2023_Day1\Testlog 23.09.18.txt"
MODEL_PATH = r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\1. DTU\4. Fall 2025\gnss jamming\databases\artifacts\xgb_run_20251104_224603\xgb_20251104_224815\xgb_trainval.joblib"

# Local test date/time (for logbook parsing)
LOCAL_DATE       = "2023-09-18"   # date of the test in LOCAL time
LOCAL_UTC_OFFSET = 2.0            # LOCAL - UTC (e.g. CEST=+2)

# Sampling policy: keep first block at/after each boundary
SAVE_EVERY_SEC = 30.0             # 30-second cadence (UTC)

# Pre-STFT decimation for plotting/features
DECIM = 1

# Spectrogram appearance (the “good” settings you liked)
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20

# Output & runtime
CHUNK_BYTES = 1_000_000
SAVE_IMAGES = True
SAVE_PER_SAMPLE_CSV = True
SUMMARY_JSON = True

# Names used when your model exposes integer classes (0..5)
MODEL_CLASS_NAMES = ["NoJam", "SingleAM", "SingleChirp", "SingleFM", "DME", "NB"]

# STRICT mapping from jammertest logcode -> your model class names (only when certain).
# Everything else stays None (Unknown GT) but is still predicted/logged.
LOG_TO_MODEL: Dict[str, Optional[str]] = {
    "NO JAM": "NoJam",
    "h1.1":   "NB",           # Log says: High power, narrow band
    # Uncomment/adjust the lines below once you verify them in the jammertest plan/PDFs:
    "u1.1": "SingleChirp",
    "s1.2": "SingleChirp",
    "s2.1": "SingleChirp",     # or "DB" if that’s the actual label in your trained model
    "h3.1": "SingleChirp",
    "UNKNOWN/CONFUSION": None,
    "UNKNOWN": None,
}
# =======================================================================

# ---------------------- time helpers -----------------------
EPS = 1e-12
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (stable since 2017-01-01)
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
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw  # BBSamples TOW often in ms
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

# ---------------------- SBF decode -------------------------
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

# ---------------------- logbook parsing --------------------
def parse_plaintext_logbook(path: str, local_date: str, local_utc_offset_hours: float) -> List[Interval]:
    """
    Lines like:
      '16:00 - Test was started - no jamming'
      '16:05 - Jammer u1.1 was turned on'
      ...
    Returns UTC intervals with labels: 'NO JAM', 'u1.1', 's1.2', 'h1.1', 'h3.1', 's2.1', 'UNKNOWN/CONFUSION', etc.
    """
    time_re = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*-\s*(.+)$')
    events = []
    base = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    offs = timedelta(hours=float(local_utc_offset_hours))

    def label_from_text(txt: str) -> str:
        t = txt.strip().lower()
        if ("no jamming" in t) or ("turned off" in t and "confusion" not in t):
            return "NO JAM"
        if "confusion" in t:
            return "UNKNOWN/CONFUSION"
        m = re.search(r"jammer\s+([^\s]+)\s+was turned on", txt, flags=re.IGNORECASE)
        if m: return m.group(1).strip()
        m2 = re.search(r"jammer\s+([^\s]+)\s+turned on", txt, flags=re.IGNORECASE)
        if m2: return m2.group(1).strip()
        return "UNKNOWN"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line)
            if not m: continue
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
    if log_label is None:
        return None
    key = log_label if log_label in LOG_TO_MODEL else log_label.lower()
    return LOG_TO_MODEL.get(key, None)

# ====================== FEATURE EXTRACTOR ======================
# (Same logic as your data_preparation.py — trimmed comments)
WELCH_NPERSEG = 1024
WELCH_OVERLAP = 256
MAX_LAG_S     = 200e-6
INB_BW_HZ     = 2_000_000.0
ENV_FFT_BAND  = (30.0, 7_000.0)
CHIRP_TARGET_SLICE_S = 0.125e-3
CHIRP_MIN_SLICES     = 6
CHIRP_MAX_SLICES     = 24
CYC_ALPHA1_HZ = 1.023e6
CYC_ALPHA2_HZ = 2.046e6

def fast_autocorr_env(z: np.ndarray, fs: float, max_lag_s: float):
    env = np.abs(z).astype(np.float64)
    env -= env.mean()
    n = int(1 << int(np.ceil(np.log2(2 * env.size - 1))))
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
        return 0.0, 0.0, 0.0, 3.0
    phi = np.unwrap(np.angle(z))
    inst_f = np.diff(phi) * fs / (2*np.pi)
    if inst_f.size > 8:
        lo, hi = np.percentile(inst_f, [1, 99])
        inst_f = np.clip(inst_f, lo, hi)
    t = np.arange(inst_f.size, dtype=np.float64) / fs
    slope = float(np.polyfit(t, inst_f, 1)[0]) if inst_f.size >= 2 else 0.0
    return float(np.mean(inst_f)), float(np.std(inst_f)), slope, float(stats.kurtosis(inst_f, fisher=False, bias=False))

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
    idx, _ = stats.find_peaks(Pxx, prominence=prom)  # fallback if no scipy.signal.find_peaks
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
        mu = float(np.sum(f1 * (Pxx / (Pxx.sum() + EPS))))
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
    c1 = cyclo_lag_corr(z, L1)
    c2 = cyclo_lag_corr(z, L2)
    return c1, c2

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
                                nperseg=512, noverlap=256,
                                detrend=False, return_onesided=False,
                                scaling="density", mode="psd")
        if Sxx.ndim != 2 or Sxx.shape[1] < 4:
            return 0.0, 0.0
        sk = stats.kurtosis(Sxx, axis=1, fisher=False, bias=False)
        sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)
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

def extract_features(iq: np.ndarray, fs: float) -> np.ndarray:
    iq = iq.astype(np.complex64, copy=False)

    env_raw = np.abs(iq).astype(np.float64)
    pre_rms = float(np.sqrt(np.mean(np.abs(iq)**2)) + EPS)
    crest_env = float(env_raw.max() / (env_raw.mean() + EPS))
    env_std = float(env_raw.std() + EPS)
    kurt_env = float(np.mean(((env_raw - env_raw.mean())/max(env_std, EPS))**4))

    f0, Pxx0 = welch(iq, fs=fs, window="hann", nperseg=WELCH_NPERSEG,
                     noverlap=WELCH_OVERLAP, return_onesided=False, scaling="density")
    order0 = np.argsort(f0); f0, Pxx0 = f0[order0], np.maximum(Pxx0[order0], EPS)
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
    order = np.argsort(f1); f, Pxx = f1[order], np.maximum(Pxx[order], EPS)
    Pxx /= (Pxx.sum() + EPS)
    mu, stdv = spectral_moments(f, Pxx)
    flat = spectral_flatness(Pxx)
    roll = spectral_rolloff(f, Pxx, 0.95)
    kmax = int(np.argmax(Pxx)); fmax = float(f[kmax]); pmax = float(Pxx[kmax])
    feats += [mu, stdv, flat, roll, fmax, pmax]

    edges = np.linspace(-fs/2, fs/2, 9)
    feats += bandpowers(f, Pxx, list(zip(edges[:-1], edges[1:]))).tolist()

    f_mean, f_std, f_slope, f_kurt = inst_freq_stats(z, fs)
    cep_env = cepstral_peak_env(z, fs, 2e-4, 5e-3)
    pulse_cnt, duty = dme_pulse_proxy(z, fs)
    sal = nb_peak_salience(f, Pxx, top_k=5)
    feats += [f_mean, f_std, f_slope, f_kurt, cep_env, pulse_cnt, duty, sal]

    def find_peaks_local(pxx):
        from scipy.signal import find_peaks
        return find_peaks(pxx, prominence=0.03 * float(np.max(pxx) + EPS))[0]

    try:
        idx = find_peaks_local(Pxx)
        if idx.size < 2:
            pk_cnt, sp_med, sp_std = float(idx.size), 0.0, 0.0
        else:
            freqs = f[idx]; spac = np.diff(np.sort(freqs))
            pk_cnt, sp_med, sp_std = float(idx.size), float(np.median(spac)), float(np.std(spac))
    except Exception:
        pk_cnt, sp_med, sp_std = 0.0, 0.0, 0.0

    mi, env_f, env_pk = am_envelope_features(z, fs)
    slices = choose_chirp_slices(len(z), fs)
    ch_slope, ch_r2 = chirp_slope_proxy(z, fs, slices)
    feats += [pk_cnt, sp_med, sp_std, mi, env_f, env_pk, ch_slope, ch_r2]

    c1, c2 = cyclo_proxies(z, fs)
    c40_mag, c42_mag = cumulants_c40_c42(z)
    sk_mean, sk_max = spectral_kurtosis_stats(z, fs)
    tkeo_mean = tkeo_env_mean(z)
    feats += [c1, c2, c40_mag, c42_mag, sk_mean, sk_max, tkeo_mean]

    v = np.asarray(feats, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0
    return v

# ---------------------- plotting --------------------------
def plot_and_save(block_idx, x, fs, wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt,
                  log_label, gt_label, pred_label, pred_proba, out_dir,
                  nperseg=NPERSEG, noverlap=NOVERLAP, dpi=120,
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
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------- prediction helpers -----------------
def normalize_model_labeler(model):
    """
    Returns (predict_fn, classes_order, to_name):
      - predict_fn(x)-> (pred_name, proba_dict)
      - classes_order: list[str] (names used in CM/report)
      - to_name: function converting raw yhat to str name
    """
    # Try to discover classes_
    classes_attr = getattr(model, "classes_", None)
    to_name = None
    if classes_attr is None:
        # Assume pipeline; unwrap last estimator
        try:
            est = model.named_steps.get("clf") or model.named_steps.get("classifier")
            classes_attr = getattr(est, "classes_", None)
        except Exception:
            classes_attr = None

    if classes_attr is not None and len(classes_attr) > 0:
        if np.issubdtype(np.array(classes_attr).dtype, np.integer):
            # map ints via USER list
            idx_to_name = {int(i): MODEL_CLASS_NAMES[int(i)] for i in range(len(MODEL_CLASS_NAMES))}
            to_name = lambda y: idx_to_name.get(int(y), str(y))
            classes_order = [idx_to_name.get(int(i), str(i)) for i in classes_attr]
        else:
            # already strings
            to_name = lambda y: str(y)
            classes_order = [str(c) for c in classes_attr]
    else:
        # Fall back to USER names; assume estimator returns ints
        to_name = lambda y: MODEL_CLASS_NAMES[int(y)] if isinstance(y, (int, np.integer)) and 0 <= int(y) < len(MODEL_CLASS_NAMES) else str(y)
        classes_order = MODEL_CLASS_NAMES[:]

    def predict_fn(feat_vec: np.ndarray):
        try:
            yhat = model.predict([feat_vec])[0]
            name = to_name(yhat)
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba([feat_vec])[0]
                    # align to classes_order length if possible
                    raw_classes = getattr(model, "classes_", None) or getattr(getattr(model, "named_steps", {}), "clf", None)
                    if getattr(model, "classes_", None) is not None:
                        cls_raw = list(model.classes_)
                        # Map raw class labels to names
                        names = [to_name(c) for c in cls_raw]
                        proba = {n: float(p) for n, p in zip(names, probs)}
                except Exception:
                    proba = None
            return name, proba
        except Exception:
            return None, None

    return predict_fn, classes_order, to_name

# ---------------------- main flow --------------------------
def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = joblib_load(MODEL_PATH)
    predict_fn, model_class_names, _ = normalize_model_labeler(model)
    print("Model class names:", model_class_names)

    # Logbook → intervals
    intervals = parse_plaintext_logbook(LOGBOOK_PATH, LOCAL_DATE, LOCAL_UTC_OFFSET)
    print(f"Loaded {len(intervals)} intervals from logbook.")
    for a,b,lbl in intervals:
        print(f"  {a.strftime('%H:%M:%S')}Z → {b.strftime('%H:%M:%S')}Z : {lbl}")

    # Metrics accumulators
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
                if x is None: continue
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

                log_label = label_for_time(intervals, utc_dt)
                gt_label  = map_log_to_model(log_label)

                # features + prediction
                feats = extract_features(x, fs)
                pred_label, pred_proba = predict_fn(feats)

                # accumulate metrics when GT is known (strict)
                if gt_label is not None and pred_label is not None:
                    y_true.append(gt_label)
                    y_pred.append(pred_label)

                # save plot
                if SAVE_IMAGES:
                    _ = plot_and_save(
                        block_idx=block_i, x=x, fs=fs,
                        wnc=wnc, tow_s=tow_s, tow_hms=tow_hms,
                        utc_hms=utc_hms, utc_iso=utc_iso, utc_dt=utc_dt,
                        log_label=log_label, gt_label=gt_label,
                        pred_label=pred_label, pred_proba=pred_proba,
                        out_dir=out_dir,
                        nperseg=NPERSEG, noverlap=NOVERLAP,
                        remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB
                    )
                    saved += 1
                    if saved % 200 == 0:
                        print(f"Saved {saved} figures...")

                rows.append({
                    "block_idx": block_i,
                    "utc_iso": utc_iso,
                    "gps_week": int(wnc),
                    "tow_s": float(tow_s),
                    "log_label": log_label,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "fs_hz": float(fs),
                })
                next_save_t = next_save_t + timedelta(seconds=SAVE_EVERY_SEC)

    # ---- metrics (strict on known GT)
    print("\n=== METRICS (strict, only mapped GT) ===")
    if len(y_true) == 0:
        print("No blocks with mapped ground-truth. Edit LOG_TO_MODEL once you validate the plan/PDFs.")
    else:
        labels_used = sorted(set(y_true) | (set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_used)
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion matrix (labels order):", labels_used)
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=labels_used, zero_division=0))

        # Plot CM
        fig = plt.figure(figsize=(1.2*len(labels_used), 1.0*len(labels_used)))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(labels_used))); ax.set_xticklabels(labels_used, rotation=45, ha="right")
        ax.set_yticks(range(len(labels_used))); ax.set_yticklabels(labels_used)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(OUT_DIR) / "confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        with open(Path(OUT_DIR) / "metrics.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Accuracy: {acc:.6f}\n")
            fh.write(f"Labels: {labels_used}\n")
            fh.write("Confusion matrix:\n")
            for r in cm:
                fh.write(",".join(map(str, r)) + "\n")
            fh.write("\nClassification report:\n")
            fh.write(classification_report(y_true, y_pred, labels=labels_used, zero_division=0))

    # per-sample CSV
    if SAVE_PER_SAMPLE_CSV and rows:
        csv_path = Path(OUT_DIR) / "samples_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote per-sample log to: {csv_path}")

    if SUMMARY_JSON:
        js = {
            "sbf_path": SBF_PATH,
            "model_path": MODEL_PATH,
            "class_names_model": MODEL_CLASS_NAMES,
            "log_to_model_mapping_used": LOG_TO_MODEL,
            "n_images_saved": saved if SAVE_IMAGES else 0,
            "save_every_sec": SAVE_EVERY_SEC,
            "decim": DECIM,
        }
        with open(Path(OUT_DIR) / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(js, fh, indent=2)
        print("Wrote summary.json")

if __name__ == "__main__":
    main()
