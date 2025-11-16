# data_preparation.py
"""
Feature extraction for GNSS+jamming IQ .mat files (Jammertest-style).
- Expects per-split folders: BASE/TRAIN|VAL|TEST/<Class>/*.mat
- Each .mat must have: GNSS_plus_Jammer_awgn (complex column vector)
- Optional: meta struct with fields {CNR_dBHz or CNo_dBHz, JSR_dB, ...}
- Saves NPZ caches for train / val / test under ./artifacts/prep_run_*
"""

from pathlib import Path
import argparse, time, json
import numpy as np
import scipy.io as sio
import h5py
from scipy.signal import welch, find_peaks, spectrogram
from scipy import ndimage

# ------------------ CLI / Defaults ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract GNSS+jam features into NPZ caches.")
    p.add_argument("--base", type=str,
                   default=r"D:\datasets\maca_gen\datasets_jammertest",
                   help="Root containing TRAIN / VAL / TEST folders.")
    p.add_argument("--var", type=str, default="GNSS_plus_Jammer_awgn",
                   help="MAT variable with IQ (complex column vector).")
    p.add_argument("--fs", type=float, default=60_000_000.0,
                   help="Sampling rate (Hz). Jammertest tiles use 62.5 MHz by default.")
    p.add_argument("--out", type=str, default="../artifacts/jammertest_sim",
                   help="Artifacts root; NPZs saved under out/prep_run_YYYYmmdd_HHMMSS")
    p.add_argument("--classes", type=str,
                   default="NoJam,Chirp,NB,CW,WB,FH",
                   help="Comma-separated class names (= subfolder names).")
    p.add_argument("--dry_per_class", type=int, default=None,
                   help="Cap #files per class for a quick run. None = use all.")
    p.add_argument("--debug", action="store_true", help="Verbose timing (per file).")
    return p.parse_args()

# ------------------ Tunables ------------------
WELCH_NPERSEG = 256
WELCH_OVERLAP = 128
MAX_LAG_S     = 200e-6
INB_BW_HZ     = 2_000_000.0
ENV_FFT_BAND  = (30.0, 7_000.0)
EPS           = 1e-20

CHIRP_TARGET_SLICE_S = 0.125e-3
CHIRP_MIN_SLICES     = 6
CHIRP_MAX_SLICES     = 24

CYC_ALPHA1_HZ = 1.023e6
CYC_ALPHA2_HZ = 2.046e6

STFT_NPERSEG  = 128
STFT_NOVERLAP = 96
STFT_NFFT     = 128

# ------------------ FS helpers ------------------
def now_run_dir(root: Path) -> Path:
    t = time.strftime("%Y%m%d_%H%M%S")
    out = root / f"prep_run_{t}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _h5_find_by_name(h5obj, target):
    for k, item in h5obj.items():
        if k == target:
            return item
        if isinstance(item, h5py.Group):
            hit = _h5_find_by_name(item, target)
            if hit is not None:
                return hit
    return None

def load_mat_var(path: Path, varname: str):
    """Return numpy array for `varname` from MATLAB v7 or v7.3 file."""
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        if varname not in m:
            raise KeyError
        return np.asarray(m[varname])
    except Exception:
        with h5py.File(path, "r") as f:
            node = f.get(varname, None) or _h5_find_by_name(f, varname)
            if node is None:
                raise KeyError(f"'{varname}' not found in {Path(path).name}")
            if isinstance(node, h5py.Dataset):
                return np.asarray(node[()])
            if isinstance(node, h5py.Group):
                keys = {k.lower(): k for k in node.keys()}
                if "real" in keys and "imag" in keys:
                    return np.asarray(node[keys["real"]][()]) + 1j*np.asarray(node[keys["imag"]][()])
                for v in node.values():
                    if isinstance(v, h5py.Dataset):
                        return np.asarray(v[()])
            raise TypeError(f"Unsupported HDF5 node for '{varname}' in {Path(path).name}")

def load_meta_dict(path: Path):
    """Best-effort read of `meta` as a dict (works for v7 and v7.3)."""
    # v7 files
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        meta = m.get("meta", None)
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass
    # v7.3 fallback
    out = {}
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                return out
            g = f["meta"]
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    arr = np.asarray(v[()])
                    if np.size(arr) == 1:
                        out[k] = float(np.ravel(arr)[0]) if np.isrealobj(arr) else arr
                    else:
                        out[k] = arr
    except Exception:
        pass
    return out

def to_complex_1d(x):
    x = np.asarray(x)
    if np.isrealobj(x) and x.ndim == 2 and x.shape[1] == 2:
        x = x[:,0] + 1j*x[:,1]
    if getattr(x, "dtype", None) is not None and x.dtype.names:
        names = {n.lower(): n for n in x.dtype.names}
        r = names.get("real") or names.get("re") or names.get("r")
        i = names.get("imag") or names.get("im") or names.get("i")
        if r and i:
            x = x[r] + 1j*x[i]
    return np.array(x).ravel(order="F").astype(np.complex64)

def load_iq_meta(path: Path, var_name: str):
    iq_raw = load_mat_var(path, var_name)
    z = to_complex_1d(iq_raw)
    meta = load_meta_dict(path)

    cnr = np.nan
    for k in ("CNR_dBHz", "CNo_dBHz", "CNR_dB", "CNo"):
        if isinstance(meta, dict) and k in meta:
            try:
                cnr = float(np.asarray(meta[k]).ravel()[0])
                break
            except Exception:
                pass

    jsr = np.nan
    if isinstance(meta, dict) and "JSR_dB" in meta:
        try:
            jsr = float(np.asarray(meta["JSR_dB"]).ravel()[0])
        except Exception:
            pass

    return z, jsr, cnr

def iter_split_dirs(base: Path, classes):
    p_train = base / "TRAIN"
    p_val   = base / "VAL"
    p_test  = base / "TEST"
    if p_train.exists() or p_val.exists() or p_test.exists():
        return {"TRAIN": p_train, "VAL": p_val, "TEST": p_test}, classes
    return {
        "TRAIN": base / "Image_training_database",
        "VAL":   base / "Image_validation_database",
        "TEST":  base / "Image_testing_database",
    }, classes

def iter_files_split(base_dir: Path, classes, cap_per_class=None):
    for label, cls in enumerate(classes):
        d = base_dir / cls
        files = sorted(d.glob("*.mat")) if d.exists() else []
        if cap_per_class is not None:
            files = files[:cap_per_class]
        yield cls, label, files

# ------------------ Safe stats helpers (no SciPy warnings) ------------------
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

# ------------------ Feature helpers (base) ------------------
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
    # Robust, warning-free kurtosis across time per frequency bin.
    try:
        f, t, Sxx = spectrogram(z, fs=fs, window="hann",
                                nperseg=256, noverlap=128,
                                detrend=False, return_onesided=False,
                                scaling="density", mode="psd")
        if Sxx.ndim != 2 or Sxx.shape[1] < 4:
            return 0.0, 0.0
        # compute population kurtosis per-bin safely
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

# ------------------ NEW helpers ------------------
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

# ------------------ Feature vector ------------------
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

def extract_features(iq: np.ndarray, fs: float):
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
        order = np.argsort(f_st)
        f_st = f_st[order]; Sxx_st = np.maximum(Sxx_st[order, :], EPS)
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

    return np.asarray(feats, float)

# ------------------ Runner ------------------
def main():
    args = parse_args()
    base = Path(args.base)
    out_root = Path(args.out)
    out_dir = now_run_dir(out_root)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not base.exists():
        print(f"[WARN] Base path does not exist: {base}")
    else:
        print(f"[INFO] Using base: {base}")

    splits, classes = iter_split_dirs(base, classes)
    meta_summary = {
        "fs_hz": args.fs,
        "var_name": args.var,
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "welch_nperseg": WELCH_NPERSEG,
        "welch_overlap": WELCH_OVERLAP,
        "env_fft_band_hz": ENV_FFT_BAND,
        "max_lag_s": MAX_LAG_S,
        "oob_inband_half_bw_hz": INB_BW_HZ,
        "chirp_target_slice_s": CHIRP_TARGET_SLICE_S,
        "chirp_slices_minmax": [CHIRP_MIN_SLICES, CHIRP_MAX_SLICES],
        "cyclo_alphas_hz": [CYC_ALPHA1_HZ, CYC_ALPHA2_HZ],
        "stft_params": {"nperseg": STFT_NPERSEG, "noverlap": STFT_NOVERLAP, "nfft": STFT_NFFT},
    }

    for split_name, split_dir in splits.items():
        X, y, jsr, cnr, paths = [], [], [], [], []
        print(f"[{split_name}] scanning {split_dir}")
        for cls, label, files in iter_files_split(split_dir, classes, args.dry_per_class):
            print(f"  {cls}: {len(files)} files")
            for p in files:
                try:
                    iq, j, c = load_iq_meta(p, args.var)
                    feats = extract_features(iq, args.fs)
                    X.append(feats); y.append(label); jsr.append(j); cnr.append(c); paths.append(str(p))
                except Exception as e:
                    print(f"  [WARN] {p.name} -> {e}")
        if not X:
            print(f"[{split_name}] no files; skipping.")
            continue
        X = np.vstack(X)
        y = np.asarray(y, int)
        jsr = np.asarray(jsr, float)
        cnr = np.asarray(cnr, float)
        np.savez_compressed(
            out_dir / f"{split_name.lower()}_features.npz",
            X=X, y=y,
            jsr=jsr, cnr=cnr,
            class_names=np.array(classes, dtype=object),
            feature_names=np.array(FEATURE_NAMES, dtype=object),
            fs_hz=np.array([args.fs], dtype=np.float64),
            paths=np.array(paths, dtype=object),
        )
        print(f"[{split_name}] saved -> {out_dir / f'{split_name.lower()}_features.npz'}  shape={X.shape}  (n_features={X.shape[1]})")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2)
    print(f"Done. Artifacts under: {out_dir}")

if __name__ == "__main__":
    main()
