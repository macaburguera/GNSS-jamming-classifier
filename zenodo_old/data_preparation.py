# data_preparation.py
"""
Feature extraction for GNSS+jamming IQ .mat files (synthetic Option=1).
- Matches MATLAB generator outputs:
    var  : GNSS_plus_Jammer_awgn (complex column vector)
    meta : struct with fields CNR_dBHz, JSR_dB, band, jam_code, jam_name, jam_params, seed
- Saves: train_features.npz, val_features.npz, test_features.npz
"""

from pathlib import Path
import argparse, time, json
import numpy as np
import scipy.io as sio
from scipy.signal import welch, find_peaks, spectrogram
from scipy import ndimage, stats

# ------------------ CLI / Defaults ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract GNSS+jam features into NPZ caches.")
    # Default base so you can just run: python data_preparation.py
    p.add_argument("--base", type=str,
                   default=r"D:\datasets\zenodo\3783969\Jamming_Classifier",
                   help="Root containing Image_training_database / Image_validation_database / Image_testing_database")
    p.add_argument("--var", type=str, default="GNSS_plus_Jammer_awgn",
                   help="MAT variable with IQ (complex column vector)")
    p.add_argument("--fs", type=float, default=40_920_000.0,
                   help="Sampling rate (Hz). MATLAB Option=1 uses 40.92 MHz.")
    p.add_argument("--out", type=str, default="./artifacts",
                   help="Artifacts root; NPZs saved under out/prep_run_YYYYmmdd_HHMMSS")
    # Class order mirrors the generator's canonical order (buildClassMap -> 'order')
    p.add_argument("--classes", type=str,
                   default="NoJam,SingleAM,SingleChirp,SingleFM,DME,NB",
                   help="Comma-separated class names (= subfolder names)")
    p.add_argument("--dry_per_class", type=int, default=600,
                   help="Cap #files per class (quick run). None = use all.")
    p.add_argument("--debug", action="store_true", help="Verbose timing (per file).")
    return p.parse_args()

# ------------------ Tunables (validated against generator) ------------------
WELCH_NPERSEG = 1024         # ~40 kHz bin @ 40.92 MHz; good for global PSD + NB peaks (0.5–2 MHz spacing)
WELCH_OVERLAP = 256
MAX_LAG_S     = 200e-6       # captures DME PRI 10–80 µs (first peak well within 200 µs)
INB_BW_HZ     = 2_000_000.0  # "in-band" +/- 2 MHz around DC
ENV_FFT_BAND  = (30.0, 7_000.0)  # Hz for envelope dominant tone (AM/FM mod ~100–5 kHz; widened margin)
EPS           = 1e-20

# Target slice duration for chirp slope proxy (adapts to ~ms windows)
CHIRP_TARGET_SLICE_S = 0.125e-3  # ~0.125 ms
CHIRP_MIN_SLICES     = 6
CHIRP_MAX_SLICES     = 24

# Cyclostationary proxies at GNSS chip-related cyclic freqs (Hz)
CYC_ALPHA1_HZ = 1.023e6   # 1× chip rate
CYC_ALPHA2_HZ = 2.046e6   # 2× chip rate

# ------------------ Small utils ------------------
def now_run_dir(root: Path) -> Path:
    t = time.strftime("%Y%m%d_%H%M%S")
    out = root / f"prep_run_{t}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_iq_meta(path: Path, var_name: str):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if var_name not in m:
        raise KeyError(f"{var_name} not found in {path.name}")
    z = np.asarray(m[var_name]).ravel()

    # Ensure complex IQ. (Some toolchains could store as Nx2 [I,Q].)
    if np.iscomplexobj(z):
        z = z.astype(np.complex64, copy=False)
    else:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2 and z.shape[1] == 2:
            z = (z[:, 0] + 1j * z[:, 1]).astype(np.complex64, copy=False)
        else:
            raise TypeError(f"{path.name}: IQ is not complex and not Nx2 real; shape={z.shape}")

    meta = m.get("meta", {})
    jsr = float(meta.get("JSR_dB", np.nan)) if isinstance(meta, dict) else np.nan
    cnr = float(meta.get("CNR_dBHz", np.nan)) if isinstance(meta, dict) else np.nan
    return z, jsr, cnr

def iter_split_dirs(base: Path, classes):
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

# ------------------ Core feature helpers ------------------
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

# ---- prior extras ----
def inst_freq_stats(z: np.ndarray, fs: float):
    if len(z) < 3:
        return 0.0, 0.0, 0.0, 3.0
    phi = np.unwrap(np.angle(z))
    inst_f = np.diff(phi) * fs / (2*np.pi)  # Hz
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
    S = np.fft.rfft(env * w)          # real FFT
    log_mag = np.log(np.abs(S) + 1e-12)
    ceps = np.fft.irfft(log_mag)
    q = np.arange(ceps.size, dtype=np.float64) / fs
    mask = (q >= qmin) & (q <= qmax)
    return float(np.max(ceps[mask])) if np.any(mask) else 0.0

def dme_pulse_proxy(z: np.ndarray, fs: float):
    env = np.abs(z).astype(np.float64)
    win = max(3, int(round(0.5e-6 * fs)))  # ~0.5 µs smoothing
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

# ---- new physics-driven groups (from previous step) ----
def nb_peaks_and_spacing(f: np.ndarray, Pxx: np.ndarray):
    """Count NB peaks and their spacing stats (median, std) in Hz."""
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
    """Modulation index and dominant envelope tone (freq & normalized power)."""
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
    """Adaptive slices so each slice ~0.125 ms, clamped to [6,24]."""
    total_t = N / float(fs)
    target = CHIRP_TARGET_SLICE_S
    s = int(round(max(CHIRP_MIN_SLICES, min(CHIRP_MAX_SLICES, total_t / target))))
    return max(CHIRP_MIN_SLICES, min(CHIRP_MAX_SLICES, s))

def chirp_slope_proxy(z: np.ndarray, fs: float, slices: int):
    """PSD centroid slope over time (Hz/s) + R^2."""
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

# ---- NEW 7 FEATURES (helpers) ---------------------------------
def cyclo_lag_corr(z: np.ndarray, lag: int) -> float:
    """Normalized correlation magnitude between z[t] and z[t-lag]."""
    if lag <= 0 or lag >= len(z):
        return 0.0
    a = z[lag:]
    b = z[:-lag]
    num = np.vdot(b, a)  # sum conj(b)*a
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real) + EPS
    return float(np.abs(num) / den)

def cyclo_proxies(z: np.ndarray, fs: float):
    L1 = int(round(fs / CYC_ALPHA1_HZ)) if CYC_ALPHA1_HZ > 0 else 0
    L2 = int(round(fs / CYC_ALPHA2_HZ)) if CYC_ALPHA2_HZ > 0 else 0
    c1 = cyclo_lag_corr(z, L1)
    c2 = cyclo_lag_corr(z, L2)
    return c1, c2

def cumulants_c40_c42(z: np.ndarray):
    """Return |C40| and |C42| for unit-variance, zero-mean complex z."""
    if z.size < 8:
        return 0.0, 0.0
    zc = z - np.mean(z)
    p = np.mean(np.abs(zc)**2) + EPS
    zn = zc / np.sqrt(p)
    m20 = np.mean(zn**2)
    m40 = np.mean(zn**4)
    m42 = np.mean((np.abs(zn)**2) * (zn**2))
    c40 = m40 - 3.0 * (m20**2)
    # For unit variance: m21 = E[|zn|^2] = 1
    c42 = m42 - (np.abs(m20)**2) - 2.0
    return float(np.abs(c40)), float(np.abs(c42))

def spectral_kurtosis_stats(z: np.ndarray, fs: float):
    """
    Spectral kurtosis across time:
      1) compute spectrogram (|STFT|^2),
      2) kurtosis over time at each frequency bin,
      3) return mean and max across frequency.
    """
    try:
        f, t, Sxx = spectrogram(z, fs=fs, window="hann",
                                nperseg=512, noverlap=256,
                                detrend=False, return_onesided=False,
                                scaling="density", mode="psd")
        if Sxx.ndim != 2 or Sxx.shape[1] < 4:
            return 0.0, 0.0
        # kurtosis across time axis for each freq row
        sk = stats.kurtosis(Sxx, axis=1, fisher=False, bias=False)
        sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.mean(sk)), float(np.max(sk))
    except Exception:
        return 0.0, 0.0

def tkeo_env_mean(z: np.ndarray):
    """Mean Teager–Kaiser energy on |z|, normalized by mean(|z|)^2."""
    e = np.abs(z).astype(np.float64)
    if e.size < 3:
        return 0.0
    psi = e[1:-1]**2 - e[:-2]*e[2:]
    psi = np.maximum(psi, 0.0)
    denom = (np.mean(e)**2 + EPS)
    return float(np.mean(psi) / denom)

# ------------------ Feature vector ------------------
FEATURE_NAMES = (
    # time-domain shape
    ["meanI","meanQ","stdI","stdQ","corrIQ","mag_mean","mag_std",
     "ZCR_I","ZCR_Q","PAPR_dB","env_ac_peak","env_ac_lag_s",
     # power-aware (raw)
     "pre_rms","psd_power","oob_ratio","crest_env","kurt_env","spec_entropy",
     # spectral shape
     "spec_centroid_Hz","spec_spread_Hz","spec_flatness",
     "spec_rolloff95_Hz","spec_peak_freq_Hz","spec_peak_power"]
    + [f"bandpower_{i}" for i in range(8)]
    # extras
    + ["instf_mean_Hz","instf_std_Hz","instf_slope_Hzps","instf_kurtosis",
       "cep_peak_env","dme_pulse_count","dme_duty","nb_peak_salience"]
    # physics-driven (previous step)
    + ["nb_peak_count","nb_spacing_med_Hz","nb_spacing_std_Hz",
       "env_mod_index","env_dom_freq_Hz","env_dom_peak_norm",
       "chirp_slope_Hzps","chirp_r2"]
    # ---- NEW 7 FEATURES appended here ----
    + ["cyclo_chip_corr","cyclo_2chip_corr",
       "cumulant_c40_mag","cumulant_c42_mag",
       "spec_kurtosis_mean","spec_kurtosis_max",
       "tkeo_env_mean"]
)

def extract_features(iq: np.ndarray, fs: float):
    iq = iq.astype(np.complex64, copy=False)

    # ---------- POWER-AWARE on RAW ----------
    env_raw = np.abs(iq).astype(np.float64)
    pre_rms = float(np.sqrt(np.mean(np.abs(iq)**2)) + EPS)
    crest_env = float(env_raw.max() / (env_raw.mean() + EPS))
    env_std = float(env_raw.std() + EPS)
    kurt_env = float(np.mean(((env_raw - env_raw.mean())/max(env_std, EPS))**4))

    f0, Pxx0 = welch(iq, fs=fs, window="hann", nperseg=WELCH_NPERSEG,
                     noverlap=WELCH_OVERLAP, return_onesided=False, scaling="density")
    order0 = np.argsort(f0)
    f0, Pxx0 = f0[order0], np.maximum(Pxx0[order0], EPS)
    psd_power = float(Pxx0.sum())
    Pprob0 = Pxx0 / (Pxx0.sum() + EPS)
    spec_entropy = float(-np.sum(Pprob0 * np.log(Pprob0)))
    inb = (np.abs(f0) <= INB_BW_HZ)
    oob_ratio = float(Pxx0[~inb].sum() / (Pxx0[inb].sum() + EPS))

    # ---------- NORMALIZE for shape features ----------
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

    # ---------- SPECTRAL SHAPE (normalized) ----------
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

    # ---------- Extras ----------
    f_mean, f_std, f_slope, f_kurt = inst_freq_stats(z, fs)
    cep_env = cepstral_peak_env(z, fs, 2e-4, 5e-3)  # 0.2–5 ms quefrency
    pulse_cnt, duty = dme_pulse_proxy(z, fs)
    sal = nb_peak_salience(f, Pxx, top_k=5)
    feats += [f_mean, f_std, f_slope, f_kurt, cep_env, pulse_cnt, duty, sal]

    # ---------- Physics-driven ----------
    pk_cnt, sp_med, sp_std = nb_peaks_and_spacing(f, Pxx)
    mi, env_f, env_pk = am_envelope_features(z, fs)
    slices = choose_chirp_slices(len(z), fs)
    ch_slope, ch_r2 = chirp_slope_proxy(z, fs, slices)
    feats += [pk_cnt, sp_med, sp_std, mi, env_f, env_pk, ch_slope, ch_r2]

    # ---------- NEW 7 FEATURES ----------
    c1, c2 = cyclo_proxies(z, fs)
    c40_mag, c42_mag = cumulants_c40_c42(z)
    sk_mean, sk_max = spectral_kurtosis_stats(z, fs)
    tkeo_mean = tkeo_env_mean(z)
    feats += [c1, c2, c40_mag, c42_mag, sk_mean, sk_max, tkeo_mean]

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
        print(f"[{split_name}] saved -> {out_dir / f'{split_name.lower()}_features.npz'}  shape={X.shape}")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2)
    print(f"Done. Artifacts under: {out_dir}")

if __name__ == "__main__":
    main()
