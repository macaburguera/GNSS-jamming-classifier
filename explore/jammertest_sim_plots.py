from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import spectrogram, resample_poly
import itertools
import h5py

# ================== CONFIG (Jammertest-like) ==================
BASE_DIR = Path(r"D:\datasets\maca_gen\datasets_jammertest\TRAIN")
CLASSES  = ["WB", "NB", "NoJam", "CW", "Chirp", "FH"]

# Variable names saved by your generator
VAR_IQ   = "GNSS_plus_Jammer_awgn"
VAR_META = "meta"

# Your generator's snap: 2048 @ 62.5 MHz => 32.768 µs
FS_SNAP_TARGET  = 60_000_000
N_SNAP_TARGET   = 2048
T_SNAP_S        = N_SNAP_TARGET / FS_SNAP_TARGET  # 32.768 us

# STFT (matches the style you used before)
NPERSEG   = 128
NOVERLAP  = 96
NFFT      = 128

# Visualization
PERCENTILE_CLIM = (5, 99)     # robust color scaling
DRAW_IFTRACE    = False
DB_RANGE_DEFAULT = 40

# ================== MAT I/O helpers ==================
def load_mat_any(path):
    """
    Load .mat (v7 or v7.3). Returns a dict-like accessor.
    For v7.3, returns a light wrapper that can read datasets/groups.
    """
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        return ("v7", m)
    except NotImplementedError:
        f = h5py.File(path, "r")
        return ("v7.3", f)

def close_mat_any(kind, obj):
    if kind == "v7.3":
        obj.close()

def get_var(kind, m, name):
    """Return raw object for variable `name` (array/group/etc.)."""
    if kind == "v7":
        if name not in m:
            raise KeyError(f"'{name}' not found")
        return m[name]
    else:
        node = m.get(name, None)
        if node is None:
            # try deep search by exact name
            node = _h5_find_by_name(m, name)
        if node is None:
            raise KeyError(f"'{name}' not found")
        return node

def _h5_find_by_name(h5obj, target):
    for k, item in h5obj.items():
        if k == target:
            return item
        if isinstance(item, h5py.Group):
            hit = _h5_find_by_name(item, target)
            if hit is not None:
                return hit
    return None

def to_complex_1d(x):
    """Convert many MATLAB representations into 1D complex64."""
    x = np.asarray(x)
    # compound dtype with ('real','imag')
    if hasattr(x, "dtype") and x.dtype.names:
        names = {n.lower(): n for n in x.dtype.names}
        r = names.get("real") or names.get("re") or names.get("r")
        i = names.get("imag") or names.get("im") or names.get("i")
        if r and i:
            x = x[r] + 1j * x[i]
    # 2-column real-as-(I,Q)
    if np.isrealobj(x) and x.ndim == 2 and x.shape[1] == 2:
        x = x[:, 0] + 1j * x[:, 1]
    return np.ravel(x, order="F").astype(np.complex64)

def load_iq_and_fs(mat_path):
    """
    Read the short snap IQ and its sampling rate from YOUR files:
    - IQ in VAR_IQ (complex vector)
    - Fs in meta.fs_Hz (falls back to target if missing)
    Resamples to FS_SNAP_TARGET if needed and enforces N_SNAP_TARGET length.
    """
    kind, m = load_mat_any(mat_path)
    try:
        # --- IQ ---
        if kind == "v7":
            iq_raw = get_var(kind, m, VAR_IQ)
            iq = to_complex_1d(iq_raw)
        else:  # v7.3
            node = get_var(kind, m, VAR_IQ)
            if isinstance(node, h5py.Group):
                # MATLAB sometimes stores complex as /var/real /var/imag
                keys = {k.lower(): k for k in node.keys()}
                if "real" in keys and "imag" in keys:
                    iq = np.asarray(node[keys["real"]][()]) + 1j * np.asarray(node[keys["imag"]][()])
                else:
                    # single dataset inside group
                    for v in node.values():
                        if isinstance(v, h5py.Dataset):
                            iq = np.asarray(v[()])
                            break
                    else:
                        raise TypeError("Unsupported HDF5 group format for IQ")
            else:
                iq = np.asarray(node[()])
            iq = to_complex_1d(iq)

        # --- Fs from meta ---
        Fs = FS_SNAP_TARGET  # default
        try:
            meta = get_var(kind, m, VAR_META)
            Fs_maybe = _extract_meta_number(kind, meta,
                                            ["fs_hz", "Fs_Hz", "fs", "Fs"])
            if Fs_maybe is not None:
                Fs = float(Fs_maybe)
        except Exception:
            pass

    finally:
        close_mat_any(kind, m)

    # The file already contains the short snap; just align to target setup
    if abs(Fs - FS_SNAP_TARGET) > 1:
        p, q = _rat(FS_SNAP_TARGET / Fs)
        iq = resample_poly(iq, p, q)
        Fs = FS_SNAP_TARGET

    iq = _fix_len(iq, N_SNAP_TARGET)
    return iq, Fs

def _extract_meta_number(kind, meta_obj, keys_try):
    """
    Pull a scalar (e.g., Fs) out of a MATLAB struct (v7 dict / v7.3 group).
    """
    if kind == "v7":
        # meta is likely a dict (due to simplify_cells=True)
        if isinstance(meta_obj, dict):
            keys_map = {k.lower(): k for k in meta_obj.keys()}
            for k in keys_try:
                if k.lower() in keys_map:
                    v = meta_obj[keys_map[k.lower()]]
                    v = np.asarray(v).ravel()
                    if v.size:
                        return v[0]
        # fallback: numpy void with dtype.names
        if hasattr(meta_obj, "dtype") and meta_obj.dtype.names:
            names = {n.lower(): n for n in meta_obj.dtype.names}
            for k in keys_try:
                if k.lower() in names:
                    v = np.asarray(meta_obj[names[k.lower()]]).ravel()
                    if v.size:
                        return v[0]
        return None
    else:
        # v7.3: 'meta' is an HDF5 group with fields as datasets
        if not isinstance(meta_obj, h5py.Group):
            return None
        keys_map = {k.lower(): k for k in meta_obj.keys()}
        for k in keys_try:
            kk = keys_map.get(k.lower())
            if kk is not None:
                ds = meta_obj[kk]
                if isinstance(ds, h5py.Dataset):
                    arr = np.asarray(ds[()]).ravel()
                    if arr.size:
                        return arr[0]
        return None

def _fix_len(x, N):
    if x.size == N:
        return x
    y = np.zeros(N, dtype=x.dtype)
    n = min(N, x.size)
    y[:n] = x[:n]
    return y

def _rat(x, tol=1e-9):
    from fractions import Fraction
    fr = Fraction(x).limit_denominator(10_000)
    return fr.numerator, fr.denominator

# ================== Spectrogram & plotting ==================
def compute_spectrogram(iq, fs, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                        percentile_clim=PERCENTILE_CLIM, db_range=DB_RANGE_DEFAULT):
    eps = 1e-20
    f, t, Sxx = spectrogram(
        iq, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        detrend=False, scaling="density", mode="psd"
    )
    # Two-sided: center at DC and make frequency axis monotonic
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f   = np.fft.fftshift(f)
    # Map to [-fs/2, fs/2)
    f = np.where(f >= fs/2, f - fs, f)
    # Ensure ascending order (important for pcolormesh)
    idx = np.argsort(f)
    f = f[idx]
    Sxx = Sxx[idx, :]

    SdB = 10 * np.log10(Sxx + eps)
    if percentile_clim is not None:
        vmin = float(np.percentile(SdB, percentile_clim[0]))
        vmax = float(np.percentile(SdB, percentile_clim[1]))
    else:
        vmax = float(np.max(SdB)); vmin = vmax - db_range
    return f, t, SdB, vmin, vmax

def format_freq_mhz(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1e6:.1f}"))

# ================== Pick one .mat per class ==================
picked = {}
for cls in CLASSES:
    folder = BASE_DIR / cls
    files = sorted(folder.glob("*.mat"))
    if files:
        picked[cls] = files[0]
        print(f"[OK] {cls}: {files[0].name}")
    else:
        print(f"[WARN] no .mat in {folder}")

# ================== Grid (3x2) like Jammertest ==================
def grid_jammertest():
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = list(itertools.chain.from_iterable(axes))
    im = None

    for ax, cls in zip(axes, CLASSES):
        if cls not in picked:
            ax.axis("off"); continue

        snap, Fs_snap = load_iq_and_fs(picked[cls])
        f, tt, SdB, vmin, vmax = compute_spectrogram(snap, Fs_snap)

        im = ax.pcolormesh(tt, f, SdB, shading="auto", vmin=vmin, vmax=vmax)
        if DRAW_IFTRACE:
            # Instantaneous frequency trace (optional)
            phi = np.unwrap(np.angle(snap))
            finst = (Fs_snap/(2*np.pi))*np.diff(phi, prepend=phi[0])
            k = max(1, int(0.05e-3 * Fs_snap))
            if k > 1:
                finst = np.convolve(finst, np.ones(k)/k, mode="same")
            ax.plot(np.arange(finst.size)/Fs_snap, finst, lw=0.7, alpha=0.9, color="white")

        ax.set_title(cls)
        format_freq_mhz(ax)

    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")
    for ax in axes[::2]:
        ax.set_ylabel("Frequency (MHz)")

    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Power (dB)")
    fig.suptitle(
        f"Spectrograms — Jammertest style (nperseg={NPERSEG}, noverlap={NOVERLAP}, "
        f"Fs={FS_SNAP_TARGET/1e6:.2f} MHz, ≈{T_SNAP_S*1e6:.3f} µs)",
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()

grid_jammertest()
