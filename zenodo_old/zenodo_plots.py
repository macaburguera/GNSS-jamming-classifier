# zenodo_batch_compare.py
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import spectrogram
import itertools

# ---------- CONFIG ----------
BASE_DIR = Path(r"D:\datasets\zenodo\3783969\Jamming_Classifier\image_training_database")  # <-- change if needed
CLASSES  = ["DME", "NB", "NoJam", "SingleAM", "SingleChirp", "SingleFM"]
VAR_NAME = "GNSS_plus_Jammer_awgn"
FS_HZ    = 40_920_000 #16_368_000  # 16.368 MHz (1 ms chunks of 16368 samples) or 40_920_000  # 40.92 MHz if data from synthetic generator (option==1)

# Spectrogram settings (same for all classes → comparable)
NPERSEG  = 256          # good compromise across all types (incl. chirp & DME) 256
NOVERLAP = int(0.90*NPERSEG)
NFFT     = 4096         # padding for smooth display (no true resolution gain)
DB_RANGE = 60           # show top 60 dB
NORM_RMS = True         # unit-RMS normalization for visual comparability

# ---------- HELPERS ----------
def load_iq_from_mat(path, var=VAR_NAME):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    iq = np.asarray(m[var]).ravel().astype(np.complex64)
    return iq

def to_time_axis(iq, fs):
    n = iq.size
    return np.arange(n, dtype=np.float64) / float(fs)

def compute_spectrogram(iq, fs):
    # (optional) RMS-normalize for comparability
    if NORM_RMS:
        rms = np.sqrt(np.mean(np.abs(iq)**2)) + 1e-20
        x = iq / rms
    else:
        x = iq

    f, t, Sxx = spectrogram(
        x, fs=fs, window="hann",
        nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
        detrend=False, scaling="density", mode="psd"
    )
    # Center around 0 Hz (no subtraction of fs/2)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f   = np.fft.fftshift(f)  # now ~[-Fs/2, +Fs/2]
    Sxx_dB = 10*np.log10(Sxx + 1e-20)

    # Fixed “top DB_RANGE dB” window
    vmax = Sxx_dB.max()
    vmin = vmax - DB_RANGE
    return f, t, Sxx_dB, vmin, vmax

def format_freq_mhz(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1e6:.1f}"))

# ---------- PICK ONE FILE PER CLASS ----------
picked = {}
for cls in CLASSES:
    folder = BASE_DIR / cls
    candidates = sorted(folder.glob("*.mat"))
    if not candidates:
        print(f"[WARN] No .mat files in: {folder}")
        continue
    picked[cls] = candidates[0]  # pick first
    print(f"[OK] {cls}: {picked[cls].name}")

# ---------- PLOT ONE-BY-ONE ----------
for cls in CLASSES:
    if cls not in picked:
        continue
    mat_path = picked[cls]
    iq = load_iq_from_mat(mat_path)
    t = to_time_axis(iq, FS_HZ)

    # IQ vs time
    plt.figure(figsize=(10, 3.6))
    plt.plot(t, iq.real, label="I")
    plt.plot(t, iq.imag, label="Q", alpha=0.85)
    plt.title(f"{cls} — IQ vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Spectrogram
    f, tt, Sxx_dB, vmin, vmax = compute_spectrogram(iq, FS_HZ)
    plt.figure(figsize=(10, 3.6))
    plt.pcolormesh(tt, f, Sxx_dB, shading="auto", vmin=vmin, vmax=vmax)
    plt.title(f"{cls} — Spectrogram (Hann, nperseg={NPERSEG}, 90% overlap)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    format_freq_mhz(plt.gca())
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)")
    plt.tight_layout()
    plt.show()

# ---------- COMPARISON GRIDS ----------
# 1) All IQ plots in one figure (3x2)
fig_iq, axes_iq = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
axes_iq = list(itertools.chain.from_iterable(axes_iq))  # flatten

for ax, cls in zip(axes_iq, CLASSES):
    if cls not in picked:
        ax.axis("off")
        continue
    iq = load_iq_from_mat(picked[cls])
    t = to_time_axis(iq, FS_HZ)
    ax.plot(t, iq.real, label="I")
    ax.plot(t, iq.imag, label="Q", alpha=0.85)
    ax.set_title(cls)
    ax.grid(True, linestyle="--", alpha=0.3)

for ax in axes_iq[-2:]:  # bottom row x-labels
    ax.set_xlabel("Time (s)")
for ax in axes_iq[::2]:  # left column y-labels
    ax.set_ylabel("Amplitude")
fig_iq.suptitle("IQ vs Time — All Classes", y=0.98)
fig_iq.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 2) All spectrograms in one figure (3x2)
fig_sp, axes_sp = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
axes_sp = list(itertools.chain.from_iterable(axes_sp))

for ax, cls in zip(axes_sp, CLASSES):
    if cls not in picked:
        ax.axis("off")
        continue
    iq = load_iq_from_mat(picked[cls])
    f, tt, Sxx_dB, vmin, vmax = compute_spectrogram(iq, FS_HZ)
    im = ax.pcolormesh(tt, f, Sxx_dB, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(cls)
    format_freq_mhz(ax)
    ax.grid(False)

# Axis labels
for ax in axes_sp[-2:]:
    ax.set_xlabel("Time (s)")
for ax in axes_sp[::2]:
    ax.set_ylabel("Frequency (MHz)")

# One shared colorbar
cax = fig_sp.add_axes([0.92, 0.12, 0.02, 0.76])
cb = fig_sp.colorbar(im, cax=cax)
cb.set_label("Power (dB)")

fig_sp.suptitle(f"Spectrograms — All Classes (Hann, nperseg={NPERSEG}, 90% overlap, nfft={NFFT})", y=0.98)
fig_sp.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()
