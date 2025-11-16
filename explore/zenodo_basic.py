from pathlib import Path
import scipy.io as sio
import numpy as np
from scipy.signal import spectrogram


mat_path = Path(r"D:\datasets\maca_gen\datasets_jammertest\TRAIN\Chirp\TRAIN_Chirp_000030.mat")
assert mat_path.exists(), f"Not found: {mat_path}"

m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
iq = np.asarray(m["GNSS_plus_Jammer_awgn"]).ravel().astype(np.complex64)
print(iq.shape, iq.dtype)


import matplotlib.pyplot as plt

# --- Optional: set your sample rate (Hz). If unknown, leave as None and the x-axis will be sample index.
FS_HZ = 40_920_000 # None  # e.g., FS_HZ = 5e6

# Build x-axis: time in seconds if FS_HZ provided, else sample index
N = iq.size
if FS_HZ:
    t = np.arange(N) / float(FS_HZ)
    xlab = "Time (s)"
else:
    t = np.arange(N)
    xlab = "Sample index"

# Downsample for visualization if very long (max ~100k points)
MAX_POINTS = 100_000
step = max(1, N // MAX_POINTS)
t_plot = t[::step]
I = iq.real[::step]
Q = iq.imag[::step]
mag = np.abs(iq)[::step]

# --- Plot I and Q over time
plt.figure(figsize=(10, 4))
plt.plot(t_plot, I, label="I")
plt.plot(t_plot, Q, label="Q", alpha=0.8)
plt.title("IQ signal in time")
plt.xlabel(xlab)
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ---- Spectrogram (in dB)
# (optional) normalize power for visual comparability
iqn = iq / (np.sqrt(np.mean(np.abs(iq)**2)) + 1e-20)

# shorter window to resolve time-varying freq (good for chirps & DME)
nperseg  = 128           # try 256..1024
noverlap = int(0.80*nperseg)
nfft     = 4096          # padding for smoother display (doesn't add true resolution)

f, t, Sxx = spectrogram(
    iqn, fs=FS_HZ, window="hann",
    nperseg=nperseg, noverlap=noverlap, nfft=nfft,
    detrend=False, scaling="density", mode="psd"
)

# center around 0 Hz correctly (no extra -FS_HZ/2)
Sxx = np.fft.fftshift(Sxx, axes=0)
f   = np.fft.fftshift(f)  # now ~[-Fs/2, +Fs/2]

Sxx_dB = 10*np.log10(Sxx + 1e-20)
vmax   = Sxx_dB.max()
vmin   = vmax - 60

plt.figure(figsize=(10,4))
plt.pcolormesh(t, f, Sxx_dB, shading="auto", vmin=vmin, vmax=vmax)
plt.title("Spectrogram (chirp-friendly: nperseg=256, 90% overlap, centered)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()