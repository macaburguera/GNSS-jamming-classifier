"""
raw_iq_spectrograms_snaps_every_Xsec.py

Make spectrograms from a Kraken raw IQ file:
CH3_RX01_Tues_Bleik_Meas_RampTest_2

Assumptions:
- Data type: little-endian float32
- Layout: interleaved IQ -> [I0, Q0, I1, Q1, ...]
- One channel per file (CH3 here).

This version:
- Takes fixed-size "snaps" of NSNAP complex samples.
- Assumes a sample rate FS (Hz).
- Saves ONE snap & image every SNAP_PERIOD_SEC seconds by skipping
  the IQ in between snaps.
- Annotates each snap with:
    * Testplan info (Jammertest 2023, Test 7.1.4)
    * Approx absolute time-of-day for the snap
    * fs, LO, STFT params, etc.

Edit RAW_PATH, OUT_DIR, FS, NSNAP, SNAP_PERIOD_SEC, TEST_START_LOCAL
and LO_HZ before running.
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scipy.signal import stft

# ---------------- USER CONFIG ----------------
RAW_PATH = r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\Geopositioning and Navigation - Jammertest 2023\23.09.19 - Jammertest 2023 - Day 2\Tuesday - SDR Measurements\Kraken\RX01-MPC1\CH3_RX01_Tues_Bleik_Meas_RampTest_2"
OUT_DIR  = r"D:\datasets\Jammertest2026_Day2\alt02-sbf-morning\CH3_RX01_Tues_Bleik_Meas_RampTest_2"

# Sample rate (correct this if needed – 2.048e6 is also plausible)
FS       = 2_050_000.0     # [Hz]
NSNAP    = 2048            # samples per snap (complex) 2048 per 1ms duration samples
SNAP_PERIOD_SEC = 10.0     # one snap every 10 seconds (start-to-start)

MAX_SNAPS = None           # e.g. 100 to limit; None = until EOF

# STFT params (like your training specs)
NPERSEG  = 64
NOVERLAP = 56

REMOVE_DC = True
VMIN_DB   = -80
VMAX_DB   = -20
EPS       = 1e-12

# ---------- Testplan metadata (Jammertest 2023) ----------
# Test group 7: Stationary high-power jamming, ramp power with PRN - Ramnan
TEST_ID   = "7.1.4"
TEST_NAME = "Stationary high-power PRN ramp – Ramnan"
TEST_DESC = (
    "Porcus Major PRN jammer, 0.1 µW→20 W EIRP, 2 dB steps, 10 s/level; "
    "bands: L1 + G1 + L2 + L5"
)
TEST_LOCATION = "Ramnan (mountainside NW of Bleik, Test Area 1)"

# Nominal test duration from testplan: 13.67 minutes
TEST_DURATION_SEC = 13.67 * 60.0  # ≈ 820.2 s

# Assumed local start time of this specific recording (you told me ~09:30)
# Adjust seconds if your file does not start exactly at the jamming ramp start.
TEST_START_LOCAL = datetime(2023, 9, 19, 9, 30, 0)

# LO / centre frequency of Kraken front-end for this file (fill from your notes)
# Example for GPS L1: LO_HZ = 1575.42e6
LO_HZ = None
# ---------------------------------------------------------


def iter_snaps_every_Xsec(path: str,
                          fs: float,
                          nsnap: int,
                          snap_period_sec: float):
    """
    Yields (snap_idx, x) where x is a 1D complex64 array of length <= nsnap,
    and each snap starts snap_period_sec seconds after the previous one.

    Implementation:
    - Read nsnap samples.
    - Skip the remaining samples in one "period":
          samples_per_period = round(snap_period_sec * fs)
          skip_samples = samples_per_period - nsnap  (must be >= 0)
    """
    samples_per_period = int(round(snap_period_sec * fs))
    if samples_per_period < nsnap:
        raise ValueError(
            f"SNAP_PERIOD_SEC * FS = {samples_per_period} samples "
            f"is smaller than NSNAP = {nsnap}. Overlap not supported."
        )

    # How many samples to skip AFTER each snap
    skip_samples = samples_per_period - nsnap

    # Convert to bytes
    floats_per_snap = nsnap * 2            # I + Q
    bytes_per_snap  = floats_per_snap * 4  # float32

    floats_to_skip = skip_samples * 2
    bytes_to_skip  = floats_to_skip * 4

    with open(path, "rb") as f:
        snap_idx = 0
        while True:
            # Read snap
            raw = f.read(bytes_per_snap)
            if not raw:
                break
            if len(raw) < 8:
                break  # too short to bother

            arr = np.frombuffer(raw, dtype="<f4")  # little-endian float32

            # Ensure even number of floats (I/Q pairs)
            if arr.size % 2:
                arr = arr[:-1]
            if arr.size < 4:
                break  # less than 2 complex samples

            I = arr[0::2]
            Q = arr[1::2]
            x = I + 1j * Q

            yield snap_idx, x
            snap_idx += 1

            # Skip the gap until the next period start
            if bytes_to_skip > 0:
                # Relative seek from current position
                f.seek(bytes_to_skip, 1)


def plot_and_save(
    snap_idx: int,
    x: np.ndarray,
    fs: float,
    out_dir: Path,
    t_start_rel: float = None,
    t_mid_rel: float = None,
    t_start_abs: datetime = None,
    t_mid_abs: datetime = None,
    lo_hz: float = None,
) -> Path:
    """
    Plot spectrogram + IQ waveform for one snap and save to PNG.

    Adds rich metadata in the figure suptitle: test ID, PRN ramp,
    bands, fs, LO, times, etc.
    """
    N = x.size
    if N < 8:
        return None

    if REMOVE_DC:
        x = x - np.mean(x)

    nperseg_eff  = min(NPERSEG, N)
    noverlap_eff = min(NOVERLAP, max(0, nperseg_eff - 1))

    # STFT
    f, t, Z = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    # Center zero freq (baseband)
    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)

    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    # Time axis for IQ
    tt = np.arange(N, dtype=np.float32) / fs
    I = np.real(x)
    Q = np.imag(x)

    fig = plt.figure(figsize=(10, 7))

    # Spectrogram
    ax1 = fig.add_subplot(2, 1, 1)
    if t.size >= 2:
        pcm = ax1.pcolormesh(
            t, f, S_dB, shading="auto",
            vmin=VMIN_DB, vmax=VMAX_DB
        )
        fig.colorbar(pcm, ax=ax1, label="dB")
    else:
        im = ax1.imshow(
            S_dB,
            aspect="auto",
            origin="lower",
            extent=[
                0.0,
                max(1.0 / fs, nperseg_eff / fs),
                f[0],
                f[-1],
            ],
            vmin=VMIN_DB,
            vmax=VMAX_DB,
        )
        fig.colorbar(im, ax=ax1, label="dB")

    if lo_hz is not None:
        ax1.set_ylabel(
            "Baseband frequency [Hz]\n"
            f"(around LO={lo_hz/1e6:.3f} MHz)"
        )
    else:
        ax1.set_ylabel("Baseband frequency [Hz]")

    ax1.set_title("Spectrogram (STFT of 1 snap)")

    # IQ time series
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(tt, I, linewidth=0.6, label="I")
    ax2.plot(tt, Q, linewidth=0.6, alpha=0.85, label="Q")
    ax2.set_xlabel("Time within snap [s]")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")

    # ---- Rich suptitle with test metadata ----
    meta_lines = []

    meta_lines.append(
        f"Test {TEST_ID} – {TEST_NAME}"
    )
    meta_lines.append(TEST_DESC)
    meta_lines.append(f"Location: {TEST_LOCATION} | Jammer: Porcus Major")

    if t_mid_abs is not None and t_mid_rel is not None:
        meta_lines.append(
            f"Snap {snap_idx:04d} – mid time {t_mid_abs.strftime('%H:%M:%S')} "
            f"(Δt={t_mid_rel:6.1f} s from test start)"
        )
    else:
        meta_lines.append(f"Snap {snap_idx:04d}")

    if lo_hz is not None:
        meta_lines.append(
            f"fs={fs/1e6:.3f} Msps, LO={lo_hz/1e6:.3f} MHz, "
            f"N={N}, nperseg={nperseg_eff}, noverlap={noverlap_eff}"
        )
    else:
        meta_lines.append(
            f"fs={fs/1e6:.3f} Msps, N={N}, "
            f"nperseg={nperseg_eff}, noverlap={noverlap_eff}"
        )

    fig.suptitle("\n".join(meta_lines), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])  # leave space for suptitle

    out_path = out_dir / f"spec_snap{snap_idx:06d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_dur_sec = NSNAP / FS

    print("========== RAW IQ → SPECTROGRAM SNAPSHOTS ==========")
    print(f"Reading raw IQ from : {RAW_PATH}")
    print(f"Saving plots to     : {out_dir}")
    print()
    print(f"Testplan           : {TEST_ID} – {TEST_NAME}")
    print(f"Description        : {TEST_DESC}")
    print(f"Location           : {TEST_LOCATION}")
    print(f"Assumed start time : {TEST_START_LOCAL} (local)")
    print(f"Nominal duration   : {TEST_DURATION_SEC/60:.2f} minutes")
    print()
    print(f"fs                 : {FS/1e6:.3f} MHz")
    if LO_HZ is not None:
        print(f"LO                 : {LO_HZ/1e6:.3f} MHz")
    print(f"NSNAP              : {NSNAP} samples → {snap_dur_sec*1e6:.3f} µs per snap")
    print(f"SNAP_PERIOD_SEC    : {SNAP_PERIOD_SEC} s (one snap every X seconds)")
    print("=====================================================")

    n_saved = 0
    for snap_idx, x in iter_snaps_every_Xsec(RAW_PATH, FS, NSNAP, SNAP_PERIOD_SEC):
        N = x.size

        # Relative times wrt TEST_START_LOCAL
        snap_start_rel = snap_idx * SNAP_PERIOD_SEC
        snap_mid_rel   = snap_start_rel + 0.5 * (N / FS)

        snap_start_abs = TEST_START_LOCAL + timedelta(seconds=snap_start_rel)
        snap_mid_abs   = TEST_START_LOCAL + timedelta(seconds=snap_mid_rel)

        # Classify where we are in the nominal ramp
        if 0.0 <= snap_mid_rel <= TEST_DURATION_SEC:
            if snap_mid_rel < TEST_DURATION_SEC / 2.0:
                phase = "ramp_up"
            else:
                phase = "ramp_down"
        else:
            phase = "outside_nominal_ramp"

        out_path = plot_and_save(
            snap_idx,
            x,
            FS,
            out_dir,
            t_start_rel=snap_start_rel,
            t_mid_rel=snap_mid_rel,
            t_start_abs=snap_start_abs,
            t_mid_abs=snap_mid_abs,
            lo_hz=LO_HZ,
        )
        n_saved += 1

        print(
            f"[{n_saved:03d}] snap {snap_idx:04d} | "
            f"t_rel_mid={snap_mid_rel:7.2f} s | "
            f"t_abs_mid={snap_mid_abs.strftime('%H:%M:%S')} | "
            f"phase={phase:18s} | "
            f"N={N:4d} -> {out_path}"
        )

        if MAX_SNAPS is not None and n_saved >= MAX_SNAPS:
            break

    print("Done.")


if __name__ == "__main__":
    main()
