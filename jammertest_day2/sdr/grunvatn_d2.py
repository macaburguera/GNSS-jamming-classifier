"""
kraken_grunvatn_multifile_spectrograms.py

Loop over all Kraken raw IQ files in a directory that contain "Grunvatn"
in their filename, take fixed-size IQ "snaps" every SNAP_PERIOD_SEC seconds,
and save spectrograms for each snap.

- Reads ONE complex channel per file (float32, interleaved I/Q)
- Uses the Grunvatn section of the logbook (Testlog Sep 19 2023.txt)
  to annotate each file with:
    * Test number (#0–#9, #11)
    * Short test name
    * Text description
    * Jammer information (H3.1 etc.)
- We do NOT have exact start times for each test, so we assume that all
  tests start at local time 10:00 on 2023-09-19 and use that only as a
  convenient time reference.

Outputs (per input file):
- Folder:  OUTPUT_ROOT/<snap_length_label>/<filename>/
           with PNG spectrograms + CSV log.
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import re
import csv
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from scipy.signal import stft

# =====================================================================
# ------------------------ USER CONFIG --------------------------------
# =====================================================================

# Root directory containing Kraken raw IQ files (e.g. RX01-MPC1, Day 2)
INPUT_ROOT = Path(
    r"E:\Jammertest23\23.09.19 - Jammertest 2023 - Day 2\SDR\RX02-MPC2"
)

# Root directory where plots will be stored
# We will add an intermediate folder for snap length: '33us', '1ms', etc.
OUTPUT_ROOT = Path(
    r"E:\Jammertest23\Plots\Day2\SDR\RX02-MPC2\Grunvatn"
)

# Sample rate of THESE Kraken recordings (Hz)
FS = 1241_000_000.0  # [Hz] (~2.05 MHz); adjust if you later confirm 2.048 MHz

# Snap configuration
# Default snap length (you can change this, or override with --snap_dur_us)
DEFAULT_SNAP_DUR_US = 1000.0        # e.g. 33, 1000, 2000
DEFAULT_SNAP_PERIOD_SEC = 0.1      # seconds between starts of consecutive snaps
MAX_SNAPS_PER_FILE = None           # e.g. 100 to limit; None = until EOF

# STFT params for spectrograms
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20
DPI_FIG = 140

# LO / centre frequency of Kraken front-end for these files (OPTIONAL)
# If unknown, leave as None and the y-axis will show "Baseband frequency"
LO_HZ: Optional[float] = None

# Reference start time (local) we ASSUME for all Grunvatn tests
GRUNVATN_ASSUMED_START_LOCAL = datetime(2023, 9, 19, 10, 0, 0)

# =====================================================================
# ----------------------- TEST METADATA -------------------------------
# =====================================================================

@dataclass
class TestMeta:
    test_id: str
    test_name: str
    test_desc: str
    location: str
    jammer_info: str
    start_local: datetime


GRUNVATN_LOCATION = "Grunvatn (dual-Kraken ground-truth site near Bleik)"

# Base metadata per numbered test, distilled from Testlog Sep 19 2023
GRUNVATN_TESTS = {
    0: TestMeta(
        test_id="0",
        test_name="Background reference test",
        test_desc="Array present, jammer off; reference GNSS-only/background noise.",
        location=GRUNVATN_LOCATION,
        jammer_info="No jammer (background reference).",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    1: TestMeta(
        test_id="1",
        test_name="Static – greater circle",
        test_desc="Static scenario; jammer moved on the outer (greater) circle of reference points around the array.",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    2: TestMeta(
        test_id="2",
        test_name="Static – small circle",
        test_desc="Static scenario; jammer moved on the inner (small) circle of reference points around the array.",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    3: TestMeta(
        test_id="3",
        test_name="Dynamic – greater circle",
        test_desc="Dynamic scenario; jammer walked the greater circle around the array (clockwise and counter-clockwise runs).",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    4: TestMeta(
        test_id="4",
        test_name="Dynamic – small circle",
        test_desc="Dynamic scenario; jammer walked the small circle around the array (clockwise and counter-clockwise runs).",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    5: TestMeta(
        test_id="5",
        test_name="Zigzag walk",
        test_desc="Zigzag walk test: criss-cross/rectangular walk path drawn out from the eight reference points.",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    6: TestMeta(
        test_id="6",
        test_name="Jammer separation",
        test_desc=(
            "Two-jammer test. One jammer kept static near boresight, the other moved. "
            "Variant A: across-track passes in front of stationary jammer. "
            "Variant B: along-track motion drawing a square around the site."
        ),
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1 + secondary jammer (separation study).",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    7: TestMeta(
        test_id="7",
        test_name="Elevation test",
        test_desc="Jammer elevated on a pole and moved along array boresight (from trailer towards the array).",
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    8: TestMeta(
        test_id="8",
        test_name="ABORTED test",
        test_desc="Test aborted due to power loss on MPC-2; replaced by series #9 after power was restored.",
        location=GRUNVATN_LOCATION,
        jammer_info="Unreliable / aborted run (see logbook).",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    9: TestMeta(
        test_id="9",
        test_name="Baseline Karlsvogn track",
        test_desc=(
            "Baseline tests with jammer moving in the 'Karlsvogn' track. "
            "Different separations: 199 cm (standard), 238.5 cm and 245 cm."
        ),
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1.",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
    11: TestMeta(
        test_id="11",
        test_name="Wardriving",
        test_desc=(
            "Wardriving around the site with jammer inside and outside the car. "
            "Variants A–C: jammer inside car. D–F: jammer on the roof; "
            "three recordings in each driving direction."
        ),
        location=GRUNVATN_LOCATION,
        jammer_info="Primary jammer H3.1 (in/around vehicle).",
        start_local=GRUNVATN_ASSUMED_START_LOCAL,
    ),
}

# =====================================================================
# ---------------------- SNAP ITERATOR --------------------------------
# =====================================================================

def iter_snaps_every_Xsec(
    path: Path,
    fs: float,
    nsnap: int,
    snap_period_sec: float,
):
    """
    Yields (snap_idx, x) where x is a 1D complex64 array of length <= nsnap,
    and each snap starts snap_period_sec seconds after the previous one.
    """
    samples_per_period = int(round(snap_period_sec * fs))
    if samples_per_period < nsnap:
        raise ValueError(
            f"SNAP_PERIOD_SEC * FS = {samples_per_period} samples "
            f"is smaller than NSNAP = {nsnap}. Overlap not supported."
        )

    skip_samples = samples_per_period - nsnap

    floats_per_snap = nsnap * 2             # I + Q
    bytes_per_snap = floats_per_snap * 4    # float32

    floats_to_skip = skip_samples * 2
    bytes_to_skip = floats_to_skip * 4

    with path.open("rb") as f:
        snap_idx = 0
        while True:
            raw = f.read(bytes_per_snap)
            if not raw:
                break
            if len(raw) < 8:
                break

            arr = np.frombuffer(raw, dtype="<f4")  # little-endian float32

            if arr.size % 2:
                arr = arr[:-1]
            if arr.size < 4:
                break

            I = arr[0::2]
            Q = arr[1::2]
            x = I + 1j * Q

            yield snap_idx, x.astype(np.complex64, copy=False)
            snap_idx += 1

            if bytes_to_skip > 0:
                f.seek(bytes_to_skip, 1)


# =====================================================================
# ---------------------- FILE / TEST HELPERS --------------------------
# =====================================================================

def find_grunvatn_kraken_files(root: Path) -> List[Path]:
    """
    Return all files under root that look like Kraken raw files and contain
    'Grunvatn' in their name. This is intentionally simple; adjust the filter
    if you later discover non-IQ files slipping through.
    """
    files: List[Path] = []
    for p in root.iterdir():
        if p.is_file() and "Grunvatn" in p.name:
            files.append(p)
    return sorted(files)


def parse_test_number_from_filename(name: str) -> Optional[int]:
    """
    Extract the Grunvatn test number from a filename, e.g.:

        CH0_RX01_Tues_Kraken-Grunvatn-Test_01A.bin         -> 1
        CH0_RX01_Tues_Kraken-Grunvatn-Test_03_Clockwise    -> 3
        CH0_RX01_Tues_Kraken-Grunvatn-Test_11A_Drive_NLOS  -> 11
    """
    m = re.search(r"Test_(\d{2})", name)
    if not m:
        return None
    return int(m.group(1))


def parse_variant_from_filename(name: str) -> Optional[str]:
    """
    Extract an optional variant tag after the test number, e.g.:

        'Test_01A.bin'              -> '01A'
        'Test_03_Clockwise.bin'     -> 'Clockwise'
        'Test_11A_Drive_LOS.bin'    -> 'A_Drive_LOS'
    """
    m = re.search(r"Test_(\d{2})(.*)$", name)
    if not m:
        return None

    rest = m.group(2)  # everything after the two digits
    # Strip extension
    if "." in rest:
        rest = rest.split(".", 1)[0]
    rest = rest.lstrip("_")
    if not rest:
        return None
    return rest


def infer_test_meta_for_file(path: Path) -> TestMeta:
    """
    Map a Grunvatn Kraken filename to a TestMeta based on the test number
    encoded in the name. If we cannot map it, return a generic 'unknown' meta.
    """
    name = path.name
    test_num = parse_test_number_from_filename(name)
    variant = parse_variant_from_filename(name)

    base = GRUNVATN_TESTS.get(test_num) if test_num is not None else None

    if base is None:
        # Fallback generic meta
        return TestMeta(
            test_id=str(test_num) if test_num is not None else "unknown",
            test_name="Unknown Grunvatn test",
            test_desc="File could not be mapped to a documented Grunvatn test in the logbook.",
            location=GRUNVATN_LOCATION,
            jammer_info="Unknown jammer configuration (see logbook).",
            start_local=GRUNVATN_ASSUMED_START_LOCAL,
        )

    # Clone base meta and optionally append variant info to the test_name
    test_name = base.test_name
    if variant:
        # Make the variant human readable
        pretty_variant = variant.replace("_", " ")
        test_name = f"{test_name} – variant {pretty_variant}"

    return TestMeta(
        test_id=base.test_id,
        test_name=test_name,
        test_desc=base.test_desc,
        location=base.location,
        jammer_info=base.jammer_info,
        start_local=base.start_local,
    )


# =====================================================================
# ---------------------- PLOTTING -------------------------------------
# =====================================================================

def plot_and_save_snap(
    snap_idx: int,
    x: np.ndarray,
    fs: float,
    out_dir: Path,
    t_mid_rel: float,
    t_mid_abs_local: Optional[datetime],
    lo_hz: Optional[float],
    test_meta: TestMeta,
) -> Optional[Path]:
    """
    Make a spectrogram + I/Q time series for a single snap and save as PNG.
    """
    N = x.size
    if N < 8:
        return None

    if REMOVE_DC:
        x = x - np.mean(x)

    nperseg_eff = min(NPERSEG, N)
    noverlap_eff = min(NOVERLAP, max(0, nperseg_eff - 1))

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

    # Frequency axis: shift so that 0 Hz is in the middle
    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + 1e-12)

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

    # Time-domain I/Q
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(tt, I, linewidth=0.6, label="I")
    ax2.plot(tt, Q, linewidth=0.6, alpha=0.85, label="Q")
    ax2.set_xlabel("Time within snap [s]")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")

    # ---------- Metadata in the figure title ----------
    meta_lines = [
        f"Grunvatn test #{test_meta.test_id} – {test_meta.test_name}",
    ]
    if test_meta.test_desc:
        meta_lines.append(test_meta.test_desc)
    if test_meta.location:
        meta_lines.append(f"Location: {test_meta.location}")
    if test_meta.jammer_info:
        meta_lines.append(f"Jammer: {test_meta.jammer_info}")

    if t_mid_abs_local is not None:
        meta_lines.append(
            f"Snap {snap_idx:04d} – mid local time {t_mid_abs_local.strftime('%H:%M:%S')} "
            f"(Δt={t_mid_rel:6.1f} s from assumed test start)"
        )
    else:
        meta_lines.append(
            f"Snap {snap_idx:04d} – mid time Δt={t_mid_rel:6.1f} s "
            f"(test start assumed but not mapped precisely)"
        )

    meta_lines.append(
        f"fs={fs/1e6:.3f} Msps, N={N}, "
        f"nperseg={nperseg_eff}, noverlap={noverlap_eff}"
    )

    fig.suptitle("\n".join(meta_lines), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = out_dir / f"spec_snap{snap_idx:06d}.png"
    fig.savefig(out_path, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)

    return out_path


# =====================================================================
# ------------------------------ MAIN ---------------------------------
# =====================================================================

def format_snap_length_folder(snap_dur_sec: float) -> str:
    """
    Format the snap length into a folder-friendly string, e.g.
        3.3e-05  -> '33us'
        0.001    -> '1ms'
        0.002    -> '2ms'
    """
    us = snap_dur_sec * 1e6
    # If it's an integer number of milliseconds, use ms
    if abs(us % 1000.0) < 1e-6:
        ms = us / 1000.0
        if abs(ms - round(ms)) < 1e-6:
            return f"{int(round(ms))}ms"
        return f"{ms:.3f}ms"
    # Fallback: microseconds
    if abs(us - round(us)) < 1e-3:
        return f"{int(round(us))}us"
    return f"{us:.1f}us"


def process_single_file(
    raw_path: Path,
    out_root: Path,
    fs: float,
    nsnap: int,
    snap_period_sec: float,
    max_snaps: Optional[int] = None,
) -> None:
    """
    Process one Kraken raw IQ file: take snaps, make spectrograms, and
    write a small CSV log with timing + test metadata.
    """
    test_meta = infer_test_meta_for_file(raw_path)
    out_dir = out_root / raw_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_dur_sec = nsnap / fs

    print("\n========================================================")
    print(f"File           : {raw_path}")
    print(f"Output dir     : {out_dir}")
    print(f"Grunvatn test  : #{test_meta.test_id} – {test_meta.test_name}")
    print(f"fs             : {fs/1e6:.3f} MHz")
    print(f"SNAP_DUR       : {snap_dur_sec*1e6:.3f} µs")
    print(f"NSNAP          : {nsnap} samples")
    print(f"SNAP_PERIOD    : {snap_period_sec} s (start-to-start)")
    print(f"Assumed start  : {test_meta.start_local} (local)")

    rows = []
    n_saved = 0
    snap_count = 0

    for snap_idx, x in iter_snaps_every_Xsec(raw_path, fs, nsnap, snap_period_sec):
        N = x.size
        snap_count += 1

        snap_start_rel = snap_idx * snap_period_sec
        snap_mid_rel = snap_start_rel + 0.5 * (N / fs)

        snap_mid_abs_local = test_meta.start_local + timedelta(seconds=snap_mid_rel)

        out_path = plot_and_save_snap(
            snap_idx=snap_idx,
            x=x,
            fs=fs,
            out_dir=out_dir,
            t_mid_rel=snap_mid_rel,
            t_mid_abs_local=snap_mid_abs_local,
            lo_hz=LO_HZ,
            test_meta=test_meta,
        )

        n_saved += 1

        row = {
            "snap_idx": snap_idx,
            "t_mid_local": snap_mid_abs_local.isoformat(timespec="seconds"),
            "t_mid_rel_s": float(snap_mid_rel),
            "test_id": test_meta.test_id,
            "test_name": test_meta.test_name,
            "jammer_info": test_meta.jammer_info,
        }

        rows.append(row)

        print(
            f"[{n_saved:03d}] snap {snap_idx:04d} | "
            f"t_mid_rel={snap_mid_rel:8.2f} s | "
            f"test=#{test_meta.test_id:>2s} | "
            f"png={out_path.name if out_path else 'None'}"
        )

        if max_snaps is not None and n_saved >= max_snaps:
            break

    print(f"Processed {snap_count} snaps; saved {n_saved} plots.")

    # Per-file CSV log
    if rows:
        csv_path = out_dir / "kraken_snaps_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "snap_idx",
                "t_mid_local",
                "t_mid_rel_s",
                "test_id",
                "test_name",
                "jammer_info",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote per-snap log to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrograms for all Grunvatn Kraken files in a folder."
    )
    parser.add_argument(
        "--snap_period",
        type=float,
        default=DEFAULT_SNAP_PERIOD_SEC,
        help="Seconds between consecutive snaps (default: %(default)s).",
    )
    parser.add_argument(
        "--max_snaps",
        type=int,
        default=MAX_SNAPS_PER_FILE,
        help="Optional limit of snaps per file (default: no limit).",
    )
    parser.add_argument(
        "--snap_dur_us",
        type=float,
        default=DEFAULT_SNAP_DUR_US,
        help="Snap duration in microseconds (e.g., 33, 1000, 2000). "
             "Default: %(default)s.",
    )
    args = parser.parse_args()

    snap_period_sec = float(args.snap_period)
    max_snaps = args.max_snaps
    snap_dur_sec = float(args.snap_dur_us) * 1e-6
    if snap_dur_sec <= 0:
        raise ValueError("snap_dur_us must be positive.")

    nsnap = int(round(FS * snap_dur_sec))
    if nsnap < 8:
        raise ValueError(
            f"Snap duration {snap_dur_sec} s is too short: nsnap={nsnap} < 8 samples."
        )

    # Build intermediate folder for this snap length, e.g. '33us', '1ms'
    length_folder = format_snap_length_folder(snap_dur_sec)
    out_root_with_len = OUTPUT_ROOT / length_folder
    out_root_with_len.mkdir(parents=True, exist_ok=True)

    print("====== KRAKEN RAW IQ → SPECTROGRAMS (GRUNVATN, MULTI-FILE) ======")
    print(f"Input root   : {INPUT_ROOT}")
    print(f"Output root  : {out_root_with_len}")
    print(f"fs           : {FS/1e6:.3f} MHz")
    print(f"SNAP_DUR     : {snap_dur_sec*1e6:.3f} µs")
    print(f"NSNAP        : {nsnap} samples")
    print(f"SNAP_PERIOD  : {snap_period_sec} s")
    if max_snaps is not None:
        print(f"Max snaps/file: {max_snaps}")
    print()

    files = find_grunvatn_kraken_files(INPUT_ROOT)
    if not files:
        print("No files with 'Grunvatn' in their name were found under INPUT_ROOT.")
        return

    print("Discovered Grunvatn Kraken files:")
    for p in files:
        meta = infer_test_meta_for_file(p)
        print(
            f"  - {p.name:45s} | "
            f"test #{meta.test_id:>2s} | {meta.test_name}"
        )

    for i, raw_path in enumerate(files, start=1):
        print(f"\n=== [{i}/{len(files)}] Processing file: {raw_path.name} ===")
        process_single_file(
            raw_path=raw_path,
            out_root=out_root_with_len,
            fs=FS,
            nsnap=nsnap,
            snap_period_sec=snap_period_sec,
            max_snaps=max_snaps,
        )

    print("\nAll files processed. Done.")


if __name__ == "__main__":
    main()
