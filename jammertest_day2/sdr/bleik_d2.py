"""
kraken_bleik_multifile_spectrograms.py

Loop over all Kraken raw IQ files in a directory that contain "Bleik" in
their filename, take fixed-size IQ "snaps" every SNAP_PERIOD_SEC seconds,
and save spectrograms for each snap.

- Reads ONE complex channel per file (float32, interleaved I/Q)
- Uses the Jammertest plaintext logbook to annotate each snap with the
  jammer state, when we can map the file to an official test (7.1.1 / 7.1.4)
- For files we cannot map (e.g. "*Route*"), logbook labels are omitted.

Outputs (per input file):
- Folder:  OUTPUT_ROOT/<snap_length_label>/<filename>/
          with PNG spectrograms + CSV log.
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
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

# Root directory containing Kraken raw IQ files (RX01-MPC1, Day 2)
INPUT_ROOT = Path(
    r"E:\Jammertest23\23.09.19 - Jammertest 2023 - Day 2\SDR\RX02-MPC2"
)

# Root directory where plots will be stored
# (we will add an intermediate folder for snap length: '33us', '1ms', etc.)
OUTPUT_ROOT = Path(
    r"E:\Jammertest23\Plots\Day2\SDR\RX02-MPC2\Bleik"
)

# Sample rate of THESE Kraken recordings (Hz)
FS = 2_050_000.0  # [Hz] (~2.05 MHz); adjust if you later confirm 2.048 MHz

# Snap configuration
# Default snap length (you can change this, or override with --snap_dur_us)
DEFAULT_SNAP_DUR_US = 1000.0        # e.g. 33, 1000, 2000
DEFAULT_SNAP_PERIOD_SEC = 10.0      # seconds between starts of consecutive snaps
MAX_SNAPS_PER_FILE = None           # e.g. 100 to limit; None = until EOF

# STFT params for spectrograms
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20
DPI_FIG = 140

# ---------- Logbook (plaintext) for 2023-09-19 ----------
LOGBOOK_PATH = Path(r"E:\Jammertest23\Experiment Logs\Testlog Sep 19 2023.txt")
LOCAL_DATE = "2023-09-19"    # date of the test in LOCAL time
LOCAL_UTC_OFFSET = 2.0       # LOCAL - UTC (CEST = +2)

# LO / centre frequency of Kraken front-end for these files (OPTIONAL)
# If unknown, leave as None and the y-axis will show "Baseband frequency"
LO_HZ: Optional[float] = None

# ---------- Testplan metadata (used only for annotations on plots) ----------

@dataclass
class TestMeta:
    test_id: str
    test_name: str
    test_desc: str
    location: str
    duration_sec: Optional[float]   # used to classify ramp_up / ramp_down
    start_local: Optional[datetime] # local datetime when test started


# Nominal test duration from testplan: 13.67 minutes (for the PRN ramps)
PRN_RAMP_DURATION_SEC = 13.67 * 60.0  # ≈ 820.2 s

# You can edit these strings if you want to match the official wording exactly.
TESTMETA_RAMP1 = TestMeta(
    test_id="7.1.1",
    test_name="PRN ramp – Bleik (approx. Test 7.1.1)",
    test_desc=(
        "Porcus Major PRN jammer, 0.1 µW→20 W EIRP, 2 dB steps, 10 s/level; "
        "bands: L1 + G1 + L2 + L5"
    ),
    location="Bleik (see testplan for exact description)",
    duration_sec=PRN_RAMP_DURATION_SEC,
    start_local=datetime(2023, 9, 19, 9, 0, 0),  # local time
)

TESTMETA_RAMP2 = TestMeta(
    test_id="7.1.4",
    test_name="PRN ramp – Ramnan (approx. Test 7.1.4)",
    test_desc=(
        "Porcus Major PRN jammer, 0.1 µW→20 W EIRP, 2 dB steps, 10 s/level; "
        "bands: L1 + G1 + L2 + L5"
    ),
    location="Ramnan (mountainside NW of Bleik, Test Area 1)",
    duration_sec=PRN_RAMP_DURATION_SEC,
    start_local=datetime(2023, 9, 19, 9, 30, 0),  # local time
)

TESTMETA_UNKNOWN = TestMeta(
    test_id="unknown",
    test_name="Unmapped Bleik recording",
    test_desc="Recording could not be mapped to a specific testplan entry.",
    location="Bleik area",
    duration_sec=None,
    start_local=None,
)

# =====================================================================
# ---------------------- LOGBOOK HELPERS ------------------------------
# =====================================================================

Interval = Tuple[datetime, datetime, str]  # (UTC start, UTC end, raw_label)
EPS = 1e-12


def parse_plaintext_logbook(
    path: Path,
    local_date: str,
    local_utc_offset_hours: float,
) -> List[Interval]:
    """
    Parse the Jammertest plaintext logbook for a given local date.

    Example line formats:
      '16:00 - Test was started - no jamming'
      '16:05 - Jammer u1.1 was turned on'

    Returns intervals in UTC with raw string labels:
        'NO JAM', 'u1.1', 's2.1', 'UNKNOWN/CONFUSION', etc.
    """
    time_re = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*[-–—]\s*(.+)$")

    events = []
    base = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    offs = timedelta(hours=float(local_utc_offset_hours))

    def label_from_text(txt: str) -> str:
        t_norm = " ".join(txt.strip().split())
        tl = t_norm.lower()

        # "no jamming" / "turned off" → NO JAM
        if ("no jamming" in tl) or (re.search(r"\bturned\s+off\b", tl) and "confusion" not in tl):
            return "NO JAM"
        if "confusion" in tl:
            return "UNKNOWN/CONFUSION"

        # Jammer <CODE> ... was turned on
        pat = re.compile(
            r"jammer\s+([A-Za-z0-9.\-]+)"
            r"(?:\s*\([^)]*\))?"
            r"\s+(?:was\s+)?turned\s+on\b",
            flags=re.IGNORECASE,
        )
        m = pat.search(t_norm)
        if m:
            return m.group(1).lower()

        # Fallback: Jammer <CODE> ... on   (avoid "turned off")
        if "turned off" not in tl:
            m2 = re.search(
                r"jammer\s+([A-Za-z0-9.\-]+)\b.*\bon\b",
                t_norm,
                flags=re.IGNORECASE,
            )
            if m2:
                return m2.group(1).lower()

        return "UNKNOWN"

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line)
            if not m:
                continue
            hh, mm, rest = m.groups()
            local_dt = base.replace(
                hour=int(hh), minute=int(mm), second=0, microsecond=0
            )
            lbl = label_from_text(rest)
            events.append((local_dt, lbl))

    events.sort(key=lambda e: e[0])
    intervals: List[Interval] = []
    for i in range(len(events)):
        start_local, lbl = events[i]
        end_local = events[i + 1][0] if i + 1 < len(events) else (
            start_local + timedelta(minutes=60)
        )
        intervals.append((start_local - offs, end_local - offs, lbl))
    return intervals


def label_for_time(intervals: List[Interval], t_utc: datetime) -> Optional[str]:
    """Return raw logbook label for a UTC datetime, or None if out of range."""
    for a, b, label in intervals:
        if a <= t_utc < b:
            return label
    return None


# =====================================================================
# ---------------------- KRAKEN SNAP ITERATOR -------------------------
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

def infer_test_meta_for_file(path: Path) -> TestMeta:
    """
    Try to infer which test a given file corresponds to, based on its name.

    Rules:
      - filenames containing "RampTest_1" -> Test 7.1.1 (Bleik PRN ramp)
      - filenames containing "RampTest_2" -> Test 7.1.4 (Ramnan PRN ramp)
      - filenames containing "Route"      -> unmapped "route" recordings
      - everything else -> generic "unknown" Bleik recording
    """
    name = path.name

    if "RampTest_1" in name:
        return TESTMETA_RAMP1
    if "RampTest_2" in name:
        return TESTMETA_RAMP2
    if "Route" in name:
        # We know it's some drive/route but cannot map it to a specific test.
        return TestMeta(
            test_id="route",
            test_name="Bleik route recording (unmapped)",
            test_desc="Drive/route recording that could not be matched to the official tests.",
            location="Bleik area",
            duration_sec=None,
            start_local=None,
        )
    return TESTMETA_UNKNOWN


def find_bleik_kraken_files(root: Path) -> List[Path]:
    """
    Return all files under root that look like Kraken raw files and contain
    'Bleik' in their name. This is intentionally simple; adjust the filter
    if you later discover non-IQ files slipping through.
    """
    files: List[Path] = []
    for p in root.iterdir():
        if p.is_file() and "Bleik" in p.name:
            files.append(p)
    return sorted(files)


# =====================================================================
# ---------------------- PLOTTING FOR KRAKEN --------------------------
# =====================================================================

def plot_and_save_snap(
    snap_idx: int,
    x: np.ndarray,
    fs: float,
    out_dir: Path,
    t_mid_rel: float,
    t_mid_abs_local: Optional[datetime],
    lo_hz: Optional[float],
    phase: str,
    log_label_raw: Optional[str],
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
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

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
    meta_lines = []
    if test_meta.test_id != "unknown":
        meta_lines.append(f"Test {test_meta.test_id} – {test_meta.test_name}")
    else:
        meta_lines.append(test_meta.test_name)
    if test_meta.test_desc:
        meta_lines.append(test_meta.test_desc)
    if test_meta.location:
        meta_lines.append(f"Location: {test_meta.location}")

    if t_mid_abs_local is not None:
        meta_lines.append(
            f"Snap {snap_idx:04d} – mid local time {t_mid_abs_local.strftime('%H:%M:%S')} "
            f"(Δt={t_mid_rel:6.1f} s from test start, phase={phase})"
        )
    else:
        meta_lines.append(
            f"Snap {snap_idx:04d} – mid time Δt={t_mid_rel:6.1f} s "
            f"(test start unknown, phase={phase})"
        )

    jam_txt = f"{log_label_raw}" if log_label_raw is not None else "(no logbook label)"
    meta_lines.append(f"Logbook jammer: {jam_txt}")

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
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    log_tag = (log_label_raw or "nolabel").replace(" ", "_").replace("/", "-")
    out_path = out_dir / f"spec_snap{snap_idx:06d}_{log_tag}.png"
    fig.savefig(out_path, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)

    return out_path


# =====================================================================
# ------------------------------ MAIN --------------------------------
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
    intervals: List[Interval],
    fs: float,
    nsnap: int,
    snap_period_sec: float,
    max_snaps: Optional[int] = None,
) -> None:
    """
    Process one Kraken raw IQ file: take snaps, make spectrograms, and
    write a small CSV log with timing + logbook labels.
    """
    test_meta = infer_test_meta_for_file(raw_path)
    out_dir = out_root / raw_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_dur_sec = nsnap / fs

    print("\n========================================================")
    print(f"File           : {raw_path}")
    print(f"Output dir     : {out_dir}")
    print(f"Inferred test  : {test_meta.test_id} – {test_meta.test_name}")
    print(f"fs             : {fs/1e6:.3f} MHz")
    print(f"SNAP_DUR       : {snap_dur_sec*1e6:.3f} µs")
    print(f"NSNAP          : {nsnap} samples")
    print(f"SNAP_PERIOD    : {snap_period_sec} s (start-to-start)")
    if test_meta.start_local is not None:
        print(f"Start time     : {test_meta.start_local} (local)")
        if test_meta.duration_sec is not None:
            print(f"Nominal dur.   : {test_meta.duration_sec/60:.2f} minutes")
    else:
        print("Start time     : unknown (no mapping to testplan)")

    rows = []
    n_saved = 0
    snap_count = 0

    for snap_idx, x in iter_snaps_every_Xsec(raw_path, fs, nsnap, snap_period_sec):
        N = x.size
        snap_count += 1

        snap_start_rel = snap_idx * snap_period_sec
        snap_mid_rel = snap_start_rel + 0.5 * (N / fs)

        if test_meta.start_local is not None:
            snap_mid_abs_local = test_meta.start_local + timedelta(seconds=snap_mid_rel)
            snap_mid_abs_utc = snap_mid_abs_local - timedelta(hours=LOCAL_UTC_OFFSET)
            log_label_raw = label_for_time(intervals, snap_mid_abs_utc)
        else:
            snap_mid_abs_local = None
            snap_mid_abs_utc = None
            log_label_raw = None

        # Ramp phase classification (only if we know duration)
        if (
            test_meta.duration_sec is not None
            and 0.0 <= snap_mid_rel <= test_meta.duration_sec
        ):
            if snap_mid_rel < test_meta.duration_sec / 2.0:
                phase = "ramp_up"
            else:
                phase = "ramp_down"
        else:
            phase = "outside_nominal_ramp" if test_meta.duration_sec is not None else "unknown"

        out_path = plot_and_save_snap(
            snap_idx=snap_idx,
            x=x,
            fs=fs,
            out_dir=out_dir,
            t_mid_rel=snap_mid_rel,
            t_mid_abs_local=snap_mid_abs_local,
            lo_hz=LO_HZ,
            phase=phase,
            log_label_raw=log_label_raw,
            test_meta=test_meta,
        )

        n_saved += 1

        row = {
            "snap_idx": snap_idx,
            "t_mid_rel_s": float(snap_mid_rel),
            "phase": phase,
            "log_label_raw": log_label_raw,
        }
        if snap_mid_abs_local is not None:
            row["t_mid_local"] = snap_mid_abs_local.isoformat(timespec="seconds")
        if snap_mid_abs_utc is not None:
            row["t_mid_utc"] = snap_mid_abs_utc.replace(
                tzinfo=timezone.utc
            ).isoformat(timespec="seconds")

        rows.append(row)

        print(
            f"[{n_saved:03d}] snap {snap_idx:04d} | "
            f"t_mid_rel={snap_mid_rel:8.2f} s | "
            f"phase={phase:18s} | "
            f"log={log_label_raw or '-':15s} | "
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
                "t_mid_utc",
                "t_mid_rel_s",
                "phase",
                "log_label_raw",
            ]
            # Only keep fields that actually appear
            present = set().union(*[set(r.keys()) for r in rows])
            fieldnames = [f for f in fieldnames if f in present]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote per-snap log to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrograms for all Bleik Kraken files in a folder."
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

    print("========== KRAKEN RAW IQ → SPECTROGRAMS (MULTI-FILE) ==========")
    print(f"Input root   : {INPUT_ROOT}")
    print(f"Output root  : {out_root_with_len}")
    print(f"Logbook path : {LOGBOOK_PATH}")
    print(f"Local date   : {LOCAL_DATE} (UTC offset = +{LOCAL_UTC_OFFSET} h)")
    print(f"fs           : {FS/1e6:.3f} MHz")
    print(f"SNAP_DUR     : {snap_dur_sec*1e6:.3f} µs")
    print(f"NSNAP        : {nsnap} samples")
    print(f"SNAP_PERIOD  : {snap_period_sec} s")
    if max_snaps is not None:
        print(f"Max snaps/file: {max_snaps}")
    print()

    files = find_bleik_kraken_files(INPUT_ROOT)
    if not files:
        print("No files with 'Bleik' in their name were found under INPUT_ROOT.")
        return

    print("Discovered Bleik Kraken files:")
    for p in files:
        meta = infer_test_meta_for_file(p)
        start_str = meta.start_local.strftime("%H:%M:%S") if meta.start_local else "unknown"
        print(f"  - {p.name:40s} | test_id={meta.test_id:7s} | start_local={start_str}")

    print("\nParsing logbook ...")
    intervals = parse_plaintext_logbook(LOGBOOK_PATH, LOCAL_DATE, LOCAL_UTC_OFFSET)
    print(f"Loaded {len(intervals)} logbook intervals (UTC).")

    for i, raw_path in enumerate(files, start=1):
        print(f"\n=== [{i}/{len(files)}] Processing file: {raw_path.name} ===")
        process_single_file(
            raw_path=raw_path,
            out_root=out_root_with_len,
            intervals=intervals,
            fs=FS,
            nsnap=nsnap,
            snap_period_sec=snap_period_sec,
            max_snaps=max_snaps,
        )

    print("\nAll files processed. Done.")


if __name__ == "__main__":
    main()
