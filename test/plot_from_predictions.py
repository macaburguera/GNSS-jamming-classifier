#!/usr/bin/env python3
# plot_from_predictions.py

"""
Generate spectrogram plots AFTER the scan, using predictions CSV to decide what to plot.

Workflow:
1) Run scan_incidents_only_context_noplots.py -> produces run_dir with:
   - predictions_context_10min.csv (and logs.txt, detections.csv)

2) Run this script pointing to that run_dir:
   - Reads predictions CSV
   - Filters by classes / thresholds
   - Re-opens each SBF and locates blocks by (block_index, LO)
   - Generates plots (model-space or dB)

No need to rerun inference. This only plots.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


import numpy as np

# --- SBF parser ---
try:
    from sbf_parser import load
except ImportError as e:
    raise SystemExit("Install first:  pip install sbf-parser") from e

# --- optional: use your project preproc helpers (recommended) ---
# If you prefer standalone plots without importing your project, you can replace TE usage
# by implementing STFT+normalization locally. This version reuses TE for exact match.
import sys
import torch


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    ROOT_DIR: Path = Path(r"E:\Roadtest")  # where 25143/..., SBF files live
    RUN_DIR: Path = Path(r"E:\Roadtest\scan_outputs\run_20251220_210040")

    # If None, will be extracted from logs.txt ("[MODEL] ...") if possible
    MODEL_PT: Optional[Path] = None

    PROJECT_DIR: Path = Path(
        r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\1. DTU\4. Fall 2025\gnss jamming\GNSS-jamming-classifier"
    )

    # Which predictions file to use
    PRED_CSV_NAME: str = "predictions_context_10min.csv"

    # Output plots
    OUT_DIRNAME: str = "plots_from_predictions"
    DPI: int = 160

    # Filters
    INCLUDE_CLASSES: Optional[List[str]] = None  # e.g. ["NB", "Chirp", "WB"] ; None => all
    EXCLUDE_CLASSES: Optional[List[str]] = field(default_factory=lambda: ["NoJam", "NB"])  # e.g. ["NoJam"]
    MIN_PINTF: float = 0.0  # e.g. 0.8
    MAX_PLOTS: Optional[int] = None  # e.g. 200

    # Plot style
    # "model" => model-space (same spec_mode/spec_norm as checkpoint)
    # "db"    => dB spectrogram (ignores model normalization, for human inspection)
    PLOT_KIND: str = "model"


CFG = Config()


# =============================================================================
# Helpers
# =============================================================================

def as_u16_array(x):
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(x, dtype="<u2")
    return np.asarray(x, dtype=np.uint16)

def unpack_iq_int8(words_u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    I8 = ((words_u16 >> 8) & 0xFF).astype(np.int8)
    Q8 = (words_u16 & 0xFF).astype(np.int8)
    return I8, Q8

def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)

def read_model_path_from_logs(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\[MODEL\]\s+(.*)\s*$", txt, flags=re.MULTILINE)
    if not m:
        return None
    p = Path(m.group(1).strip())
    return p

def load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_preproc_from_checkpoint(model_pt: Path) -> Tuple[List[str], Dict[str, Any]]:
    bundle = load_bundle(model_pt)
    if not isinstance(bundle, dict):
        raise RuntimeError("Checkpoint is not a dict bundle.")
    classes = list(bundle.get("classes", ["NoJam", "Chirp", "NB", "WB"]))

    if isinstance(bundle.get("preproc"), dict):
        preproc = dict(bundle["preproc"])
    else:
        C = bundle.get("config", {})
        if not isinstance(C, dict):
            C = {}
        preproc = dict(
            fs_default=float(C.get("FS_HZ", 60_000_000.0)),
            use_npz_fs=bool(C.get("USE_NPZ_FS", True)),
            target_len=int(C.get("TARGET_LEN", 2048)),
            nfft=int(C.get("NFFT", 256)),
            win=int(C.get("WIN", 256)),
            hop=int(C.get("HOP", 64)),
            spec_mode=str(C.get("SPEC_MODE", "logpow")),
            spec_norm=str(C.get("SPEC_NORM", "zscore")),
            fftshift=bool(C.get("FFTSHIFT", True)),
            eps=float(C.get("EPS", 1e-12)),
        )
    return classes, preproc


def save_plot(out_png: Path, z_plot: np.ndarray, S: np.ndarray, fs: float, nfft: int, hop: int, fftshift: bool,
              title: str, spec_label: str, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_iq = (np.arange(z_plot.size, dtype=np.float64) / float(fs))

    spec = S[0] if (S.ndim == 3) else S
    F, T = spec.shape

    dt = float(hop) / float(fs)
    x0, x1 = 0.0, float(T) * dt

    f = np.fft.fftfreq(int(nfft), d=1.0 / float(fs))
    if fftshift:
        f = np.fft.fftshift(f)
    if int(F) == int(nfft):
        y0, y1 = float(f[0]), float(f[-1])
    else:
        y0, y1 = float(f.min()), float(f.max())

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])

    ax_spec = fig.add_subplot(gs[0, 0])
    im = ax_spec.imshow(spec, aspect="auto", origin="lower", extent=[x0, x1, y0, y1])
    ax_spec.set_ylabel("Baseband freq [Hz]")
    cb = fig.colorbar(im, ax=ax_spec)
    cb.set_label(spec_label)

    ax_iq = fig.add_subplot(gs[1, 0])
    ax_iq.plot(t_iq, np.real(z_plot), label="I")
    ax_iq.plot(t_iq, np.imag(z_plot), label="Q")
    ax_iq.set_xlabel("Time [s]")
    ax_iq.set_ylabel("Amplitude (norm.)")
    ax_iq.grid(True, alpha=0.25)
    ax_iq.legend(loc="upper right")

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    run_dir = CFG.RUN_DIR.expanduser().resolve()
    pred_csv = run_dir / CFG.PRED_CSV_NAME
    log_path = run_dir / "logs.txt"

    if not pred_csv.exists():
        raise SystemExit(f"Missing predictions CSV: {pred_csv}")

    model_pt = CFG.MODEL_PT
    if model_pt is None:
        model_pt = read_model_path_from_logs(log_path)
    if model_pt is None:
        raise SystemExit("MODEL_PT not set and could not be parsed from logs.txt ([MODEL] line).")
    model_pt = model_pt.expanduser().resolve()

    # Import TE for exact same preprocessing utilities
    sys.path.insert(0, str(CFG.PROJECT_DIR))
    import train_eval_cnn_spectrogram as TE  # noqa

    classes, preproc = load_preproc_from_checkpoint(model_pt)
    target_len = int(preproc.get("target_len", 2048))
    nfft = int(preproc.get("nfft", 256))
    win = int(preproc.get("win", 256))
    hop = int(preproc.get("hop", 64))
    spec_mode = str(preproc.get("spec_mode", "logpow"))
    spec_norm = str(preproc.get("spec_norm", "zscore"))
    fftshift = bool(preproc.get("fftshift", True))
    eps = float(preproc.get("eps", 1e-12))
    use_npz_fs = bool(preproc.get("use_npz_fs", True))
    fs_default = float(preproc.get("fs_default", 60_000_000.0))

    out_dir = run_dir / CFG.OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions and filter
    rows: List[Dict[str, str]] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    def want_row(row: Dict[str, str]) -> bool:
        pred = row.get("pred", "")
        if CFG.INCLUDE_CLASSES is not None and pred not in CFG.INCLUDE_CLASSES:
            return False
        if CFG.EXCLUDE_CLASSES is not None and pred in CFG.EXCLUDE_CLASSES:
            return False
        try:
            p_intf = float(row.get("p_intf", "nan"))
        except Exception:
            return False
        if not np.isfinite(p_intf) or p_intf < float(CFG.MIN_PINTF):
            return False
        return True

    rows = [x for x in rows if want_row(x)]
    if CFG.MAX_PLOTS is not None:
        rows = rows[: int(CFG.MAX_PLOTS)]

    print(f"[PLOTS] selected {len(rows)} rows from {pred_csv.name}")
    print(f"[PLOTS] plot_kind={CFG.PLOT_KIND} (model uses {spec_mode}/{spec_norm})")

    # Index rows by file so we open each SBF once
    want: Dict[Tuple[str, str, int, int], Dict[str, str]] = {}
    for row in rows:
        folder = row["folder"]
        file = row["file"]
        lo = int(row["lo_hz"])
        bidx = int(row["block_index"])
        want[(folder, file, lo, bidx)] = row

    # Group keys by (folder,file)
    by_file: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for (folder, file, lo, bidx) in want.keys():
        by_file.setdefault((folder, file), []).append((lo, bidx))

    plotted = 0
    for (folder, file), targets in by_file.items():
        sbf_path = (CFG.ROOT_DIR / folder / file).resolve()
        if not sbf_path.exists():
            print(f"[WARN] missing SBF: {sbf_path}")
            continue

        # Convert to a fast lookup set
        target_set = set(targets)

        bb_index_in_file = 0
        with sbf_path.open("rb") as f:
            for block_type, info in load(f):
                if block_type != "BBSamples":
                    continue
                bb_index_in_file += 1

                Fs = int(info.get("SampleFreq", 0))
                LO = int(info.get("LOFreq", 0))
                samples = info.get("Samples", None)
                if Fs <= 0 or samples is None:
                    continue

                key = (LO, bb_index_in_file)
                if key not in target_set:
                    continue

                row = want[(folder, file, LO, bb_index_in_file)]

                words = as_u16_array(samples)
                I8, Q8 = unpack_iq_int8(words)
                z_raw = (I8.astype(np.float32) / 128.0) + 1j * (Q8.astype(np.float32) / 128.0)
                z_raw = z_raw.astype(np.complex64, copy=False)

                fs_used = float(Fs) if use_npz_fs else fs_default

                # Plot IQ uses cropped/padded but NOT necessarily normalized
                z_plot = TE.center_crop_or_pad(z_raw, target_len)

                # Spectrogram data
                if CFG.PLOT_KIND.lower() == "model":
                    z_in = TE.normalize_iq(z_plot.copy())
                    S = TE.make_spectrogram(
                        z=z_in,
                        fs=fs_used,
                        nfft=nfft,
                        win=win,
                        hop=hop,
                        spec_mode=spec_mode,
                        spec_norm=spec_norm,
                        eps=eps,
                        do_fftshift=fftshift,
                    ).astype(np.float32, copy=False)
                    spec_label = f"{spec_mode} / {spec_norm}"
                elif CFG.PLOT_KIND.lower() == "db":
                    # Human-friendly dB plot (no model normalization)
                    z_in = TE.normalize_iq(z_plot.copy())
                    S0 = TE.make_spectrogram(
                        z=z_in,
                        fs=fs_used,
                        nfft=nfft,
                        win=win,
                        hop=hop,
                        spec_mode="db",
                        spec_norm="none",
                        eps=eps,
                        do_fftshift=fftshift,
                    ).astype(np.float32, copy=False)
                    S = S0
                    spec_label = "dB (no norm)"
                else:
                    raise SystemExit("PLOT_KIND must be 'model' or 'db'")

                pred = row.get("pred", "")
                p_pred = row.get("p_pred", "")
                p_intf = row.get("p_intf", "")
                t_utc = row.get("t_utc", "")
                off = row.get("offset_s", "")
                in_ctx = row.get("in_context", "")

                out_png = out_dir / folder / safe_name(file) / f"LO_{LO}" / (
                    f"bb_{bb_index_in_file:09d}_LO{LO}_off{safe_name(off)}_{safe_name(pred)}_pintf{safe_name(p_intf)}.png"
                )

                title = (
                    f"{folder}/{file} | bb_index={bb_index_in_file} | LO={LO} Hz\n"
                    f"t={t_utc} | offset={off}s | in_ctx={in_ctx} | pred={pred} p_pred={p_pred} p_intf={p_intf}"
                )

                save_plot(
                    out_png=out_png,
                    z_plot=z_plot,
                    S=S,
                    fs=fs_used,
                    nfft=nfft,
                    hop=hop,
                    fftshift=fftshift,
                    title=title,
                    spec_label=spec_label,
                    dpi=CFG.DPI,
                )

                plotted += 1
                print(f"[OK] {out_png.relative_to(run_dir)}")

    print(f"[DONE] plotted={plotted} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
