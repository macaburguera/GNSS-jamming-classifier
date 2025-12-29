#!/usr/bin/env python3
"""
scan_sbf_fullfile_dl.py

Scan ONE SBF file.
Run DL inference on EVERY BBSamples block.
Optionally generate spectrogram plots.

Self-contained version with:
- SAME spectrogram color scale as old plots (blue → yellow)
- timestamp
- sampling frequency
- LO (band) frequency (MHz)
- predicted label + probability
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ---- plotting ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- SBF parser ----------
from sbf_parser import load


# =============================================================================
#                                   CONFIG
# =============================================================================

@dataclass
class Config:
    PROJECT_DIR: Path = Path(
        r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\1. DTU\4. Fall 2025\gnss jamming\GNSS-jamming-classifier"
    )

    SBF_FILE: Path = Path(
        r"E:\Jammertest23\23.09.21 - Jammertest 2023 - Day 4\Roadside test 2\alt03333.sbf"
    )

    OUT_DIR: Path = Path(
        r"E:\Jammertest23\23.09.21 - Jammertest 2023 - Day 4\Roadside test 2\alt03333_out"
    )

    MODEL_PT: Path = Path(
        r"..\artifacts\finetuned_DL\finetune_spec_20251216_161529\model_finetuned.pt"
    )

    DEVICE: str = "auto"
    MAKE_PLOTS: bool = True
    PLOTS_DIRNAME: str = "plots"


CFG = Config()


# =============================================================================
#                              IQ HELPERS
# =============================================================================

def as_u16_array(x):
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(x, dtype="<u2")
    return np.asarray(x, dtype=np.uint16)

def unpack_iq_int8(words_u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    I8 = ((words_u16 >> 8) & 0xFF).astype(np.int8)
    Q8 = (words_u16 & 0xFF).astype(np.int8)
    return I8, Q8


# =============================================================================
#                       MODEL + PREPROCESSING LOADING
# =============================================================================

def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_model_and_preproc(model_path: Path):
    bundle = torch_load_bundle(model_path)

    classes = list(bundle.get("classes", ["NoJam", "Chirp", "NB", "WB"]))
    preproc = dict(bundle.get("preproc", {}))
    config = dict(bundle.get("config", {}))

    sys.path.insert(0, str(CFG.PROJECT_DIR))
    import train_eval_cnn_spectrogram as TE  # noqa

    model_kind = str(config.get("MODEL", "se_cnn"))
    in_ch = 3 if preproc.get("spec_mode", "logpow") == "logpow_phase3" else 1
    K = len(classes)

    if model_kind == "cnn":
        model = TE.SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=False)
    elif model_kind == "se_cnn":
        model = TE.SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=True)
    else:
        raise ValueError(f"Unsupported MODEL={model_kind}")

    state = bundle.get("model_state") or bundle.get("state_dict")
    model.load_state_dict(state, strict=False)

    return model, classes, preproc, TE


# =============================================================================
#                               PLOTTING
# =============================================================================

def plot_block(
    out_png: Path,
    S: np.ndarray,
    z: np.ndarray,
    fs: float,
    lo_hz: int,
    nfft: int,
    hop: int,
    fftshift: bool,
    block_index: int,
    t_signal: float,
    pred: str,
    p_pred: float,
):
    spec = S[0] if S.ndim == 3 else S

    dt = hop / fs
    x1 = spec.shape[1] * dt

    f = np.fft.fftfreq(nfft, d=1.0 / fs)
    if fftshift:
        f = np.fft.fftshift(f)

    t_iq = np.arange(z.size) / fs
    lo_mhz = lo_hz / 1e6

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=[0.0, x1, f.min(), f.max()],
        cmap="viridis",
    )
    ax0.set_ylabel("Baseband frequency [Hz]")
    cb = fig.colorbar(im, ax=ax0)
    cb.set_label("Model-space power")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t_iq, np.real(z), label="I")
    ax1.plot(t_iq, np.imag(z), label="Q")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude (norm.)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    fig.suptitle(
        f"Block {block_index} | t={t_signal:.6f} s | fs={fs/1e6:.1f} MHz | LO={lo_mhz:.3f} MHz\n"
        f"Pred: {pred} ({p_pred*100:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
#                                   MAIN
# =============================================================================

def main() -> int:
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = CFG.OUT_DIR / CFG.PLOTS_DIRNAME
    if CFG.MAKE_PLOTS:
        plots_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = CFG.OUT_DIR / "predictions.csv"

    device = resolve_device(CFG.DEVICE)
    model, classes, preproc, TE = load_model_and_preproc(CFG.MODEL_PT)
    model.to(device).eval()

    target_len = int(preproc.get("target_len", 2048))
    nfft = int(preproc.get("nfft", 256))
    win = int(preproc.get("win", 256))
    hop = int(preproc.get("hop", 64))
    spec_mode = str(preproc.get("spec_mode", "logpow"))
    spec_norm = str(preproc.get("spec_norm", "zscore"))
    fftshift = bool(preproc.get("fftshift", True))
    eps = float(preproc.get("eps", 1e-12))
    fs_default = float(preproc.get("fs_default", 60_000_000.0))

    sample_counter = 0

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["block_index", "t_signal_s", "fs_hz", "lo_hz", "n_samp", "pred", "p_pred"]
            + [f"p_{c}" for c in classes]
        )

        with CFG.SBF_FILE.open("rb") as fh:
            bb_index = 0

            for block_type, info in load(fh):
                if block_type != "BBSamples":
                    continue

                bb_index += 1
                Fs = int(info.get("SampleFreq", fs_default))
                LO = int(info.get("LOFreq", 0))
                samples = info.get("Samples", None)
                if samples is None:
                    continue

                words = as_u16_array(samples)
                I8, Q8 = unpack_iq_int8(words)
                z = (I8.astype(np.float32) / 128.0) + 1j * (Q8.astype(np.float32) / 128.0)
                z = z.astype(np.complex64, copy=False)

                z_proc = TE.center_crop_or_pad(z, target_len)
                z_norm = TE.normalize_iq(z_proc.copy())

                S = TE.make_spectrogram(
                    z=z_norm,
                    fs=float(Fs),
                    nfft=nfft,
                    win=win,
                    hop=hop,
                    spec_mode=spec_mode,
                    spec_norm=spec_norm,
                    eps=eps,
                    do_fftshift=fftshift,
                ).astype(np.float32, copy=False)

                xt = torch.from_numpy(S[None, ...]).to(device)
                with torch.no_grad():
                    pr = torch.softmax(model(xt), dim=-1).cpu().numpy().ravel()

                pred_i = int(np.argmax(pr))
                pred = classes[pred_i]
                p_pred = float(pr[pred_i])

                t_signal = sample_counter / Fs

                writer.writerow(
                    [bb_index, f"{t_signal:.6f}", Fs, LO, len(z_proc), pred, f"{p_pred:.6f}"]
                    + [f"{float(p):.6f}" for p in pr]
                )

                if CFG.MAKE_PLOTS:
                    lo_mhz = LO / 1e6
                    plot_block(
                        plots_dir / f"block_{bb_index:06d}_LO{lo_mhz:.0f}_{pred}.png",
                        S,
                        z_proc,
                        float(Fs),
                        LO,
                        nfft,
                        hop,
                        fftshift,
                        bb_index,
                        t_signal,
                        pred,
                        p_pred,
                    )

                sample_counter += len(z)

    print(f"[DONE] {pred_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
