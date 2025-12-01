# validation_cnn_rawiq.py
"""
Validate a SIM-trained 1-D CNN (model.pt) on real Jammertest SBF IQ data (raw IQ inference).

What this script does (differences vs. XGB/feature validator):
- Loads a PyTorch CNN checkpoint (the same architecture you trained with).
- Reads raw IQ from SBF BBSamples blocks, crops/pads to model 'target_len',
  applies *exact* training-time normalization (mean removal + unit-RMS), and predicts.
- Keeps your full evaluation pipeline:
    * 30 s gating in UTC (first block at/after boundary)
    * logbook parsing → GT intervals → canonical label mapping
    * strict metrics (fixed class order = order in checkpoint)
    * spectrogram + I/Q time plot per kept block
    * per-sample CSV + summary.json
    * optional conservative NoJam veto based on probabilities and power

Assumptions:
- The checkpoint was created by your train_eval_cnn_rawiq.py and contains:
    {"model_state": ..., "classes": [...], "config": {"target_len": 2048, "fs": 60e6, ...}}
- sbf_parser.SbfParser yields ('BBSamples', infos) with keys:
    "Samples" (interleaved int8 IQ), "N", "SampleFreq", "WNc", "TOW".
"""

from pathlib import Path
import re, json, csv, os
from typing import List, Tuple, Optional, Dict
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----- PyTorch -----
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- SBF parser (your local module) -----
from sbf_parser import SbfParser

# ============================ USER VARIABLES ============================
SBF_PATH   = r"D:\datasets\Jammertest2023_Day1\Altus06 - 150m\alt06001.sbf"
OUT_DIR    = r"D:\datasets\Jammertest2023_Day1\plots\alt06001_predicted_10s_SIMCNN_4cl"
LOGBOOK_PATH = r"D:\datasets\Jammertest2023_Day1\Testlog 23.09.18.txt"

# CNN checkpoint (.pt) produced by train_eval_cnn_rawiq.py
MODEL_PATH = r"..\artifacts\jammertest_sim\cnn_run_20251128_185416\model.pt"

# Local test date/time for logbook parsing
LOCAL_DATE       = "2023-09-18"   # date of the test in LOCAL time
LOCAL_UTC_OFFSET = 2.0            # LOCAL - UTC (e.g., CEST=+2)

# Sampling policy: keep first block at/after each boundary (UTC)
SAVE_EVERY_SEC = 30.0

# Optional decimation before feeding the model (1 keeps exact original fs)
DECIM = 1

# Spectrogram appearance
NPERSEG = 64
NOVERLAP = 56
REMOVE_DC = True
VMIN_DB = -80
VMAX_DB = -20
DPI_FIG = 140

# Save options
CHUNK_BYTES = 1_000_000
SAVE_IMAGES = True
SAVE_PER_SAMPLE_CSV = True
SUMMARY_JSON = True
DEBUG_PRINT_SAMPLE_LABELS = False

# ----------------- Mapping: Jammertest logcode -> model class -----------------
# Keep your semantics: all listed → "Chirp" EXCEPT "h1.1" → "NB",
# "no jam"/"off"/"no jamming" → "NoJam". Unknown/confusion → None (excluded from GT).
LOG_TO_MODEL: Dict[str, Optional[str]] = {
    "no jam": "NoJam",
    "off": "NoJam",
    "no jamming": "NoJam",
    "h1.1": "NB",
    "h1.2": "Chirp",
    "u1.1": "Chirp",
    "u1.2": "Chirp",
    "s1.2": "Chirp",
    "h3.1": "Chirp",
    "s2.1": "Chirp",
    "unknown/confusion": None,
    "unknown": None,
}
# ============================================================================

# ====================== TIME / LOGBOOK HELPERS ======================
EPS = 1e-20
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds

Interval = Tuple[datetime, datetime, str]  # (UTC start, UTC end, raw log label)

def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600); m = int((tsec % 3600) // 60)
    s = tsec - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def extract_time_labels(infos):
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0))
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

def parse_plaintext_logbook(path: str, local_date: str, local_utc_offset_hours: float) -> List[Interval]:
    """
    Lines example:
      '16:00 - Test was started - no jamming'
      '16:05 - Jammer u1.1 was turned on'
    Returns UTC intervals with raw labels like 'NO JAM', 'u1.1', 's1.2'.
    """
    time_re = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*[-–—]\s*(.+)$')

    events = []
    base = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    offs = timedelta(hours=float(local_utc_offset_hours))

    def label_from_text(txt: str) -> str:
        t_norm = " ".join(txt.strip().split())
        tl = t_norm.lower()
        if ("no jamming" in tl) or (re.search(r"\bturned\s+off\b", tl) and "confusion" not in tl):
            return "NO JAM"
        if "confusion" in tl:
            return "UNKNOWN/CONFUSION"
        pat = re.compile(
            r"jammer\s+([A-Za-z0-9.\-]+)"
            r"(?:\s*\([^)]*\))?\s+(?:was\s+)?turned\s+on\b",
            flags=re.IGNORECASE
        )
        m = pat.search(t_norm)
        if m:
            return m.group(1).lower()
        if "turned off" not in tl:
            m2 = re.search(r"jammer\s+([A-Za-z0-9.\-]+)\b.*\bon\b", t_norm, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).lower()
        return "UNKNOWN"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line)
            if not m:
                continue
            hh, mm, rest = m.groups()
            local_dt = base.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            lbl = label_from_text(rest)
            events.append((local_dt, lbl))

    events.sort(key=lambda e: e[0])
    intervals: List[Interval] = []
    for i in range(len(events)):
        start_local, lbl = events[i]
        end_local = events[i+1][0] if i+1 < len(events) else (start_local + timedelta(minutes=60))
        intervals.append((start_local - offs, end_local - offs, lbl))
    return intervals

# ====================== SBF IQ DECODING ======================
def decode_bbsamples_iq(infos):
    if "Samples" not in infos or "N" not in infos:
        return None, None
    buf = infos["Samples"]; N = int(infos["N"])
    arr = np.frombuffer(buf, dtype=np.int8)
    if arr.size != 2 * N:
        return None, None
    I = arr[0::2].astype(np.float32) / 128.0
    Q = arr[1::2].astype(np.float32) / 128.0
    x = I + 1j * Q
    fs = float(infos.get("SampleFreq", 1.0))
    return x, fs

# ====================== CNN MODEL (same as in training) ======================
class RawIQ_CNN(nn.Module):
    """
    Lightweight 1-D CNN over I/Q with dilations to capture lines/chirps/hops.
    Input : (B, 2, N)
    Output: logits (B, K)
    """
    def __init__(self, num_classes: int, in_ch: int=2):
        super().__init__()
        chs = [32, 64, 128, 128]
        self.block1 = self._conv_block(in_ch, chs[0], k=15, d=1)
        self.block2 = self._conv_block(chs[0], chs[1], k=15, d=2)
        self.block3 = self._conv_block(chs[1], chs[2], k=31, d=4)
        self.block4 = self._conv_block(chs[2], chs[3], k=31, d=8)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(chs[3], 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def _conv_block(cin, cout, k=15, d=1):
        pad = ((k - 1) // 2) * d
        return nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=1, padding=pad, dilation=d, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, kernel_size=3, stride=1, padding=d, dilation=d, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool1d(x, 2)
        x = self.block2(x)
        x = F.max_pool1d(x, 2)
        x = self.block3(x)
        x = F.max_pool1d(x, 2)
        x = self.block4(x)
        x = self.head(x)
        return x

# ====================== LABEL CANONICALIZATION ======================
def build_canon(classes: List[str]):
    cls = [str(c).strip() for c in classes]
    return {c.lower(): c for c in cls}

def canon(name: Optional[str], CANON: Dict[str,str]) -> Optional[str]:
    if name is None: return None
    s = str(name).strip()
    return CANON.get(s.lower(), s)

def map_log_to_model(log_label: Optional[str], CANON: Dict[str,str], CLASSES: List[str]) -> Optional[str]:
    """Map raw log label (e.g., 'NO JAM', 'u1.1') to a class present in the model; else None."""
    if log_label is None:
        return None
    key = log_label.strip().lower()
    mapped = LOG_TO_MODEL.get(key, None)
    mapped = canon(mapped, CANON) if mapped is not None else None
    if mapped is not None and mapped not in CLASSES:
        return None
    return mapped

# ====================== NOJAM VETO (OPTIONAL) ======================
USE_NOJAM_VETO = True
P_TOP_MIN       = 0.50   # if max prob < this and p(NoJam) ≥ P_NOJAM_MIN → NoJam
P_NOJAM_MIN     = 0.50
ENERGY_RMS_MAX  = 0.12   # very low absolute power → NoJam if p(NoJam) ≥ P_NOJAM_LOWPOW
P_NOJAM_LOWPOW  = 0.35

def apply_nojam_veto(pred_idx: Optional[int],
                     probs: Optional[np.ndarray],
                     classes: List[str],
                     pre_rms: float) -> (Optional[int], bool, Dict[str,float]):
    if not USE_NOJAM_VETO or pred_idx is None or probs is None:
        return pred_idx, False, {}
    p_top = float(np.max(probs)) if probs.size else 0.0
    p_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    p_nj = float(p_dict.get("NoJam", 0.0))
    veto = False
    if (p_top < P_TOP_MIN and p_nj >= P_NOJAM_MIN) or (pre_rms <= ENERGY_RMS_MAX and p_nj >= P_NOJAM_LOWPOW):
        if "NoJam" in classes:
            pred_idx = int(classes.index("NoJam"))
            veto = True
    meta = {"p_top": p_top, "p_nojam": p_nj, "pre_rms": float(pre_rms)}
    meta.update(p_dict)
    return pred_idx, veto, meta

# ====================== PLOTTING ======================
def plot_and_save(block_idx, x, fs, wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt,
                  log_label, gt_label, pred_label, pred_proba_dict, out_dir,
                  nperseg=NPERSEG, noverlap=NOVERLAP, dpi=DPI_FIG,
                  remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB):
    if x.size < 8:
        return None
    xx = x - np.mean(x) if remove_dc else x
    nperseg_eff = min(int(nperseg), len(xx))
    noverlap_eff = min(int(noverlap), max(0, nperseg_eff - 1))
    f, t, Z = stft(xx, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                   return_onesided=False, boundary=None, padded=False)
    if t.size < 2:
        nperseg_eff = max(16, min(len(xx)//4, nperseg_eff))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = stft(xx, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                       return_onesided=False, boundary=None, padded=False)
    Z = np.fft.fftshift(Z, axes=0); f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)
    tt = np.arange(len(xx), dtype=np.float32) / fs
    I = np.real(xx); Q = np.imag(xx)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2,1,1)
    if t.size >= 2:
        pcm = ax1.pcolormesh(t, f, S_dB, shading="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax1, label="dB")
    else:
        im = ax1.imshow(S_dB, aspect="auto", origin="lower",
                        extent=[0.0, max(1.0/fs, nperseg_eff/fs), f[0], f[-1]],
                        vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax1, label="dB")

    jam_txt = f" | Jammer: {log_label}" if log_label else " | Jammer: (none)"
    gt_txt  = f" | GT: {gt_label}" if gt_label is not None else " | GT: Unknown"

    if pred_label is None:
        pred_txt = " | Pred: (model failed)"
    else:
        if isinstance(pred_proba_dict, dict) and pred_label in pred_proba_dict:
            pred_txt = f" | Pred: {pred_label} ({pred_proba_dict[pred_label]:.2f})"
        else:
            pred_txt = f" | Pred: {pred_label}"

    title = (f"Spectrogram (BBSamples #{block_idx})  |  GPS week {wnc}  |  "
             f"TOW {tow_s:.3f}s ({tow_hms})  |  UTC {utc_hms}{jam_txt}{gt_txt}{pred_txt}\n"
             f"nperseg={nperseg_eff}, noverlap={noverlap_eff}")
    ax1.set_title(title)
    ax1.set_ylabel("Frequency [Hz]")

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(tt, I, linewidth=0.7, label="I")
    ax2.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.legend(loc="upper right")
    ax2.text(0.01, 0.02, utc_iso, transform=ax2.transAxes, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fname_log  = (log_label or "nolabel").replace(" ", "_").replace("/", "-")
    fname_pred = (pred_label or "nopred").replace(" ", "_")
    out_path = Path(out_dir) / f"spec_{utc_dt.strftime('%H%M%S')}_{fname_log}_GT-{gt_label or 'Unknown'}_PRED-{fname_pred}_blk{block_idx:06d}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ====================== MAIN ======================
def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Load CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CNN checkpoint: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt.get("classes", None)
    if classes is None:
        # Fallback to a common default (will still run)
        classes = ["NoJam","Chirp","NB","CW","WB","FH"]
    classes = [str(c) for c in classes]
    CANON = build_canon(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    metrics_order = classes[:]      # fixed order for CM/report = checkpoint order

    cfg = ckpt.get("config", {}) or {}
    target_len = int(cfg.get("target_len", 2048))
    fs_model = float(cfg.get("fs", 60_000_000.0))

    model = RawIQ_CNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model classes: {classes}")
    print(f"Model expects target_len={target_len}, trained fs≈{fs_model:.2f} Hz")

    # Logbook → intervals
    intervals = parse_plaintext_logbook(LOGBOOK_PATH, LOCAL_DATE, LOCAL_UTC_OFFSET)
    print(f"Loaded {len(intervals)} intervals from logbook.")
    for a,b,lbl in intervals:
        print(f"  {a.strftime('%H:%M:%S')}Z → {b.strftime('%H:%M:%S')}Z : {lbl}")

    # Metrics accumulators (strict: only mapped GT)
    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[dict] = []

    parser = SbfParser()
    block_i = -1
    saved = 0
    next_save_t: Optional[datetime] = None

    with open(SBF_PATH, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            for blk, infos in parser.parse(chunk):
                if blk != "BBSamples":
                    continue
                block_i += 1

                x, fs = decode_bbsamples_iq(infos)
                if x is None:
                    continue
                if DECIM and DECIM > 1:
                    x = x[::DECIM]; fs = fs / DECIM

                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)

                # 30 s gating (UTC)
                if next_save_t is None:
                    stride = int(SAVE_EVERY_SEC)
                    floor = utc_dt.replace(second=(utc_dt.second // stride) * stride, microsecond=0)
                    if floor > utc_dt:
                        floor -= timedelta(seconds=stride)
                    next_save_t = floor + timedelta(seconds=stride)

                if utc_dt < next_save_t:
                    continue
                while utc_dt >= next_save_t + timedelta(seconds=SAVE_EVERY_SEC):
                    next_save_t += timedelta(seconds=SAVE_EVERY_SEC)

                # ---------- Preprocess for CNN ----------
                # Keep a copy for plotting (pre-normalization)
                x_raw = x
                pre_rms = float(np.sqrt(np.mean(np.abs(x_raw)**2) + 1e-12))

                # center-crop or pad to target_len
                N = x_raw.size
                T = int(target_len)
                if N > T:
                    k0 = (N - T) // 2
                    z = x_raw[k0:k0+T]
                elif N < T:
                    z = np.zeros(T, dtype=np.complex64)
                    z[:N] = x_raw.astype(np.complex64, copy=False)
                else:
                    z = x_raw.astype(np.complex64, copy=False)

                # per-sample normalization (exactly as training dataset)
                z = z - np.mean(z)
                rms = float(np.sqrt(np.mean(np.abs(z)**2) + 1e-12))
                z = (z / rms).astype(np.complex64, copy=False)

                xb = np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)], axis=0)  # (2, T)
                xb_t = torch.from_numpy(xb[None, ...]).to(device)  # (1, 2, T)

                with torch.no_grad():
                    logits = model(xb_t)               # (1, K)
                    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]  # (K,)
                    pred_i = int(np.argmax(probs))

                # Optional NoJam veto
                pred_i_v, veto_applied, veto_meta = apply_nojam_veto(pred_i, probs, classes, pre_rms)
                pred_i = pred_i_v

                # Labels (GT from logbook; canonicalized & present in model)
                log_label_raw = None
                try:
                    log_label_raw = next(lbl for a,b,lbl in intervals if a <= utc_dt < b)
                except StopIteration:
                    pass
                gt_label = map_log_to_model(log_label_raw, CANON, classes)

                # accumulate metrics when GT is known (strict)
                if gt_label is not None and pred_i is not None:
                    y_true.append(int(class_to_idx[gt_label]))
                    y_pred.append(int(pred_i))

                if DEBUG_PRINT_SAMPLE_LABELS:
                    pred_name_dbg = classes[pred_i] if pred_i is not None else None
                    print(f"[{utc_iso}] GT={gt_label} | Pred={pred_name_dbg} | Veto={veto_applied}")

                # save plot
                if SAVE_IMAGES:
                    pred_name = classes[pred_i] if pred_i is not None else None
                    # Build a small dict of class→prob for title
                    proba_dict = {classes[k]: float(probs[k]) for k in range(len(classes))}
                    _ = plot_and_save(
                        block_idx=block_i, x=x_raw, fs=fs,
                        wnc=wnc, tow_s=tow_s, tow_hms=tow_hms,
                        utc_hms=utc_hms, utc_iso=utc_iso, utc_dt=utc_dt,
                        log_label=log_label_raw, gt_label=gt_label,
                        pred_label=pred_name, pred_proba_dict=proba_dict,
                        out_dir=out_dir,
                        nperseg=NPERSEG, noverlap=NOVERLAP,
                        remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB
                    )
                    saved += 1
                    if saved % 200 == 0:
                        print(f"Saved {saved} figures...")

                # per-sample row
                row = {
                    "block_idx": block_i,
                    "utc_iso": utc_iso,
                    "gps_week": int(wnc),
                    "tow_s": float(tow_s),
                    "log_label_raw": log_label_raw,
                    "gt_label": gt_label,
                    "pred_label_raw": (classes[pred_i] if pred_i is not None else None),
                    "veto_applied": bool(veto_applied),
                    "fs_hz": float(fs),
                    "pre_rms": float(pre_rms),
                }
                # add per-class probabilities
                for k, cname in enumerate(classes):
                    row[f"p_{cname}"] = float(probs[k])
                # add veto diagnostics
                if veto_meta:
                    row["veto_p_top"] = float(veto_meta.get("p_top", 0.0))
                    row["veto_p_nojam"] = float(veto_meta.get("p_nojam", 0.0))
                    row["veto_pre_rms"] = float(veto_meta.get("pre_rms", 0.0))

                rows.append(row)
                next_save_t = next_save_t + timedelta(seconds=SAVE_EVERY_SEC)

    # ---- metrics (strict on known GT)
    print("\n=== METRICS (strict, only mapped GT) ===")
    if len(y_true) == 0:
        print("No blocks with mapped ground-truth. Refine LOG_TO_MODEL using the official plan.")
    else:
        ct_true = Counter(y_true); ct_pred = Counter(y_pred)
        # Pretty-print counts by name
        gt_counts = {classes[i]: int(ct_true.get(i,0)) for i in range(len(classes))}
        pd_counts = {classes[i]: int(ct_pred.get(i,0)) for i in range(len(classes))}
        print("GT counts:", gt_counts)
        print("Pred counts:", pd_counts)

        labels_for_metrics = list(range(len(classes)))   # indices 0..K-1
        cm = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion matrix (labels order):", classes)
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=labels_for_metrics,
                                    target_names=classes, zero_division=0))

        # Plot CM
        fig = plt.figure(figsize=(1.1*len(classes)+2, 1.0*len(classes)+2))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(OUT_DIR) / "confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # Write textual metrics
        with open(Path(OUT_DIR) / "metrics.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Accuracy: {acc:.6f}\n")
            fh.write(f"Labels (fixed order): {classes}\n")
            fh.write("GT counts:\n")
            for i,c in enumerate(classes):
                fh.write(f"  {c}: {int(ct_true.get(i,0))}\n")
            fh.write("Pred counts:\n")
            for i,c in enumerate(classes):
                fh.write(f"  {c}: {int(ct_pred.get(i,0))}\n")
            fh.write("Confusion matrix (rows=True, cols=Pred):\n")
            for r in cm:
                fh.write(",".join(map(str, r)) + "\n")
            fh.write("\nClassification report:\n")
            fh.write(classification_report(y_true, y_pred, labels=labels_for_metrics,
                                           target_names=classes, zero_division=0))

    # per-sample CSV
    if SAVE_PER_SAMPLE_CSV and rows:
        csv_path = Path(OUT_DIR) / "samples_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            # dynamic header: core fields + per-class p_*
            core = ["block_idx","utc_iso","gps_week","tow_s",
                    "log_label_raw","gt_label","pred_label_raw","veto_applied",
                    "fs_hz","pre_rms","veto_p_top","veto_p_nojam","veto_pre_rms"]
            per_class = [f"p_{c}" for c in classes]
            fieldnames = core + per_class
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote per-sample log to: {csv_path}")

    if SUMMARY_JSON:
        js = {
            "sbf_path": SBF_PATH,
            "model_path": MODEL_PATH,
            "model_classes": classes,
            "target_len": target_len,
            "trained_fs_hz": fs_model,
            "log_to_model_mapping_used": LOG_TO_MODEL,
            "n_images_saved": saved if SAVE_IMAGES else 0,
            "save_every_sec": SAVE_EVERY_SEC,
            "decim": DECIM,
            "spectrogram_plot": {
                "nperseg": NPERSEG, "noverlap": NOVERLAP,
                "vmin_db": VMIN_DB, "vmax_db": VMAX_DB
            },
            "veto": {
                "enabled": USE_NOJAM_VETO,
                "P_TOP_MIN": P_TOP_MIN,
                "P_NOJAM_MIN": P_NOJAM_MIN,
                "ENERGY_RMS_MAX": ENERGY_RMS_MAX,
                "P_NOJAM_LOWPOW": P_NOJAM_LOWPOW
            }
        }
        with open(Path(OUT_DIR) / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(js, fh, indent=2)
        print("Wrote summary.json")

if __name__ == "__main__":
    # Small nicety: avoid MKL/OMP oversubscription on big servers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
