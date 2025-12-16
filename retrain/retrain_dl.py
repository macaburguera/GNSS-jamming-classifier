#!/usr/bin/env python3
"""
finetune_cnn_from_labels.py

Fine-tune a *spectrogram deep learning* jammer classifier (CNN/SE-CNN/ViT)
that was trained on synthetic .mat data, using REAL labelled NPZ samples from
your labelling GUI (labels CSV pointing to .npz with iq + fs_hz).

Key goals (mirrors your XGB finetune script idea):
- Keep the original 4-class output head: ["NoJam","Chirp","NB","WB"]
  even if REAL data has zero samples for some class (e.g. WB).
- Adapt decision boundaries to REAL data with gentle updates (low LR).
- Optional rehearsal: mix in a subset of synthetic samples (e.g. WB only)
  to reduce catastrophic forgetting.

Run:
  python finetune_cnn_from_labels.py

Edits:
  Only edit CONFIG below (no CLI args).

Expected REAL labels CSV columns (same spirit as your XGB script):
  label, iq_path, block_idx, gps_week, tow_s, utc_iso, sbf_path
(Extra columns ignored.)

Expected NPZ keys:
  - "iq" (complex IQ array OR Nx2 real/imag)
  - "fs_hz" (scalar or size-1 array)
  - optional: "metadata" (ignored)

Outputs (under OUT_ROOT/RUN_NAME/):
  - model_finetuned.pt
  - model_best_state.pt
  - curves.png
  - history.csv
  - val_cm.csv + val_cm.png + val_cm_rownorm.png
  - val_report.csv
  - test_cm.csv + test_cm.png + test_cm_rownorm.png
  - test_report.csv
  - test_preds.npz (y_true, y_pred, probs, classes)
  - pred_log_test.csv
  - run_meta.json
  - summary.txt
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ---------------- PyTorch ----------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
    HAVE_TORCH = True
except Exception as e:
    HAVE_TORCH = False
    TORCH_ERR = str(e)

# ---------------- SciPy STFT ----------------
import scipy.io as sio
import h5py
from scipy.signal import stft as scipy_stft


# =============================================================================
# CONFIG (EDIT THESE)
# =============================================================================

CONFIG = dict(
    # ---------------- REAL labelled dataset ----------------
    LABELS_CSV=r"E:\Jammertest23\23.09.20 - Jammertest 2023 - Day 3\Roadside test\alt01004-labelled\alt01004_labels.csv",

    # ---------------- Model to finetune ----------------
    # This should be the synthetic-trained bundle produced by train_eval_cnn_spectrogram.py (model.pt)
    MODEL_IN=r"..\artifacts\jammertest_sim_DL\spec_run_20251215_230651\model.pt",

    # Output
    OUT_ROOT=str(Path(r"..\artifacts\finetuned_DL")),
    RUN_NAME=None,  # None -> auto timestamped

    # Always keep the original 4 classes (even if absent in real data)
    CLASSES=["NoJam", "Chirp", "NB", "WB"],
    IGNORED_LABELS={"Interference"},

    # NPZ keys
    NPZ_IQ_KEY="iq",
    NPZ_FS_KEY="fs_hz",

    # Splitting
    SPLIT_MODE="random",# "time" or "random"
    TRAIN_FRAC=0.70,
    VAL_FRAC=0.15,
    SEED=42,

    # If a class exists in the dataset but ends up missing in VAL/TEST due to splitting,
    # attempt to move a few samples across splits to ensure coverage.
    # (If the class truly has 0 samples overall, nothing to do.)
    ENSURE_MIN_PER_CLASS_IN_VAL_TEST=True,
    MIN_PER_CLASS_VAL=1,
    MIN_PER_CLASS_TEST=1,

    # Fine-tuning hyperparams (gentle by default)
    BATCH_SIZE=128,
    FINETUNE_EPOCHS=50,
    LR=2e-4,
    WEIGHT_DECAY=1e-3,
    PATIENCE=25,
    NUM_WORKERS=0,
    DEVICE=None,  # None -> auto cuda/cpu

    # Optional: freeze early layers for first N epochs (helps not to destroy features)
    FREEZE_BACKBONE=True,
    FREEZE_EPOCHS=5,

    # Data / preprocessing: default to the values stored inside MODEL_IN checkpoint config
    USE_MODEL_CONFIG_FOR_PREPROC=True,

    # If True, use fs_hz stored in each NPZ; otherwise use FS_HZ from model config (or override below)
    USE_NPZ_FS=True,

    # Overrides (only used if USE_MODEL_CONFIG_FOR_PREPROC=False OR keys missing from checkpoint)
    FS_HZ=60_000_000.0,
    TARGET_LEN=2048,
    NFFT=256,
    WIN=256,
    HOP=64,
    SPEC_MODE="logpow",        # "logpow" | "logmag" | "logpow_phase3"
    SPEC_NORM="zscore",        # "none" | "zscore" | "minmax"
    FFTSHIFT=True,
    EPS=1e-8,

    # Augmentation on IQ before STFT (train only)
    AUGMENT=True,
    CFO_JITTER=False,
    TIME_SHIFT=True,

    # Rehearsal (optional): mix synthetic samples during finetune (e.g., WB only)
    REHEARSAL_SYNTH_BASE="",           # e.g. r"D:\datasets\maca_gen\datasets_jammertest" ; "" disables
    REHEARSAL_SPLITS=("TRAIN",),       # which splits under SYNTH_BASE to draw from
    REHEARSAL_ONLY_CLASSES=("WB",),    # e.g. ("WB",) to preserve WB, or None for all
    REHEARSAL_MAX_PER_CLASS=800,       # cap per class
    REHEARSAL_ENABLE_VAL=False,        # if True, also mixes rehearsal into val (usually False)

    # Logging
    VERBOSE=True,
    BATCH_LOG_EVERY=0,         # 0=off, else prints every N batches
    WRITE_TEST_PRED_LOG=True,
)

# =============================================================================
# Small helpers
# =============================================================================

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _print(msg: str, enabled: bool=True):
    if enabled:
        print(msg, flush=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0


# =============================================================================
# Label CSV parsing
# =============================================================================

@dataclass
class SampleRow:
    label: str
    iq_path: Path
    block_idx: int
    gps_week: Optional[int]
    tow_s: Optional[float]
    utc_iso: str
    sbf_path: str

def _resolve_path_maybe_relative(p: str, base_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    p1 = (base_dir / pp).resolve()
    if p1.exists():
        return p1
    return (base_dir / pp.name).resolve()

def read_labels_csv(csv_path: Path, ignored: set) -> List[SampleRow]:
    base_dir = csv_path.parent
    rows: List[SampleRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            label = (r.get("label") or "").strip()
            if not label or label in ignored:
                continue

            iq_path_raw = (r.get("iq_path") or "").strip()
            if not iq_path_raw:
                continue
            iq_path = _resolve_path_maybe_relative(iq_path_raw, base_dir)

            try:
                block_idx = int(r.get("block_idx", "-1"))
            except Exception:
                block_idx = -1

            gps_week = None
            tow_s = None
            try:
                v = r.get("gps_week")
                gps_week = int(v) if v not in (None, "", "None") else None
            except Exception:
                pass
            try:
                v = r.get("tow_s")
                tow_s = float(v) if v not in (None, "", "None") else None
            except Exception:
                pass

            utc_iso = (r.get("utc_iso") or "").strip()
            sbf_path = (r.get("sbf_path") or "").strip()

            rows.append(SampleRow(
                label=label,
                iq_path=iq_path,
                block_idx=block_idx,
                gps_week=gps_week,
                tow_s=tow_s,
                utc_iso=utc_iso,
                sbf_path=sbf_path,
            ))
    return rows


# =============================================================================
# NPZ + IQ helpers
# =============================================================================

def safe_npz_load(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except Exception:
        return None

def to_complex_1d(x) -> np.ndarray:
    x = np.asarray(x)
    # Nx2 real/imag
    if np.isrealobj(x) and x.ndim == 2 and x.shape[1] == 2:
        x = x[:, 0] + 1j * x[:, 1]
    # structured dtypes with real/imag
    if getattr(x, "dtype", None) is not None and x.dtype.names:
        names = {n.lower(): n for n in x.dtype.names}
        r = names.get("real") or names.get("re") or names.get("r")
        i = names.get("imag") or names.get("im") or names.get("i")
        if r and i:
            x = x[r] + 1j * x[i]
    return np.array(x).ravel(order="F").astype(np.complex64)

def center_crop_or_pad(z: np.ndarray, target_len: int) -> np.ndarray:
    z = np.asarray(z)
    N = z.size
    T = int(target_len)
    if N == T:
        return z
    if N > T:
        k0 = (N - T) // 2
        return z[k0:k0+T]
    out = np.zeros(T, dtype=z.dtype)
    out[:N] = z
    return out

def normalize_iq(z: np.ndarray) -> np.ndarray:
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z)**2) + 1e-12)
    return (z / rms).astype(np.complex64, copy=False)

def iq_augment(z: np.ndarray, fs: float, rng: np.random.Generator,
               augment: bool, cfo_jitter: bool, time_shift: bool) -> np.ndarray:
    if augment:
        amp = float(rng.uniform(0.9, 1.1))
        phi = float(rng.uniform(-np.pi, np.pi))
        z = amp * z * np.exp(1j * phi)
    if cfo_jitter:
        f_off = float(rng.uniform(-2e5, 2e5))  # ±200 kHz
        n = z.size
        t = np.arange(n, dtype=np.float32) / float(fs)
        z = z * np.exp(1j * (2*np.pi * f_off * t))
    if time_shift:
        sh = int(rng.integers(-64, 65))
        if sh != 0:
            z = np.roll(z, sh)
    return z

def make_spectrogram(
    z: np.ndarray,
    fs: float,
    nfft: int,
    win: int,
    hop: int,
    spec_mode: str,
    spec_norm: str,
    eps: float,
    do_fftshift: bool
) -> np.ndarray:
    """
    Returns (C, F, T) float32.
    """
    noverlap = max(0, int(win) - int(hop))
    _, _, Z = scipy_stft(
        z,
        fs=float(fs),
        window="hann",
        nperseg=int(win),
        noverlap=int(noverlap),
        nfft=int(nfft),
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
        axis=-1
    )  # (F, T) complex

    if do_fftshift:
        Z = np.fft.fftshift(Z, axes=0)

    mag2 = (np.abs(Z)**2).astype(np.float32)
    mag = np.sqrt(mag2 + eps).astype(np.float32)

    if spec_mode == "logpow":
        S = np.log(mag2 + eps).astype(np.float32)[None, :, :]              # (1,F,T)
    elif spec_mode == "logmag":
        S = np.log(mag + eps).astype(np.float32)[None, :, :]
    elif spec_mode == "logpow_phase3":
        phase = np.angle(Z).astype(np.float32)
        lp = np.log(mag2 + eps).astype(np.float32)
        S = np.stack([lp, np.cos(phase), np.sin(phase)], axis=0).astype(np.float32)  # (3,F,T)
    else:
        raise ValueError(f"Unknown spec_mode: {spec_mode}")

    if spec_norm == "none":
        return S
    if spec_norm == "zscore":
        mu = float(S.mean())
        sd = float(S.std() + 1e-6)
        return ((S - mu) / sd).astype(np.float32)
    if spec_norm == "minmax":
        mn = float(S.min())
        mx = float(S.max())
        den = (mx - mn) if (mx - mn) > 1e-6 else 1.0
        return ((S - mn) / den).astype(np.float32)

    raise ValueError(f"Unknown spec_norm: {spec_norm}")


# =============================================================================
# Optional: MAT readers for rehearsal dataset (same as your train script)
# =============================================================================

def _h5_find_by_name(h5obj, target):
    for k, item in h5obj.items():
        if k == target:
            return item
        if isinstance(item, h5py.Group):
            hit = _h5_find_by_name(item, target)
            if hit is not None:
                return hit
    return None

def load_mat_var(path: Path, varname: str):
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        if varname not in m:
            raise KeyError
        return np.asarray(m[varname])
    except Exception:
        with h5py.File(path, "r") as f:
            node = f.get(varname, None) or _h5_find_by_name(f, varname)
            if node is None:
                raise KeyError(f"'{varname}' not found in {path.name}")
            if isinstance(node, h5py.Dataset):
                return np.asarray(node[()])
            if isinstance(node, h5py.Group):
                keys = {k.lower(): k for k in node.keys()}
                if "real" in keys and "imag" in keys:
                    return np.asarray(node[keys["real"]][()]) + 1j*np.asarray(node[keys["imag"]][()])
                for v in node.values():
                    if isinstance(v, h5py.Dataset):
                        return np.asarray(v[()])
            raise TypeError(f"Unsupported HDF5 node for '{varname}' in {path.name}")

def list_split_files_mat(base: Path, split: str, classes: List[str], cap_per_class: Optional[int]=None):
    files, labels = [], []
    root = base / split
    for lab, cls in enumerate(classes):
        d = root / cls
        mats = sorted(d.glob("*.mat")) if d.exists() else []
        if cap_per_class is not None:
            mats = mats[:cap_per_class]
        for p in mats:
            files.append(p)
            labels.append(lab)
    return files, np.array(labels, int)


# =============================================================================
# Splitting helpers
# =============================================================================

def make_splits(
    y: np.ndarray,
    used_rows: List[SampleRow],
    split_mode: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    N = y.size
    ntr = int(round(train_frac * N))
    nva = int(round(val_frac * N))

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N).astype(np.int64)
        return perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    gps_week = np.array([r.gps_week if r.gps_week is not None else -1 for r in used_rows], dtype=np.int64)
    tow_s = np.array([r.tow_s if r.tow_s is not None else -1.0 for r in used_rows], dtype=np.float64)
    block_idx = np.array([r.block_idx for r in used_rows], dtype=np.int64)
    order = np.lexsort((block_idx, tow_s, gps_week)).astype(np.int64)
    return order[:ntr], order[ntr:ntr + nva], order[ntr + nva:]


def ensure_min_per_class(
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    y: np.ndarray,
    K: int,
    min_val: int,
    min_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If a class exists overall but is missing in VAL/TEST, move some samples from TRAIN into that split.
    Keeps sizes roughly stable (we just append to val/test; you can re-tune TRAIN_FRAC if you care).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).reshape(-1)

    idx_tr = idx_tr.astype(np.int64).tolist()
    idx_va = idx_va.astype(np.int64).tolist()
    idx_te = idx_te.astype(np.int64).tolist()

    overall = [int(np.sum(y == c)) for c in range(K)]
    for c in range(K):
        if overall[c] == 0:
            continue  # truly absent

        # VAL
        have_val = int(np.sum(y[np.array(idx_va, dtype=np.int64)] == c)) if idx_va else 0
        need = max(0, int(min_val) - have_val)
        if need > 0:
            candidates = [i for i in idx_tr if int(y[i]) == c]
            if candidates:
                take = candidates if len(candidates) <= need else rng.choice(candidates, size=need, replace=False).tolist()
                for t in take:
                    idx_tr.remove(t)
                    idx_va.append(int(t))

        # TEST
        have_test = int(np.sum(y[np.array(idx_te, dtype=np.int64)] == c)) if idx_te else 0
        need = max(0, int(min_test) - have_test)
        if need > 0:
            candidates = [i for i in idx_tr if int(y[i]) == c]
            if candidates:
                take = candidates if len(candidates) <= need else rng.choice(candidates, size=need, replace=False).tolist()
                for t in take:
                    idx_tr.remove(t)
                    idx_te.append(int(t))

    return np.array(idx_tr, np.int64), np.array(idx_va, np.int64), np.array(idx_te, np.int64)


# =============================================================================
# Datasets
# =============================================================================

class RealNPZSpecDataset(Dataset):
    def __init__(
        self,
        rows: List[SampleRow],
        y: np.ndarray,
        *,
        classes: List[str],
        npz_iq_key: str,
        npz_fs_key: str,
        target_len: int,
        fs_hz_default: float,
        use_npz_fs: bool,
        train: bool,
        seed: int,
        nfft: int,
        win: int,
        hop: int,
        spec_mode: str,
        spec_norm: str,
        eps: float,
        fftshift: bool,
        augment: bool,
        cfo_jitter: bool,
        time_shift: bool,
    ):
        self.rows = list(rows)
        self.y = np.asarray(y, int).reshape(-1)
        self.classes = list(classes)

        self.npz_iq_key = str(npz_iq_key)
        self.npz_fs_key = str(npz_fs_key)

        self.target_len = int(target_len)
        self.fs_hz_default = float(fs_hz_default)
        self.use_npz_fs = bool(use_npz_fs)

        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

        self.nfft = int(nfft)
        self.win = int(win)
        self.hop = int(hop)
        self.spec_mode = str(spec_mode)
        self.spec_norm = str(spec_norm)
        self.eps = float(eps)
        self.fftshift = bool(fftshift)

        self.augment = bool(augment)
        self.cfo_jitter = bool(cfo_jitter)
        self.time_shift = bool(time_shift)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        d = safe_npz_load(r.iq_path)
        if d is None or (self.npz_iq_key not in d):
            raise RuntimeError(f"Bad NPZ or missing '{self.npz_iq_key}': {r.iq_path}")

        iq_raw = to_complex_1d(d[self.npz_iq_key])

        fs = self.fs_hz_default
        if self.use_npz_fs and (self.npz_fs_key in d):
            arr = d[self.npz_fs_key]
            try:
                fs = float(arr) if np.ndim(arr) == 0 else float(np.asarray(arr).ravel()[0])
            except Exception:
                fs = self.fs_hz_default

        z = center_crop_or_pad(iq_raw, self.target_len)

        if self.train:
            z = iq_augment(z, fs, self.rng, self.augment, self.cfo_jitter, self.time_shift)

        z = normalize_iq(z)

        S = make_spectrogram(
            z=z,
            fs=fs,
            nfft=self.nfft,
            win=self.win,
            hop=self.hop,
            spec_mode=self.spec_mode,
            spec_norm=self.spec_norm,
            eps=self.eps,
            do_fftshift=self.fftshift,
        )
        return S.astype(np.float32, copy=False), int(self.y[idx])


class SynthMatSpecDataset(Dataset):
    """
    Optional rehearsal dataset from synthetic .mat files (same preprocessing).
    """
    def __init__(
        self,
        files: List[Path],
        labels: np.ndarray,
        *,
        var_name: str,
        target_len: int,
        fs: float,
        train: bool,
        seed: int,
        nfft: int,
        win: int,
        hop: int,
        spec_mode: str,
        spec_norm: str,
        eps: float,
        fftshift: bool,
        augment: bool,
        cfo_jitter: bool,
        time_shift: bool,
    ):
        self.files = list(files)
        self.labels = np.asarray(labels, int).reshape(-1)

        self.var_name = str(var_name)
        self.target_len = int(target_len)
        self.fs = float(fs)
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

        self.nfft = int(nfft)
        self.win = int(win)
        self.hop = int(hop)
        self.spec_mode = str(spec_mode)
        self.spec_norm = str(spec_norm)
        self.eps = float(eps)
        self.fftshift = bool(fftshift)

        self.augment = bool(augment)
        self.cfo_jitter = bool(cfo_jitter)
        self.time_shift = bool(time_shift)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        iq_raw = to_complex_1d(load_mat_var(p, self.var_name))
        z = center_crop_or_pad(iq_raw, self.target_len)

        if self.train:
            z = iq_augment(z, self.fs, self.rng, self.augment, self.cfo_jitter, self.time_shift)

        z = normalize_iq(z)

        S = make_spectrogram(
            z=z,
            fs=self.fs,
            nfft=self.nfft,
            win=self.win,
            hop=self.hop,
            spec_mode=self.spec_mode,
            spec_norm=self.spec_norm,
            eps=self.eps,
            do_fftshift=self.fftshift,
        )
        return S.astype(np.float32, copy=False), int(self.labels[idx])


def _collate(batch):
    xs = torch.tensor(np.stack([b[0] for b in batch], axis=0))  # (B,C,F,T)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


# =============================================================================
# Models (same as your train_eval_cnn_spectrogram.py)
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, c: int, r: int=16):
        super().__init__()
        h = max(4, c // r)
        self.fc1 = nn.Linear(c, h, bias=True)
        self.fc2 = nn.Linear(h, c, bias=True)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

def conv2d_block(cin, cout, use_se=False, se_r=16):
    block = nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )
    return nn.Sequential(block, SEBlock(cout, r=se_r)) if use_se else block

class SpecCNN2D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, use_se: bool=False):
        super().__init__()
        chs = [32, 64, 128, 192]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.GELU(),
        )
        self.b1 = conv2d_block(chs[0], chs[0], use_se=use_se)
        self.d1 = nn.MaxPool2d(2)

        self.b2 = conv2d_block(chs[0], chs[1], use_se=use_se)
        self.d2 = nn.MaxPool2d(2)

        self.b3 = conv2d_block(chs[1], chs[2], use_se=use_se)
        self.d3 = nn.MaxPool2d(2)

        self.b4 = conv2d_block(chs[2], chs[3], use_se=use_se)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(chs[3], 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.d1(x)
        x = self.b2(x); x = self.d2(x)
        x = self.b3(x); x = self.d3(x)
        x = self.b4(x)
        return self.head(x)

def sincos_posenc_2d(h: int, w: int, dim: int, device):
    if dim % 4 != 0:
        raise ValueError("VIT_EMBED_DIM must be divisible by 4.")
    q = dim // 4
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(q, device=device, dtype=torch.float32) / q
    omega = 1.0 / (10000.0 ** omega)

    out_y = yy[..., None] * omega[None, None, :]
    out_x = xx[..., None] * omega[None, None, :]
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)
    pe = torch.cat([pe_y, pe_x], dim=-1).view(1, h*w, dim)
    return pe

class SpecViT(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, patch: int, embed_dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.patch = int(patch)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=self.patch, stride=self.patch, bias=True)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                         # (B,E,Fp,Tp)
        Fp, Tp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)         # (B,N,E)

        cls = self.cls.expand(B, -1, -1)         # (B,1,E)
        x = torch.cat([cls, x], dim=1)           # (B,1+N,E)

        pe = sincos_posenc_2d(Fp, Tp, self.embed_dim, x.device)  # (1,N,E)
        pe = torch.cat([torch.zeros((1, 1, self.embed_dim), device=x.device), pe], dim=1)
        x = x + pe

        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])


# =============================================================================
# Metrics / plotting / saving
# =============================================================================

def plot_curves(history: Dict[str, List[float]], out_png: Path):
    plt.figure(figsize=(8, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, ls="--", alpha=0.4)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes, normalize: bool, title: str, out_png: Path):
    M = cm.astype(float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(M, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i, j]):.1f}%" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", color="white")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_csv_dict(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_history_csv(path: Path, history_rows: List[Dict]):
    fieldnames = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "best_val_loss", "es_pat"]
    save_csv_dict(path, history_rows, fieldnames)

@torch.no_grad()
def evaluate(model, loader, device, return_probs: bool=False):
    model.eval()
    all_y, all_pred, all_p = [], [], []
    total_correct, total_n, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        total_correct += (pred == yb).sum().item()
        total_loss += float(loss.item()) * yb.size(0)
        total_n += yb.size(0)

        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        if return_probs:
            all_p.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.zeros((0,), int)
    y_pred = np.concatenate(all_pred) if all_pred else np.zeros((0,), int)
    P = np.concatenate(all_p, axis=0) if return_probs and all_p else None
    return total_loss / max(1, total_n), total_correct / max(1, total_n), y_true, y_pred, P

def train_one_epoch(model, loader, optimizer, device, criterion, *, batch_log_every: int = 0, verbose: bool = True):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()

    for bi, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_correct += (pred == yb).sum().item()
            total_loss += float(loss.item()) * yb.size(0)
            total_n += yb.size(0)

        if batch_log_every and (bi % batch_log_every == 0):
            dt = time.time() - t0
            acc = total_correct / max(1, total_n)
            avg_loss = total_loss / max(1, total_n)
            _print(f"  [batch {bi:05d}/{len(loader):05d}] loss={avg_loss:.4f} acc={acc:.4f}  elapsed={_hms(dt)}", verbose)

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAVE_TORCH:
        raise RuntimeError(f"PyTorch not available: {TORCH_ERR}")

    C = CONFIG
    verbose = bool(C.get("VERBOSE", True))

    labels_csv = Path(C["LABELS_CSV"]).expanduser().resolve()
    if not labels_csv.exists():
        raise FileNotFoundError(f"LABELS_CSV not found: {labels_csv}")

    model_in = Path(C["MODEL_IN"]).expanduser().resolve()
    if not model_in.exists():
        raise FileNotFoundError(f"MODEL_IN not found: {model_in}")

    out_root = Path(C["OUT_ROOT"]).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_root = out_root / (C["RUN_NAME"] or ("finetune_spec_" + _now_id()))
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(C["DEVICE"]) if C["DEVICE"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(C["SEED"]))

    _print(f"[{_now()}] Starting finetune -> {run_root}", verbose)
    _print(f"[DEVICE] {device}", verbose)

    # ---------------- Load checkpoint ----------------
    ckpt = torch.load(model_in, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    classes = list(C["CLASSES"])
    K = len(classes)

    # Preproc params (prefer ckpt config)
    def _pick(key: str, default):
        if bool(C.get("USE_MODEL_CONFIG_FOR_PREPROC", True)):
            if key in ckpt_cfg:
                return ckpt_cfg[key]
        return C.get(key, default)

    fs_default = float(_pick("FS_HZ", C["FS_HZ"]))
    target_len = int(_pick("TARGET_LEN", C["TARGET_LEN"]))
    nfft = int(_pick("NFFT", C["NFFT"]))
    win = int(_pick("WIN", C["WIN"]))
    hop = int(_pick("HOP", C["HOP"]))
    spec_mode = str(_pick("SPEC_MODE", C["SPEC_MODE"]))
    spec_norm = str(_pick("SPEC_NORM", C["SPEC_NORM"]))
    fftshift = bool(_pick("FFTSHIFT", C["FFTSHIFT"]))
    eps = float(_pick("EPS", C["EPS"]))

    in_ch = 3 if spec_mode == "logpow_phase3" else 1

    # Model architecture (prefer ckpt config)
    model_kind = str(_pick("MODEL", ckpt_cfg.get("MODEL", "se_cnn")))

    _print(f"[MODEL_IN] {model_in}", verbose)
    _print(f"[ARCH] {model_kind} | in_ch={in_ch} | classes={classes}", verbose)
    _print(f"[PREPROC] fs_default={fs_default} target_len={target_len} nfft={nfft} win={win} hop={hop} mode={spec_mode} norm={spec_norm} fftshift={fftshift}", verbose)

    # Build model
    if model_kind == "cnn":
        model = SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=False)
    elif model_kind == "se_cnn":
        model = SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=True)
    elif model_kind == "vit":
        patch = int(_pick("VIT_PATCH", ckpt_cfg.get("VIT_PATCH", 8)))
        embed_dim = int(_pick("VIT_EMBED_DIM", ckpt_cfg.get("VIT_EMBED_DIM", 192)))
        depth = int(_pick("VIT_DEPTH", ckpt_cfg.get("VIT_DEPTH", 6)))
        heads = int(_pick("VIT_HEADS", ckpt_cfg.get("VIT_HEADS", 6)))
        mlp_ratio = float(_pick("VIT_MLP_RATIO", ckpt_cfg.get("VIT_MLP_RATIO", 4.0)))
        dropout = float(_pick("VIT_DROPOUT", ckpt_cfg.get("VIT_DROPOUT", 0.1)))
        model = SpecViT(in_ch=in_ch, num_classes=K, patch=patch, embed_dim=embed_dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
    else:
        raise ValueError(f"Unknown MODEL kind: {model_kind}")

    # Load weights
    state = ckpt.get("model_state", None) if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    if state is None:
        # maybe the checkpoint IS a raw state_dict
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            raise RuntimeError("MODEL_IN does not contain 'model_state' nor a usable state_dict.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        _print(f"[WARN] Missing keys when loading: {missing[:10]}{' ...' if len(missing) > 10 else ''}", verbose)
    if unexpected:
        _print(f"[WARN] Unexpected keys when loading: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}", verbose)

    model = model.to(device)

    # ---------------- Load REAL labelled rows ----------------
    rows_all = read_labels_csv(labels_csv, ignored=set(C["IGNORED_LABELS"]))
    if not rows_all:
        raise RuntimeError("No usable rows found in LABELS_CSV.")

    # Map labels to indices
    name_to_idx = {n: i for i, n in enumerate(classes)}
    rows_used: List[SampleRow] = []
    y_all: List[int] = []
    skipped_label = 0
    skipped_missing = 0
    skipped_bad = 0

    for r in rows_all:
        if r.label not in name_to_idx:
            skipped_label += 1
            continue
        if not r.iq_path.exists():
            skipped_missing += 1
            continue
        d = safe_npz_load(r.iq_path)
        if d is None or (C["NPZ_IQ_KEY"] not in d):
            skipped_bad += 1
            continue
        rows_used.append(r)
        y_all.append(int(name_to_idx[r.label]))

    if not rows_used:
        raise RuntimeError("No valid NPZ samples after filtering. Check NPZ paths/keys.")

    y_all = np.asarray(y_all, int).reshape(-1)

    _print(f"[REAL] total_rows={len(rows_all)} used={len(rows_used)} skipped_label={skipped_label} skipped_missing={skipped_missing} skipped_bad={skipped_bad}", verbose)

    # Split
    idx_tr, idx_va, idx_te = make_splits(
        y_all, rows_used,
        split_mode=str(C["SPLIT_MODE"]),
        train_frac=float(C["TRAIN_FRAC"]),
        val_frac=float(C["VAL_FRAC"]),
        seed=int(C["SEED"]),
    )

    if bool(C.get("ENSURE_MIN_PER_CLASS_IN_VAL_TEST", True)):
        idx_tr, idx_va, idx_te = ensure_min_per_class(
            idx_tr, idx_va, idx_te, y_all, K,
            min_val=int(C["MIN_PER_CLASS_VAL"]),
            min_test=int(C["MIN_PER_CLASS_TEST"]),
            seed=int(C["SEED"]),
        )

    rows_tr = [rows_used[int(i)] for i in idx_tr]
    rows_va = [rows_used[int(i)] for i in idx_va]
    rows_te = [rows_used[int(i)] for i in idx_te]
    y_tr = y_all[idx_tr]
    y_va = y_all[idx_va]
    y_te = y_all[idx_te]

    def _counts(yv: np.ndarray) -> Dict[str, int]:
        cc = np.bincount(np.asarray(yv, int), minlength=K).astype(int).tolist()
        return dict(zip(classes, cc))

    _print(f"[SPLIT] Train={len(rows_tr)} {_counts(y_tr)}", verbose)
    _print(f"[SPLIT] Val  ={len(rows_va)} {_counts(y_va)}", verbose)
    _print(f"[SPLIT] Test ={len(rows_te)} {_counts(y_te)}", verbose)

    # ---------------- Optional rehearsal from synthetic mats ----------------
    rehearsal_enabled = bool(str(C.get("REHEARSAL_SYNTH_BASE", "")).strip())
    ds_reh_tr = None
    ds_reh_va = None

    if rehearsal_enabled:
        synth_base = Path(C["REHEARSAL_SYNTH_BASE"]).expanduser().resolve()
        if not synth_base.exists():
            raise FileNotFoundError(f"REHEARSAL_SYNTH_BASE not found: {synth_base}")

        only = C.get("REHEARSAL_ONLY_CLASSES", None)
        only_set = set(only) if only else None

        reh_files_all: List[Path] = []
        reh_labels_all: List[int] = []

        for split in C.get("REHEARSAL_SPLITS", ("TRAIN",)):
            files_s, labels_s = list_split_files_mat(synth_base, str(split), classes, cap_per_class=None)
            # filter classes
            if only_set is not None:
                keep = [i for i, yy in enumerate(labels_s.tolist()) if classes[int(yy)] in only_set]
                files_s = [files_s[i] for i in keep]
                labels_s = labels_s[keep]

            # cap per class
            maxpc = int(C.get("REHEARSAL_MAX_PER_CLASS", 0) or 0)
            if maxpc > 0:
                rng = np.random.default_rng(int(C["SEED"]))
                out_idx = []
                for c in range(K):
                    idx_c = np.where(labels_s == c)[0]
                    if idx_c.size == 0:
                        continue
                    if idx_c.size > maxpc:
                        idx_c = rng.choice(idx_c, size=maxpc, replace=False)
                    out_idx.append(idx_c)
                if out_idx:
                    out_idx = np.concatenate(out_idx)
                    files_s = [files_s[int(i)] for i in out_idx]
                    labels_s = labels_s[out_idx]

            reh_files_all.extend(files_s)
            reh_labels_all.extend(labels_s.tolist())

        if reh_files_all:
            ds_reh_tr = SynthMatSpecDataset(
                reh_files_all,
                np.asarray(reh_labels_all, int),
                var_name=str(ckpt_cfg.get("VAR", "GNSS_plus_Jammer_awgn")),
                target_len=target_len,
                fs=fs_default,
                train=True,
                seed=int(C["SEED"]),
                nfft=nfft, win=win, hop=hop,
                spec_mode=spec_mode, spec_norm=spec_norm,
                eps=eps, fftshift=fftshift,
                augment=bool(C["AUGMENT"]), cfo_jitter=bool(C["CFO_JITTER"]), time_shift=bool(C["TIME_SHIFT"]),
            )
            _print(f"[REHEARSAL] enabled: {len(ds_reh_tr)} samples from {synth_base}", verbose)
        else:
            _print("[REHEARSAL] enabled but found 0 rehearsal samples (check REHEARSAL_* settings).", verbose)
            rehearsal_enabled = False

    # ---------------- Build datasets & loaders ----------------
    ds_tr_real = RealNPZSpecDataset(
        rows_tr, y_tr,
        classes=classes,
        npz_iq_key=str(C["NPZ_IQ_KEY"]),
        npz_fs_key=str(C["NPZ_FS_KEY"]),
        target_len=target_len,
        fs_hz_default=fs_default,
        use_npz_fs=bool(C["USE_NPZ_FS"]),
        train=True,
        seed=int(C["SEED"]),
        nfft=nfft, win=win, hop=hop,
        spec_mode=spec_mode, spec_norm=spec_norm,
        eps=eps, fftshift=fftshift,
        augment=bool(C["AUGMENT"]), cfo_jitter=bool(C["CFO_JITTER"]), time_shift=bool(C["TIME_SHIFT"]),
    )

    ds_va_real = RealNPZSpecDataset(
        rows_va, y_va,
        classes=classes,
        npz_iq_key=str(C["NPZ_IQ_KEY"]),
        npz_fs_key=str(C["NPZ_FS_KEY"]),
        target_len=target_len,
        fs_hz_default=fs_default,
        use_npz_fs=bool(C["USE_NPZ_FS"]),
        train=False,
        seed=int(C["SEED"]),
        nfft=nfft, win=win, hop=hop,
        spec_mode=spec_mode, spec_norm=spec_norm,
        eps=eps, fftshift=fftshift,
        augment=False, cfo_jitter=False, time_shift=False,
    )

    ds_te_real = RealNPZSpecDataset(
        rows_te, y_te,
        classes=classes,
        npz_iq_key=str(C["NPZ_IQ_KEY"]),
        npz_fs_key=str(C["NPZ_FS_KEY"]),
        target_len=target_len,
        fs_hz_default=fs_default,
        use_npz_fs=bool(C["USE_NPZ_FS"]),
        train=False,
        seed=int(C["SEED"]),
        nfft=nfft, win=win, hop=hop,
        spec_mode=spec_mode, spec_norm=spec_norm,
        eps=eps, fftshift=fftshift,
        augment=False, cfo_jitter=False, time_shift=False,
    )

    # Mix rehearsal into train (and optionally val)
    if rehearsal_enabled and ds_reh_tr is not None:
        ds_tr = ConcatDataset([ds_tr_real, ds_reh_tr])
        # Build labels for sampler over concat dataset
        y_sampler = np.concatenate([y_tr, np.asarray(reh_labels_all, int)])
    else:
        ds_tr = ds_tr_real
        y_sampler = y_tr

    if rehearsal_enabled and bool(C.get("REHEARSAL_ENABLE_VAL", False)) and ds_reh_tr is not None:
        # not recommended, but supported: reuse ds_reh_tr as "val rehearsal"
        ds_va = ConcatDataset([ds_va_real, ds_reh_tr])
    else:
        ds_va = ds_va_real

    ds_te = ds_te_real

    # Weighted sampler (train)
    counts = np.bincount(np.asarray(y_sampler, int), minlength=K).astype(float)
    counts[counts == 0] = 1.0
    samples_weight = 1.0 / np.take(counts, np.asarray(y_sampler, int))
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(C["BATCH_SIZE"]),
        sampler=sampler,
        num_workers=int(C["NUM_WORKERS"]),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(C["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(C["NUM_WORKERS"]),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=int(C["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(C["NUM_WORKERS"]),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    # ---------------- Loss / optimizer / scheduler ----------------
    # Class weights based on REAL train only (not rehearsal) to focus adaptation;
    # if you prefer combined, replace y_tr with y_sampler.
    ccounts_real = np.bincount(np.asarray(y_tr, int), minlength=K).astype(np.float64)
    ccounts_real[ccounts_real == 0] = 1.0
    cweights = (ccounts_real.sum() / ccounts_real)
    cweights = cweights / cweights.mean()
    cweights_t = torch.tensor(cweights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cweights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(C["LR"]), weight_decay=float(C["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, int(C["PATIENCE"]) // 2)
    )

    # ---------------- Optional freezing ----------------
    def _set_backbone_trainable(trainable: bool):
        # CNN: freeze everything except head; ViT: freeze everything except head
        for n, p in model.named_parameters():
            p.requires_grad = True

        if trainable:
            return

        # freeze backbone heuristics
        if isinstance(model, SpecCNN2D):
            for n, p in model.named_parameters():
                if not n.startswith("head."):
                    p.requires_grad = False
        elif isinstance(model, SpecViT):
            for n, p in model.named_parameters():
                if not n.startswith("head.") and not n.startswith("norm."):
                    p.requires_grad = False

    # ---------------- Training loop ----------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    history_rows: List[Dict] = []

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    es_pat = 0
    t_run0 = time.time()

    finetune_epochs = int(C["FINETUNE_EPOCHS"])
    freeze_backbone = bool(C.get("FREEZE_BACKBONE", True))
    freeze_epochs = int(C.get("FREEZE_EPOCHS", 0))

    _print(f"[FINETUNE] epochs={finetune_epochs} lr={C['LR']} wd={C['WEIGHT_DECAY']} patience={C['PATIENCE']}", verbose)
    if freeze_backbone and freeze_epochs > 0:
        _print(f"[FINETUNE] freeze_backbone=True for first {freeze_epochs} epochs", verbose)

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()

        if freeze_backbone and epoch <= freeze_epochs:
            _set_backbone_trainable(False)
        else:
            _set_backbone_trainable(True)

        tr_loss, tr_acc = train_one_epoch(
            model, dl_tr, optimizer, device, criterion,
            batch_log_every=int(C.get("BATCH_LOG_EVERY", 0)),
            verbose=verbose
        )

        va_loss, va_acc, _, _, _ = evaluate(model, dl_va, device, return_probs=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        scheduler.step(va_loss)
        lr = _get_lr(optimizer)
        epoch_time = time.time() - t0

        improved = va_loss < best_val - 1e-4
        if improved:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_pat = 0
        else:
            es_pat += 1

        history_rows.append(dict(
            epoch=epoch,
            lr=lr,
            train_loss=tr_loss,
            train_acc=tr_acc,
            val_loss=va_loss,
            val_acc=va_acc,
            epoch_time_s=epoch_time,
            best_val_loss=best_val,
            es_pat=es_pat,
        ))

        _print(
            f"[E{epoch:03d}/{finetune_epochs}] "
            f"lr={lr:.3e}  "
            f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val: loss={va_loss:.4f} acc={va_acc:.4f}  "
            f"{'★' if improved else ''} "
            f"(best E{best_epoch:03d} val_loss={best_val:.4f})  "
            f"es={es_pat}/{C['PATIENCE']}  "
            f"dt={_hms(epoch_time)}",
            verbose
        )

        save_history_csv(run_root / "history.csv", history_rows)

        if es_pat >= int(C["PATIENCE"]):
            _print(f"[early-stop] No val improvement for {C['PATIENCE']} epochs. Stop at E{epoch}.", verbose)
            break

    plot_curves(history, run_root / "curves.png")

    # Restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save(best_state, run_root / "model_best_state.pt")

    # Save model bundle
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "finetune_config": C,
        "preproc": dict(
            fs_default=fs_default,
            target_len=target_len,
            nfft=nfft, win=win, hop=hop,
            spec_mode=spec_mode, spec_norm=spec_norm,
            fftshift=fftshift, eps=eps,
            use_npz_fs=bool(C["USE_NPZ_FS"]),
        ),
        "base_checkpoint": str(model_in),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "started_at": _now(),
    }, run_root / "model_finetuned.pt")

    # ---------------- Final VAL + TEST evaluation ----------------
    # VAL
    va_loss, va_acc, yv_true, yv_pred, _ = evaluate(model, dl_va, device, return_probs=False)
    cm_val = confusion_matrix(yv_true, yv_pred, labels=list(range(K)))
    np.savetxt(run_root / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, "FINETUNE VAL CM", run_root / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True,  "FINETUNE VAL CM (row-norm)", run_root / "val_cm_rownorm.png")

    rep_val = classification_report(
        yv_true, yv_pred,
        labels=list(range(K)),
        target_names=classes,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "val_report.csv", rows_val, ["name", "precision", "recall", "f1-score", "support"])

    # TEST (+ probs)
    te_loss, te_acc, yt_true, yt_pred, Pte = evaluate(model, dl_te, device, return_probs=True)
    np.savez_compressed(
        run_root / "test_preds.npz",
        y_true=yt_true,
        y_pred=yt_pred,
        probs=Pte,
        classes=np.array(classes, dtype=object),
    )

    cm_test = confusion_matrix(yt_true, yt_pred, labels=list(range(K)))
    np.savetxt(run_root / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, "FINETUNE TEST CM", run_root / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True,  "FINETUNE TEST CM (row-norm)", run_root / "test_cm_rownorm.png")

    rep_test = classification_report(
        yt_true, yt_pred,
        labels=list(range(K)),
        target_names=classes,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "test_report.csv", rows_test, ["name", "precision", "recall", "f1-score", "support"])

    # Summary
    val_macro_f1 = float(f1_score(yv_true, yv_pred, average="macro", labels=list(range(K)), zero_division=0))
    test_macro_f1 = float(f1_score(yt_true, yt_pred, average="macro", labels=list(range(K)), zero_division=0))

    with open(run_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss_during_train={best_val:.6f}\n")
        f.write(f"val_loss={va_loss:.6f}  val_acc={va_acc:.6f}  val_macroF1={val_macro_f1:.6f}\n")
        f.write(f"test_loss={te_loss:.6f}  test_acc={te_acc:.6f}  test_macroF1={test_macro_f1:.6f}\n")

    # Per-sample TEST log (REAL test only)
    if bool(C.get("WRITE_TEST_PRED_LOG", True)):
        idx_to_name = {i: classes[i] for i in range(K)}
        rows_log = []
        # NOTE: dl_te_real order corresponds to rows_te_real (no shuffling).
        # But evaluate() is batched; we output in dataset order for simplicity:
        for i in range(len(rows_te)):
            rows_log.append({
                "iq_path": str(rows_te[i].iq_path),
                "label_true": idx_to_name[int(y_te[i])],
                "label_pred": idx_to_name[int(yt_pred[i])] if i < len(yt_pred) else "",
                "gps_week": int(rows_te[i].gps_week) if rows_te[i].gps_week is not None else "",
                "tow_s": float(rows_te[i].tow_s) if rows_te[i].tow_s is not None else "",
                "block_idx": int(rows_te[i].block_idx),
                "utc_iso": str(rows_te[i].utc_iso),
                "sbf_path": str(rows_te[i].sbf_path),
            })
        save_csv_dict(
            run_root / "pred_log_test.csv",
            rows_log,
            ["iq_path", "label_true", "label_pred", "gps_week", "tow_s", "block_idx", "utc_iso", "sbf_path"]
        )

    # Run meta
    meta = dict(
        started_at=_now(),
        run_dir=str(run_root),
        labels_csv=str(labels_csv),
        model_in=str(model_in),
        model_kind=model_kind,
        classes=classes,
        device=str(device),
        split_mode=str(C["SPLIT_MODE"]),
        counts=dict(train=_counts(y_tr), val=_counts(y_va), test=_counts(y_te)),
        preproc=dict(
            fs_default=fs_default,
            target_len=target_len,
            nfft=nfft, win=win, hop=hop,
            spec_mode=spec_mode, spec_norm=spec_norm,
            fftshift=fftshift, eps=eps,
            use_npz_fs=bool(C["USE_NPZ_FS"]),
        ),
        rehearsal=dict(
            enabled=bool(rehearsal_enabled),
            synth_base=str(C.get("REHEARSAL_SYNTH_BASE", "")),
            only_classes=list(C.get("REHEARSAL_ONLY_CLASSES", ()) or ()),
            max_per_class=int(C.get("REHEARSAL_MAX_PER_CLASS", 0) or 0),
            splits=list(C.get("REHEARSAL_SPLITS", ()) or ()),
        ),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        val_metrics=dict(loss=va_loss, acc=va_acc, macro_f1=val_macro_f1),
        test_metrics=dict(loss=te_loss, acc=te_acc, macro_f1=test_macro_f1),
        finetune_config=C,
        base_checkpoint_config=ckpt_cfg,
    )
    with open(run_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _print(f"[DONE] {run_root}", verbose)
    _print(f"[TIME] total={_hms(time.time() - t_run0)}  best_epoch={best_epoch}", verbose)


if __name__ == "__main__":
    main()
