#!/usr/bin/env python
"""
train_eval_cnn_spectrogram.py

Run:
  python train_eval_cnn_spectrogram.py

Edits:
  Only edit CONFIG below (no CLI args).

Dataset structure:
  BASE/
    TRAIN/<Class>/*.mat
    VAL/<Class>/*.mat
    TEST/<Class>/*.mat

MAT content:
  - GNSS_plus_Jammer_awgn : complex IQ vector (2048 typical)
  - meta (optional) with JSR_dB, CNR_dBHz or CNo_dBHz, etc.

Outputs (under OUT_ROOT/RUN_NAME or timestamped folder):
  - model.pt                     (best weights restored)
  - model_best_state.pt          (best raw state_dict snapshot)
  - curves.png                   (train/val loss+acc)
  - history.csv                  (epoch-by-epoch metrics, lr, timing)
  - val_cm.csv + val_cm.png + val_cm_rownorm.png
  - val_report.csv               (classification report)
  - test_cm.csv + test_cm.png + test_cm_rownorm.png
  - test_report.csv
  - test_preds.npz               (y_true, y_pred, probs, classes)
  - val_by_jsr/*, val_by_cnr/*, test_by_jsr/*, test_by_cnr/*
  - summary.txt                  (val/test loss/acc/macroF1)
  - run_meta.json                (all config + dataset counts)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time, json, csv
import math

import numpy as np
import scipy.io as sio
import h5py
from scipy.signal import stft as scipy_stft
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- PyTorch ----------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAVE_TORCH = True
except Exception as e:
    HAVE_TORCH = False
    TORCH_ERR = str(e)

# =============================================================================
#                               CONFIG (EDIT ME)
# =============================================================================
CONFIG = dict(
    # Dataset
    BASE=r"D:\datasets\maca_gen\datasets_jammertest",
    CLASSES=["NoJam", "Chirp", "NB", "WB"],
    VAR="GNSS_plus_Jammer_awgn",
    FS_HZ=60_000_000.0,
    TARGET_LEN=2048,
    CAP_PER_CLASS=None,  # e.g. 500 for quick runs

    # Output
    OUT_ROOT=str(Path("../artifacts/jammertest_sim_DL")),
    RUN_NAME=None,  # None -> auto timestamped

    # Training
    BATCH_SIZE=256,
    EPOCHS=80,
    LR=1e-3,
    WEIGHT_DECAY=1e-3,
    PATIENCE=60,
    NUM_WORKERS=0,
    SEED=42,
    DEVICE=None,  # None -> auto "cuda" if available else "cpu"

    # Verbose logging
    VERBOSE=True,            # print per-epoch stats + run info
    BATCH_LOG_EVERY=0,       # 0=off, else print every N batches during training epoch

    # STFT / Spectrogram
    NFFT=256,
    WIN=256,
    HOP=64,
    SPEC_MODE="logpow",       # "logpow" | "logmag" | "logpow_phase3"
    SPEC_NORM="zscore",       # "none" | "zscore" | "minmax"
    FFTSHIFT=True,            # True -> DC centered
    EPS=1e-8,

    # Augmentation (applied on IQ before STFT)
    AUGMENT=True,             # amp + phase jitter
    CFO_JITTER=False,         # small CFO jitter
    TIME_SHIFT=True,          # small circular time shift

    # Model
    MODEL="se_cnn",            # "cnn" | "se_cnn" | "vit"

    # ViT params (only used if MODEL="vit")
    VIT_PATCH=8,
    VIT_EMBED_DIM=192,         # must be divisible by 4
    VIT_DEPTH=6,
    VIT_HEADS=6,
    VIT_MLP_RATIO=4.0,
    VIT_DROPOUT=0.1,

    # Bin metrics
    JSR_BINS=[0, 10, 25, 40],
    CNR_BINS=[20, 30, 40, 60],
)
# =============================================================================


# ---------------- Small helpers ----------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0

def _print(msg: str, enabled: bool=True):
    if enabled:
        print(msg, flush=True)


# ---------------- Utilities: MAT readers ----------------
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
    """Read var from MAT v7 or v7.3; return numpy array."""
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

def load_meta_dict(path: Path) -> Dict:
    """
    Best-effort read of 'meta' as dict (works for v7 and v7.3).

    Patched to avoid NumPy 1.25+ DeprecationWarning:
      converting an ndarray (even size-1) directly to float is deprecated.
    """
    # v7
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        meta = m.get("meta", None)
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass

    # v7.3
    out: Dict = {}
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                return out
            g = f["meta"]
            for k, v in g.items():
                if not isinstance(v, h5py.Dataset):
                    continue
                arr = np.asarray(v[()])

                if arr.size == 1:
                    val = arr.reshape(()).item()  # <- safe scalar extraction

                    # decode bytes if present
                    if isinstance(val, (bytes, bytearray)):
                        try:
                            val = val.decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    # keep old behavior: numeric real scalars -> float
                    if np.isrealobj(arr) and isinstance(val, (int, float, np.integer, np.floating)):
                        out[k] = float(val)
                    else:
                        out[k] = val
                else:
                    out[k] = arr
    except Exception:
        pass

    return out

def to_complex_1d(x) -> np.ndarray:
    x = np.asarray(x)
    if np.isrealobj(x) and x.ndim == 2 and x.shape[1] == 2:
        x = x[:, 0] + 1j * x[:, 1]
    if getattr(x, "dtype", None) is not None and x.dtype.names:
        names = {n.lower(): n for n in x.dtype.names}
        r = names.get("real") or names.get("re") or names.get("r")
        i = names.get("imag") or names.get("im") or names.get("i")
        if r and i:
            x = x[r] + 1j * x[i]
    return np.array(x).ravel(order="F").astype(np.complex64)


# ---------------- Dataset scanning ----------------
def list_split_files(base: Path, split: str, classes: List[str], cap_per_class: Optional[int]=None):
    files, labels, jsr, cnr = [], [], [], []
    root = base / split
    for lab, cls in enumerate(classes):
        d = root / cls
        mats = sorted(d.glob("*.mat")) if d.exists() else []
        if cap_per_class is not None:
            mats = mats[:cap_per_class]
        for p in mats:
            files.append(p)
            labels.append(lab)
            meta = load_meta_dict(p)

            # CNR/CNo
            c = np.nan
            for k in ("CNR_dBHz", "CNo_dBHz", "CNR_dB", "CNo"):
                if k in meta:
                    try:
                        c = float(np.asarray(meta[k]).ravel()[0])
                        break
                    except Exception:
                        pass

            # JSR
            j = np.nan
            if "JSR_dB" in meta:
                try:
                    j = float(np.asarray(meta["JSR_dB"]).ravel()[0])
                except Exception:
                    pass

            jsr.append(j)
            cnr.append(c)

    return files, np.array(labels, int), np.array(jsr, float), np.array(cnr, float)

def _class_counts(labels: np.ndarray, K: int) -> List[int]:
    return np.bincount(labels, minlength=K).astype(int).tolist()


# ---------------- Spectrogram helpers ----------------
def center_crop_or_pad(z: np.ndarray, target_len: int) -> np.ndarray:
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

def normalize_iq(z: np.ndarray) -> np.ndarray:
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z)**2) + 1e-12)
    return (z / rms).astype(np.complex64, copy=False)

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
    Uses complex STFT with return_onesided=False (two-sided) and optional fftshift.
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
    )  # Z: (F, T), complex

    if do_fftshift:
        Z = np.fft.fftshift(Z, axes=0)

    mag2 = (np.abs(Z)**2).astype(np.float32)
    mag = np.sqrt(mag2 + eps).astype(np.float32)

    if spec_mode == "logpow":
        S = np.log(mag2 + eps).astype(np.float32)[None, :, :]              # (1,F,T)
    elif spec_mode == "logmag":
        S = np.log(mag + eps).astype(np.float32)[None, :, :]               # (1,F,T)
    elif spec_mode == "logpow_phase3":
        phase = np.angle(Z).astype(np.float32)
        lp = np.log(mag2 + eps).astype(np.float32)
        S = np.stack([lp, np.cos(phase), np.sin(phase)], axis=0).astype(np.float32)  # (3,F,T)
    else:
        raise ValueError(f"Unknown spec_mode: {spec_mode}")

    # per-sample normalization on spectrogram tensor
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


# ---------------- PyTorch Dataset ----------------
class JammerSpecDataset(Dataset):
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
        self.labels = np.array(labels, int)

        self.var_name = var_name
        self.target_len = int(target_len)
        self.fs = float(fs)
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

        self.nfft = int(nfft)
        self.win = int(win)
        self.hop = int(hop)
        self.spec_mode = spec_mode
        self.spec_norm = spec_norm
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
            do_fftshift=self.fftshift
        )
        return S.astype(np.float32, copy=False), int(self.labels[idx])


# ---------------- Models ----------------
class SEBlock(nn.Module):
    """Channel attention (Squeeze-and-Excitation)."""
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
    """2D CNN over spectrograms: (B,C,F,T) -> logits."""
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
    """(1, h*w, dim) 2D sin-cos pos enc. dim must be divisible by 4."""
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
    """Small ViT over spectrogram patches."""
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


# ---------------- Training helpers ----------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

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

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    P = np.concatenate(all_p, axis=0) if return_probs else None
    return total_loss / max(1, total_n), total_correct / max(1, total_n), y_true, y_pred, P


# ---------------- Plotting & metrics ----------------
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

def eval_by_bins(y_true, y_pred, classes, values: np.ndarray, edges, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)

    rows = []
    rows.append({
        "bin": "OVERALL",
        "count": int(y_true.size),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    })

    edges = np.asarray(edges, float)
    bin_labels = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges) - 1)]
    for b in range(len(edges) - 1):
        mask = (values >= edges[b]) & (values < edges[b + 1])
        if not np.any(mask):
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        cm = confusion_matrix(yt, yp, labels=list(range(K)))
        np.savetxt(out_dir / f"cm_{tag}_bin{b}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, classes, False, f"{tag} CM {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}.png")
        plot_confusion_matrix(cm, classes, True,  f"{tag} CM (row-norm) {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}_rownorm.png")
        rows.append({
            "bin": bin_labels[b],
            "count": int(mask.sum()),
            "acc": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro"),
        })

    save_csv_dict(out_dir / f"metrics_{tag}.csv", rows, ["bin", "count", "acc", "macro_f1"])


# ---------------- Main ----------------
def main():
    if not HAVE_TORCH:
        raise RuntimeError(f"PyTorch not available: {TORCH_ERR}")

    C = CONFIG
    verbose = bool(C.get("VERBOSE", True))

    base = Path(C["BASE"])
    classes = list(C["CLASSES"])
    out_root = Path(C["OUT_ROOT"])
    run_root = out_root / (C["RUN_NAME"] or ("spec_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(C["DEVICE"]) if C["DEVICE"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(C["SEED"]))

    _print(f"[{_now()}] Starting run -> {run_root}", verbose)
    _print(f"[DEVICE] {device}", verbose)
    _print(f"[DATASET] base={base}", verbose)
    _print(f"[CLASSES] {classes}", verbose)
    _print(f"[SPEC] nfft={C['NFFT']} win={C['WIN']} hop={C['HOP']} mode={C['SPEC_MODE']} norm={C['SPEC_NORM']} fftshift={C['FFTSHIFT']}", verbose)
    _print(f"[MODEL] {C['MODEL']}", verbose)

    # Scan dataset
    tr_files, tr_labels, tr_jsr, tr_cnr = list_split_files(base, "TRAIN", classes, C["CAP_PER_CLASS"])
    va_files, va_labels, va_jsr, va_cnr = list_split_files(base, "VAL", classes, C["CAP_PER_CLASS"])
    te_files, te_labels, te_jsr, te_cnr = list_split_files(base, "TEST", classes, C["CAP_PER_CLASS"])

    K = len(classes)
    _print(f"[COUNTS] Train={len(tr_files)} {dict(zip(classes, _class_counts(tr_labels, K)))}", verbose)
    _print(f"[COUNTS] Val  ={len(va_files)} {dict(zip(classes, _class_counts(va_labels, K)))}", verbose)
    _print(f"[COUNTS] Test ={len(te_files)} {dict(zip(classes, _class_counts(te_labels, K)))}", verbose)

    # Datasets
    ds_tr = JammerSpecDataset(
        tr_files, tr_labels,
        var_name=C["VAR"], target_len=C["TARGET_LEN"], fs=C["FS_HZ"], train=True, seed=C["SEED"],
        nfft=C["NFFT"], win=C["WIN"], hop=C["HOP"],
        spec_mode=C["SPEC_MODE"], spec_norm=C["SPEC_NORM"],
        eps=C["EPS"], fftshift=C["FFTSHIFT"],
        augment=C["AUGMENT"], cfo_jitter=C["CFO_JITTER"], time_shift=C["TIME_SHIFT"],
    )
    ds_va = JammerSpecDataset(
        va_files, va_labels,
        var_name=C["VAR"], target_len=C["TARGET_LEN"], fs=C["FS_HZ"], train=False, seed=C["SEED"],
        nfft=C["NFFT"], win=C["WIN"], hop=C["HOP"],
        spec_mode=C["SPEC_MODE"], spec_norm=C["SPEC_NORM"],
        eps=C["EPS"], fftshift=C["FFTSHIFT"],
        augment=False, cfo_jitter=False, time_shift=False,
    )
    ds_te = JammerSpecDataset(
        te_files, te_labels,
        var_name=C["VAR"], target_len=C["TARGET_LEN"], fs=C["FS_HZ"], train=False, seed=C["SEED"],
        nfft=C["NFFT"], win=C["WIN"], hop=C["HOP"],
        spec_mode=C["SPEC_MODE"], spec_norm=C["SPEC_NORM"],
        eps=C["EPS"], fftshift=C["FFTSHIFT"],
        augment=False, cfo_jitter=False, time_shift=False,
    )

    in_ch = 3 if C["SPEC_MODE"] == "logpow_phase3" else 1
    num_classes = len(classes)

    # Loss weights
    cweights = compute_class_weights(tr_labels, num_classes).to(device)

    # Weighted sampler
    class_counts = np.bincount(tr_labels, minlength=num_classes)
    samples_weight = 1.0 / np.take(class_counts, tr_labels)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    def _collate(batch):
        xs = torch.tensor(np.stack([b[0] for b in batch], axis=0))  # (B,C,F,T)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_tr = DataLoader(ds_tr, batch_size=C["BATCH_SIZE"], sampler=sampler,
                       num_workers=C["NUM_WORKERS"], pin_memory=True, collate_fn=_collate, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=C["BATCH_SIZE"], shuffle=False,
                       num_workers=C["NUM_WORKERS"], pin_memory=True, collate_fn=_collate, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=C["BATCH_SIZE"], shuffle=False,
                       num_workers=C["NUM_WORKERS"], pin_memory=True, collate_fn=_collate, drop_last=False)

    # Model
    if C["MODEL"] == "cnn":
        model = SpecCNN2D(in_ch=in_ch, num_classes=num_classes, use_se=False).to(device)
    elif C["MODEL"] == "se_cnn":
        model = SpecCNN2D(in_ch=in_ch, num_classes=num_classes, use_se=True).to(device)
    elif C["MODEL"] == "vit":
        model = SpecViT(
            in_ch=in_ch,
            num_classes=num_classes,
            patch=C["VIT_PATCH"],
            embed_dim=C["VIT_EMBED_DIM"],
            depth=C["VIT_DEPTH"],
            heads=C["VIT_HEADS"],
            mlp_ratio=C["VIT_MLP_RATIO"],
            dropout=C["VIT_DROPOUT"],
        ).to(device)
    else:
        raise ValueError(f"Unknown MODEL={C['MODEL']}")

    criterion = nn.CrossEntropyLoss(weight=cweights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C["LR"], weight_decay=C["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, int(C["PATIENCE"]) // 2)
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    history_rows: List[Dict] = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    es_pat = 0
    t_run0 = time.time()

    for epoch in range(1, int(C["EPOCHS"]) + 1):
        t0 = time.time()

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
            f"[E{epoch:03d}/{C['EPOCHS']}] "
            f"lr={lr:.3e}  "
            f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val: loss={va_loss:.4f} acc={va_acc:.4f}  "
            f"{'★' if improved else ''} "
            f"(best E{best_epoch:03d} val_loss={best_val:.4f})  "
            f"es={es_pat}/{C['PATIENCE']}  "
            f"dt={_hms(epoch_time)}",
            verbose
        )

        # persist history every epoch
        save_history_csv(run_root / "history.csv", history_rows)

        if es_pat >= int(C["PATIENCE"]):
            _print(f"[early-stop] No val improvement for {C['PATIENCE']} epochs. Stop at E{epoch}.", verbose)
            break

    # Save curves
    plot_curves(history, run_root / "curves.png")

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save(best_state, run_root / "model_best_state.pt")  # raw state_dict snapshot

    # Save final model bundle (best restored)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "config": C,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }, run_root / "model.pt")

    # VAL evaluation (best)
    va_loss, va_acc, yv_true, yv_pred, _ = evaluate(model, dl_va, device, return_probs=False)
    cm_val = confusion_matrix(yv_true, yv_pred, labels=list(range(num_classes)))
    np.savetxt(run_root / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, "SPEC VAL CM", run_root / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True,  "SPEC VAL CM (row-norm)", run_root / "val_cm_rownorm.png")
    rep_val = classification_report(yv_true, yv_pred, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "val_report.csv", rows_val, ["name", "precision", "recall", "f1-score", "support"])

    # TEST evaluation (best) + probs
    te_loss, te_acc, yt_true, yt_pred, Pte = evaluate(model, dl_te, device, return_probs=True)
    np.savez_compressed(
        run_root / "test_preds.npz",
        y_true=yt_true,
        y_pred=yt_pred,
        probs=Pte,
        classes=np.array(classes, dtype=object)
    )

    cm_test = confusion_matrix(yt_true, yt_pred, labels=list(range(num_classes)))
    np.savetxt(run_root / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, "SPEC TEST CM", run_root / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True,  "SPEC TEST CM (row-norm)", run_root / "test_cm_rownorm.png")
    rep_test = classification_report(yt_true, yt_pred, target_names=classes, digits=6, output_dict=True)
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "test_report.csv", rows_test, ["name", "precision", "recall", "f1-score", "support"])

    # Per-JSR/CNR metrics
    if va_jsr.size == len(yv_true):
        eval_by_bins(yv_true, yv_pred, classes, va_jsr, C["JSR_BINS"], run_root / "val_by_jsr", "val_JSR")
    if va_cnr.size == len(yv_true):
        eval_by_bins(yv_true, yv_pred, classes, va_cnr, C["CNR_BINS"], run_root / "val_by_cnr", "val_CNR")
    if te_jsr.size == len(yt_true):
        eval_by_bins(yt_true, yt_pred, classes, te_jsr, C["JSR_BINS"], run_root / "test_by_jsr", "test_JSR")
    if te_cnr.size == len(yt_true):
        eval_by_bins(yt_true, yt_pred, classes, te_cnr, C["CNR_BINS"], run_root / "test_by_cnr", "test_CNR")

    # Summary
    val_macro_f1 = f1_score(yv_true, yv_pred, average="macro")
    test_macro_f1 = f1_score(yt_true, yt_pred, average="macro")
    with open(run_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss_during_train={best_val:.6f}\n")
        f.write(f"val_loss={va_loss:.6f}  val_acc={va_acc:.6f}  val_macroF1={val_macro_f1:.6f}\n")
        f.write(f"test_loss={te_loss:.6f}  test_acc={te_acc:.6f}  test_macroF1={test_macro_f1:.6f}\n")

    # Run metadata
    meta = dict(
        started_at=_now(),
        run_dir=str(run_root),
        classes=classes,
        fs_hz=C["FS_HZ"],
        target_len=C["TARGET_LEN"],
        spec=dict(nfft=C["NFFT"], win=C["WIN"], hop=C["HOP"], mode=C["SPEC_MODE"], norm=C["SPEC_NORM"], fftshift=C["FFTSHIFT"]),
        num_train=len(tr_files),
        num_val=len(va_files),
        num_test=len(te_files),
        class_counts=dict(
            train=dict(zip(classes, _class_counts(tr_labels, K))),
            val=dict(zip(classes, _class_counts(va_labels, K))),
            test=dict(zip(classes, _class_counts(te_labels, K))),
        ),
        device=str(device),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        config=C,
    )
    with open(run_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _print(f"[DONE] {run_root}", verbose)
    _print(f"[TIME] total={_hms(time.time() - t_run0)}  best_epoch={best_epoch}", verbose)


if __name__ == "__main__":
    main()
