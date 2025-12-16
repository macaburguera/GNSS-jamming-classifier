#!/usr/bin/env python3
# validate_dl_from_labels.py
"""
Validate a *spectrogram DL model* (CNN/SE-CNN/ViT from train_eval_cnn_spectrogram.py)
on a labelled NPZ dataset produced by label_gui.py.

✅ Works with BOTH checkpoint formats:
  A) Original training bundle:  model.pt  (expects keys: classes, config, model_state)
  B) Fine-tuned bundle:        model_finetuned.pt (keys: classes, model_state, preproc, base_checkpoint, finetune_config)

What it does:
  - Reads *_labels.csv from label_gui.py (expects at least: label, iq_path)
  - Loads each .npz (iq + fs_hz)
  - Rebuilds the spectrogram pipeline exactly like training
  - Runs inference, computes accuracy, confusion matrix, classification report
  - Writes: confusion_matrix(.png/.csv), per-sample CSV, metrics.txt, summary.json

Edits:
  Only edit USER VARIABLES below (no CLI args).
"""

from __future__ import annotations

from pathlib import Path
import csv, json
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.signal import stft as scipy_stft

# ---------------- PyTorch ----------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception as e:
    HAVE_TORCH = False
    TORCH_ERR = str(e)

# ============================ USER VARIABLES ============================

# Path to the labels CSV produced by label_gui.py
LABELS_CSV = r"E:\Jammertest23\23.09.18 - Jammertest 2023 - Day 1\Altus06 - 150m\labelled\alt06001_labels.csv"

# Output directory for plots, metrics, logs, etc.
OUT_DIR    = r"E:\Jammertest23\23.09.18 - Jammertest 2023 - Day 1\plots\alt06001_eval_DL_from_labels"

# Your trained DL checkpoint:
# - Original training:  ...\run_xxx\model.pt
# - Fine-tuned bundle:  ...\finetune_xxx\model_finetuned.pt
MODEL_PT   = r"..\artifacts\finetuned_DL\finetune_spec_20251216_123411\model_finetuned.pt"

# Inference options
DEVICE = None           # None -> auto cuda if available
BATCH_SIZE = 512        # reduce if you OOM
NUM_WORKERS = 0         # unused (no DataLoader here)

# Save options
SAVE_PER_SAMPLE_CSV = True
SAVE_CONFUSION_PNG  = True
SAVE_ROW_NORM_PNG   = True
SUMMARY_JSON        = True
DEBUG_PRINT_SAMPLES = True   # per-sample GT/Pred prints (can be noisy)

# If your labels CSV contains classes your DL model does NOT have (e.g. "Interference"),
# list them here so they can appear in the confusion matrix order if present in GT.
EXTRA_LABELS = ["Interference"]

# If your NPZ uses different key names, change here:
NPZ_IQ_KEY = "iq"
NPZ_FS_KEY = "fs_hz"

# ======================================================================

EPS = 1e-8


# ====================== Spectrogram pipeline (must match training) ======================
def center_crop_or_pad(z: np.ndarray, target_len: int) -> np.ndarray:
    z = np.asarray(z)
    N = z.size
    T = int(target_len)
    if N == T:
        return z
    if N > T:
        k0 = (N - T) // 2
        return z[k0:k0 + T]
    out = np.zeros(T, dtype=z.dtype)
    out[:N] = z
    return out

def normalize_iq(z: np.ndarray) -> np.ndarray:
    z = z - np.mean(z)
    rms = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
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
    Returns (C,F,T) float32.
    Matches training: complex STFT (two-sided) + optional fftshift.
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
    )  # (F,T) complex

    if do_fftshift:
        Z = np.fft.fftshift(Z, axes=0)

    mag2 = (np.abs(Z) ** 2).astype(np.float32)
    mag = np.sqrt(mag2 + eps).astype(np.float32)

    if spec_mode == "logpow":
        S = np.log(mag2 + eps).astype(np.float32)[None, :, :]
    elif spec_mode == "logmag":
        S = np.log(mag + eps).astype(np.float32)[None, :, :]
    elif spec_mode == "logpow_phase3":
        phase = np.angle(Z).astype(np.float32)
        lp = np.log(mag2 + eps).astype(np.float32)
        S = np.stack([lp, np.cos(phase), np.sin(phase)], axis=0).astype(np.float32)
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


# ====================== Models (same as training script) ======================
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
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
    def __init__(self, in_ch: int, num_classes: int, use_se: bool = False):
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
    pe = torch.cat([pe_y, pe_x], dim=-1).view(1, h * w, dim)
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
        x = self.proj(x)                     # (B,E,Fp,Tp)
        Fp, Tp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)     # (B,N,E)

        cls = self.cls.expand(B, -1, -1)     # (B,1,E)
        x = torch.cat([cls, x], dim=1)       # (B,1+N,E)

        pe = sincos_posenc_2d(Fp, Tp, self.embed_dim, x.device)  # (1,N,E)
        pe = torch.cat([torch.zeros((1, 1, self.embed_dim), device=x.device), pe], dim=1)
        x = x + pe

        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])


# ====================== Utilities ======================
def canon_map(class_names: List[str]) -> Dict[str, str]:
    # case-insensitive canonical mapping
    return {c.lower(): c for c in class_names}

def canon(label: Optional[str], CANON: Dict[str, str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip()
    return CANON.get(s.lower(), s)

def plot_confusion(cm: np.ndarray, labels: List[str], out_png: Path, title: str, normalize: bool):
    M = cm.astype(float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 2))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _resolve_path_maybe_relative(p: str, base_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    p1 = (base_dir / pp).resolve()
    if p1.exists():
        return p1
    return (base_dir / pp.name).resolve()

def to_complex_1d(x) -> np.ndarray:
    x = np.asarray(x)
    # Nx2 real/imag
    if np.isrealobj(x) and x.ndim == 2 and x.shape[1] == 2:
        x = x[:, 0] + 1j * x[:, 1]
    # structured dtype {real,imag}
    if getattr(x, "dtype", None) is not None and x.dtype.names:
        names = {n.lower(): n for n in x.dtype.names}
        r = names.get("real") or names.get("re") or names.get("r")
        i = names.get("imag") or names.get("im") or names.get("i")
        if r and i:
            x = x[r] + 1j * x[i]
    return np.array(x).ravel(order="F").astype(np.complex64)

def _read_npz_iq_fs(npz_path: Path) -> Tuple[np.ndarray, float]:
    with np.load(npz_path, allow_pickle=True) as d:
        if NPZ_IQ_KEY not in d:
            raise KeyError(f"Missing key '{NPZ_IQ_KEY}' in {npz_path.name}")
        iq = to_complex_1d(d[NPZ_IQ_KEY])

        if NPZ_FS_KEY not in d:
            raise KeyError(f"Missing key '{NPZ_FS_KEY}' in {npz_path.name}")
        fs_arr = d[NPZ_FS_KEY]
        fs = float(fs_arr) if np.ndim(fs_arr) == 0 else float(np.asarray(fs_arr).ravel()[0])

    return iq, fs


# ====================== Load checkpoint (supports BOTH formats) ======================
def load_checkpoint_any(model_pt: Path):
    ckpt = torch.load(model_pt, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict. Unexpected format.")

    # ---- Format A: original training bundle ----
    if ("classes" in ckpt) and ("config" in ckpt) and (("model_state" in ckpt) or ("state_dict" in ckpt)):
        classes = list(ckpt["classes"])
        cfg = dict(ckpt["config"])
        state = ckpt.get("model_state", None) or ckpt.get("state_dict", None)
        return classes, cfg, state, ckpt, "train_bundle"

    # ---- Format B: finetune bundle ----
    if ("classes" in ckpt) and ("model_state" in ckpt):
        classes = list(ckpt["classes"])
        state = ckpt["model_state"]

        # Recover original training config if possible
        cfg = {}
        base_ckpt_path = ckpt.get("base_checkpoint", None)
        if base_ckpt_path:
            bp = Path(base_ckpt_path).expanduser()
            if bp.exists():
                base_ckpt = torch.load(bp, map_location="cpu")
                if isinstance(base_ckpt, dict):
                    cfg = dict(base_ckpt.get("config", {}))

        # Fall back to finetune_config if needed
        if not cfg:
            cfg = dict(ckpt.get("finetune_config", {}) or {})

        # Overlay actual preproc used during finetune
        pre = ckpt.get("preproc", {}) or {}
        if pre:
            cfg = dict(cfg)
            cfg["TARGET_LEN"] = int(pre.get("target_len", cfg.get("TARGET_LEN", 2048)))
            cfg["NFFT"]       = int(pre.get("nfft",       cfg.get("NFFT", 256)))
            cfg["WIN"]        = int(pre.get("win",        cfg.get("WIN", 256)))
            cfg["HOP"]        = int(pre.get("hop",        cfg.get("HOP", 64)))
            cfg["SPEC_MODE"]  = str(pre.get("spec_mode",  cfg.get("SPEC_MODE", "logpow")))
            cfg["SPEC_NORM"]  = str(pre.get("spec_norm",  cfg.get("SPEC_NORM", "zscore")))
            cfg["FFTSHIFT"]   = bool(pre.get("fftshift",  cfg.get("FFTSHIFT", True)))
            cfg["EPS"]        = float(pre.get("eps",      cfg.get("EPS", 1e-8)))

        cfg.setdefault("MODEL", "se_cnn")
        return classes, cfg, state, ckpt, "finetune_bundle"

    raise RuntimeError(f"Unrecognized checkpoint format. Keys: {sorted(list(ckpt.keys()))[:40]} ...")

def build_model_from_cfg(cfg: dict, classes: List[str]) -> nn.Module:
    spec_mode = str(cfg.get("SPEC_MODE", "logpow"))
    in_ch = 3 if spec_mode == "logpow_phase3" else 1
    num_classes = len(classes)
    model_kind = str(cfg.get("MODEL", "se_cnn"))

    if model_kind == "cnn":
        return SpecCNN2D(in_ch=in_ch, num_classes=num_classes, use_se=False)
    if model_kind == "se_cnn":
        return SpecCNN2D(in_ch=in_ch, num_classes=num_classes, use_se=True)
    if model_kind == "vit":
        return SpecViT(
            in_ch=in_ch,
            num_classes=num_classes,
            patch=int(cfg.get("VIT_PATCH", 8)),
            embed_dim=int(cfg.get("VIT_EMBED_DIM", 192)),
            depth=int(cfg.get("VIT_DEPTH", 6)),
            heads=int(cfg.get("VIT_HEADS", 6)),
            mlp_ratio=float(cfg.get("VIT_MLP_RATIO", 4.0)),
            dropout=float(cfg.get("VIT_DROPOUT", 0.1)),
        )
    raise ValueError(f"Unknown MODEL={model_kind}")


# ====================== Main ======================
def main():
    if not HAVE_TORCH:
        raise RuntimeError(f"PyTorch not available: {TORCH_ERR}")

    labels_csv_path = Path(LABELS_CSV).expanduser().resolve()
    out_dir = Path(OUT_DIR).expanduser().resolve()
    model_pt = Path(MODEL_PT).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = labels_csv_path.parent

    device = torch.device(DEVICE) if DEVICE else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Labels CSV: {labels_csv_path}")
    print(f"Output dir: {out_dir}")
    print(f"Model pt:   {model_pt}")
    print(f"Device:     {device}")

    # Load checkpoint + rebuild model (supports both formats)
    classes, cfg, state, ckpt, ckpt_kind = load_checkpoint_any(model_pt)
    classes = list(classes)
    print(f"Checkpoint kind: {ckpt_kind}")
    print("Model classes:", classes)

    # Pull spectrogram params from cfg (critical!)
    target_len = int(cfg.get("TARGET_LEN", 2048))
    nfft = int(cfg.get("NFFT", 256))
    win = int(cfg.get("WIN", 256))
    hop = int(cfg.get("HOP", 64))
    spec_mode = str(cfg.get("SPEC_MODE", "logpow"))
    spec_norm = str(cfg.get("SPEC_NORM", "zscore"))
    fftshift = bool(cfg.get("FFTSHIFT", True))
    eps = float(cfg.get("EPS", 1e-8))

    print(f"[SPEC] target_len={target_len} nfft={nfft} win={win} hop={hop} mode={spec_mode} norm={spec_norm} fftshift={fftshift}")

    model = build_model_from_cfg(cfg, classes)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    model.to(device)
    model.eval()

    # Canonical label namespace for this validation run
    model_label_namespace = classes[:] + [c for c in EXTRA_LABELS if c not in classes]
    CANON = canon_map(model_label_namespace)

    # Read CSV -> build evaluation list
    items: List[dict] = []
    n_total = 0
    n_skipped_bad_label = 0
    n_skipped_missing_npz = 0
    n_skipped_bad_npz = 0

    with open(labels_csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            label_raw = row.get("label", "") or ""
            gt = canon(label_raw, CANON)

            if gt not in model_label_namespace:
                n_skipped_bad_label += 1
                if DEBUG_PRINT_SAMPLES:
                    sid = row.get("sample_id", "") or row.get("utc_iso", "") or ""
                    print(f"[SKIP] {sid}: unknown label '{label_raw}'")
                continue

            iq_path_raw = row.get("iq_path", "") or ""
            if not iq_path_raw:
                n_skipped_missing_npz += 1
                continue

            iq_path = _resolve_path_maybe_relative(iq_path_raw, base_dir)
            if not iq_path.exists():
                n_skipped_missing_npz += 1
                if DEBUG_PRINT_SAMPLES:
                    print(f"[SKIP] missing NPZ: {iq_path_raw}")
                continue

            # Keep
            sample_id = row.get("sample_id", "") or iq_path.stem
            items.append({
                "row": row,
                "sample_id": sample_id,
                "label_raw": label_raw,
                "gt_label": gt,
                "iq_path": iq_path,
            })

    print("\n=== DATASET SUMMARY ===")
    print(f"Total rows in CSV:         {n_total}")
    print(f"Kept for evaluation:       {len(items)}")
    print(f"Skipped (unknown label):   {n_skipped_bad_label}")
    print(f"Skipped (missing NPZ):     {n_skipped_missing_npz}")
    print(f"Skipped (bad NPZ):         {n_skipped_bad_npz}")

    if not items:
        print("No items to evaluate. Check LABELS_CSV / label names / iq_path resolution.")
        return

    # Inference (batched)
    y_true: List[str] = []
    y_pred: List[str] = []
    rows_log: List[dict] = []

    @torch.no_grad()
    def _predict_batch(S_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xb = torch.from_numpy(S_batch).to(device=device, dtype=torch.float32)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        pred_idx = np.argmax(probs, axis=1)
        return pred_idx, probs

    batch_specs: List[np.ndarray] = []
    batch_meta: List[dict] = []

    def _flush_batch():
        if not batch_specs:
            return
        S = np.stack(batch_specs, axis=0)  # (B,C,F,T)
        pred_idx, probs = _predict_batch(S)

        for bi, meta in enumerate(batch_meta):
            gt_label = meta["gt_label"]
            pred_label = classes[int(pred_idx[bi])]
            proba = probs[bi]

            y_true.append(gt_label)
            y_pred.append(pred_label)

            row = meta["row"]
            fs = meta["fs"]
            pre_rms = meta["pre_rms"]
            utc_iso = (row.get("utc_iso", "") or "").strip()

            if DEBUG_PRINT_SAMPLES:
                print(f"[{utc_iso}] sample_id={meta['sample_id']} GT={gt_label} | Pred={pred_label}")

            # best-effort numeric metadata
            def _int_field(k, default=-1):
                try:
                    return int(row.get(k, default))
                except Exception:
                    return default

            def _float_field(k, default=0.0):
                try:
                    return float(row.get(k, default))
                except Exception:
                    return default

            log_row = {
                "sample_id": meta["sample_id"],
                "block_idx": _int_field("block_idx", -1),
                "utc_iso": utc_iso,
                "gps_week": _int_field("gps_week", -1),
                "tow_s": _float_field("tow_s", 0.0),
                "csv_label_raw": meta["label_raw"],
                "gt_label": gt_label,
                "pred_label": pred_label,
                "fs_hz": float(fs),
                "pre_rms": float(pre_rms),
                "iq_path": str(meta["iq_path"]),
            }

            # Probabilities for model classes
            for ci, cname in enumerate(classes):
                log_row[f"p_{cname}"] = float(proba[ci])

            rows_log.append(log_row)

        batch_specs.clear()
        batch_meta.clear()

    for it in items:
        try:
            iq_raw, fs = _read_npz_iq_fs(it["iq_path"])
        except Exception as e:
            n_skipped_bad_npz += 1
            if DEBUG_PRINT_SAMPLES:
                print(f"[SKIP] bad NPZ {it['iq_path']}: {e}")
            continue

        pre_rms = float(np.sqrt(np.mean(np.abs(iq_raw) ** 2)) + 1e-20)

        # Build spectrogram exactly like training
        iq = center_crop_or_pad(iq_raw, target_len=target_len)
        iq = normalize_iq(iq)
        S = make_spectrogram(
            z=iq,
            fs=fs,
            nfft=nfft,
            win=win,
            hop=hop,
            spec_mode=spec_mode,
            spec_norm=spec_norm,
            eps=eps,
            do_fftshift=fftshift
        )  # (C,F,T)

        batch_specs.append(S.astype(np.float32, copy=False))
        batch_meta.append({**it, "fs": fs, "pre_rms": pre_rms})

        if len(batch_specs) >= int(BATCH_SIZE):
            _flush_batch()

    _flush_batch()

    if not rows_log:
        print("No predictions produced (all NPZ failed?).")
        return

    # ===== Metrics =====
    print("\n=== METRICS ===")
    present = sorted(
        set(y_true) | set(y_pred),
        key=lambda c: model_label_namespace.index(c) if c in model_label_namespace else len(model_label_namespace),
    )
    labels_for_metrics = present

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)
    acc = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Labels order:", labels_for_metrics)
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels_for_metrics, zero_division=0))

    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    if SAVE_CONFUSION_PNG:
        plot_confusion(cm, labels_for_metrics, out_dir / "confusion_matrix.png", "Confusion Matrix", normalize=False)
    if SAVE_ROW_NORM_PNG:
        plot_confusion(cm, labels_for_metrics, out_dir / "confusion_matrix_rownorm.png", "Confusion Matrix (row-norm)", normalize=True)

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy: {acc:.6f}\n")
        fh.write(f"Labels used for metrics: {labels_for_metrics}\n\n")
        fh.write("Confusion matrix (rows=True, cols=Pred):\n")
        for r in cm:
            fh.write(",".join(map(str, r)) + "\n")
        fh.write("\nClassification report:\n")
        fh.write(classification_report(y_true, y_pred, labels=labels_for_metrics, zero_division=0))

    # Per-sample CSV
    if SAVE_PER_SAMPLE_CSV and rows_log:
        csv_out = out_dir / "samples_eval.csv"
        base_cols = [
            "sample_id", "block_idx", "utc_iso", "gps_week", "tow_s",
            "csv_label_raw", "gt_label", "pred_label", "fs_hz", "pre_rms", "iq_path",
        ] + [f"p_{c}" for c in classes]

        present_keys = set().union(*[set(r.keys()) for r in rows_log])
        fieldnames = [c for c in base_cols if c in present_keys] + [k for k in rows_log[0].keys() if k not in base_cols]

        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_log:
                w.writerow(r)
        print(f"Wrote per-sample log to: {csv_out}")

    if SUMMARY_JSON:
        js = {
            "labels_csv_path": str(labels_csv_path),
            "out_dir": str(out_dir),
            "model_pt": str(model_pt),
            "checkpoint_kind": ckpt_kind,
            "model_classes": classes,
            "extra_labels": EXTRA_LABELS,
            "device": str(device),
            "n_rows_csv": int(n_total),
            "n_kept": int(len(items)),
            "n_predicted": int(len(rows_log)),
            "n_skipped_bad_label": int(n_skipped_bad_label),
            "n_skipped_missing_npz": int(n_skipped_missing_npz),
            "n_skipped_bad_npz": int(n_skipped_bad_npz),
            "accuracy": float(acc),
            "labels_for_metrics": labels_for_metrics,
            "spec_used": {
                "TARGET_LEN": target_len,
                "NFFT": nfft,
                "WIN": win,
                "HOP": hop,
                "SPEC_MODE": spec_mode,
                "SPEC_NORM": spec_norm,
                "FFTSHIFT": fftshift,
                "EPS": eps,
                "MODEL": cfg.get("MODEL", None),
                "VIT_PATCH": cfg.get("VIT_PATCH", None),
                "VIT_EMBED_DIM": cfg.get("VIT_EMBED_DIM", None),
                "VIT_DEPTH": cfg.get("VIT_DEPTH", None),
                "VIT_HEADS": cfg.get("VIT_HEADS", None),
            },
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(js, fh, indent=2)
        print("Wrote summary.json")

    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
