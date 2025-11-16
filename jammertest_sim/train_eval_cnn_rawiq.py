# train_eval_cnn_rawiq.py
"""
Train & evaluate a 1-D CNN on raw IQ tiles (.mat) for Jammertest-style datasets.

Folder structure expected:
  BASE/
    TRAIN/<Class>/*.mat
    VAL/<Class>/*.mat
    TEST/<Class>/*.mat

Each .mat must contain:
  - GNSS_plus_Jammer_awgn : complex IQ vector (2048 samples typical)
  - meta (optional) with fields: JSR_dB, CNR_dBHz or CNo_dBHz, band, fs_Hz, etc.

Outputs (under --out):
  - run folder with model.pt, curves.png, confusion matrices (val/test), classification reports,
    per-bin (JSR/CNR) metrics, and a summary.txt.

Usage example:
  python train_eval_cnn_rawiq.py --base "D:/datasets/maca_gen/datasets_jammertest" --epochs 40 --batch_size 256
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import argparse, time, json, csv, math, os

import numpy as np
import scipy.io as sio
import h5py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")  # headless
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

# ---------------- Defaults ----------------
DEFAULT_BASE = r"D:\datasets\maca_gen\datasets_jammertest"
DEFAULT_CLASSES = "NoJam,Chirp,NB,CW,WB"
DEFAULT_VAR = "GNSS_plus_Jammer_awgn"
DEFAULT_FS = 60_000_000.0
DEFAULT_TARGET_LEN = 2048

DEFAULT_ARTIFACTS_ROOT = Path("../artifacts/jammertest_sim")
DEFAULT_JSR_BINS = [0, 10, 25, 40]  # dB
DEFAULT_CNR_BINS = [20, 30, 40, 60] # dB-Hz

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="1-D CNN on raw IQ .mat tiles (Jammertest).")
    p.add_argument("--base", type=str, default=DEFAULT_BASE, help="Dataset root with TRAIN/VAL/TEST.")
    p.add_argument("--classes", type=str, default=DEFAULT_CLASSES, help="Comma-separated class names (subfolders).")
    p.add_argument("--var", type=str, default=DEFAULT_VAR, help="MAT variable name for IQ vector.")
    p.add_argument("--fs", type=float, default=DEFAULT_FS, help="Sampling rate (Hz).")
    p.add_argument("--target_len", type=int, default=DEFAULT_TARGET_LEN, help="Crop/pad IQ to this length.")
    p.add_argument("--out", type=str, default=str(DEFAULT_ARTIFACTS_ROOT), help="Artifacts root.")
    p.add_argument("--run_name", type=str, default=None, help="Optional name suffix for outputs.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=80, help="Early stopping patience (epochs).")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap_per_class", type=int, default=None, help="Optional cap per class (for quick runs).")
    p.add_argument("--augment", action="store_true", help="Enable mild data augmentation (phase/amp jitter).")
    p.add_argument("--cfo_jitter", action="store_true", help="Enable small CFO jitter augmentation.")
    p.add_argument("--device", type=str, default=None, help="'cuda'/'cpu' (auto if None).")
    return p.parse_args()

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
    """Best-effort read of 'meta' as dict (works for v7 and v7.3)."""
    # v7
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        meta = m.get("meta", None)
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass
    # v7.3
    out = {}
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                return out
            g = f["meta"]
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    arr = np.asarray(v[()])
                    if np.size(arr) == 1:
                        out[k] = float(arr) if np.isrealobj(arr) else arr
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
            # Prefer CNR_dBHz / CNo_dBHz
            c = np.nan
            for k in ("CNR_dBHz", "CNo_dBHz", "CNR_dB", "CNo"):
                if k in meta:
                    try:
                        c = float(np.asarray(meta[k]).ravel()[0])
                        break
                    except Exception:
                        pass
            j = np.nan
            if "JSR_dB" in meta:
                try:
                    j = float(np.asarray(meta["JSR_dB"]).ravel()[0])
                except Exception:
                    pass
            jsr.append(j); cnr.append(c)
    return files, np.array(labels, int), np.array(jsr, float), np.array(cnr, float)

# ---------------- PyTorch Dataset ----------------
class JammerMatDataset(Dataset):
    def __init__(self, files: List[Path], labels: np.ndarray, var_name: str,
                 target_len: int, fs: float, train: bool=False,
                 augment: bool=False, cfo_jitter: bool=False, seed: int=42):
        self.files = list(files)
        self.labels = np.array(labels, int)
        self.var_name = var_name
        self.target_len = int(target_len)
        self.fs = float(fs)
        self.train = train
        self.augment = augment
        self.cfo_jitter = cfo_jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def _center_crop_or_pad(self, z: np.ndarray) -> np.ndarray:
        N = z.size
        T = self.target_len
        if N == T:
            return z
        if N > T:
            k0 = (N - T) // 2
            return z[k0:k0+T]
        # pad
        out = np.zeros(T, dtype=z.dtype)
        out[:N] = z
        return out

    def _augment(self, z: np.ndarray) -> np.ndarray:
        # Mild amplitude jitter and global phase rotation
        if self.augment:
            amp = float(self.rng.uniform(0.9, 1.1))
            phi = float(self.rng.uniform(-np.pi, np.pi))
            z = amp * z * np.exp(1j * phi)
        # Optional tiny CFO jitter (safe default off)
        if self.cfo_jitter:
            f_off = float(self.rng.uniform(-2e5, 2e5))  # ±200 kHz
            n = z.size
            t = np.arange(n, dtype=np.float32) / self.fs
            z = z * np.exp(1j * (2*np.pi * f_off * t))
        return z

    def __getitem__(self, idx):
        p = self.files[idx]
        iq_raw = to_complex_1d(load_mat_var(p, self.var_name))
        z = self._center_crop_or_pad(iq_raw)
        if self.train:
            z = self._augment(z)

        # Per-sample normalization: remove mean, scale to unit RMS
        z = z - np.mean(z)
        rms = np.sqrt(np.mean(np.abs(z)**2) + 1e-12)
        z = (z / rms).astype(np.complex64, copy=False)

        # To 2xN float32 tensor
        x = np.stack([z.real.astype(np.float32), z.imag.astype(np.float32)], axis=0)  # (2, N)
        y = int(self.labels[idx])
        return x, y

# ---------------- Model ----------------
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
        # x: (B, 2, N)
        x = self.block1(x)
        x = F.max_pool1d(x, 2)
        x = self.block2(x)
        x = F.max_pool1d(x, 2)
        x = self.block3(x)
        x = F.max_pool1d(x, 2)
        x = self.block4(x)
        x = self.head(x)
        return x

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

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
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
    return total_loss / max(1, total_n), total_correct / max(1, total_n)

@torch.no_grad()
def evaluate(model, loader, device, return_probs: bool=False):
    model.eval()
    all_y, all_p, all_logits = [], [], []
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
        if return_probs:
            all_p.append(probs.cpu().numpy())
        else:
            all_logits.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y)
    if return_probs:
        P = np.concatenate(all_p, axis=0)
        y_pred = P.argmax(axis=1)
        return total_loss / max(1, total_n), total_correct / max(1, total_n), y_true, y_pred, P
    else:
        y_pred = np.concatenate(all_logits)
        return total_loss / max(1, total_n), total_correct / max(1, total_n), y_true, y_pred, None

# ---------------- Plotting & metrics ----------------
def plot_curves(history: Dict[str, List[float]], out_png: Path):
    plt.figure(figsize=(8,4))
    epochs = np.arange(1, len(history["train_loss"])+1)
    ax1 = plt.subplot(1,2,1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, ls="--", alpha=0.4)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

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
            txt = f"{(100*M[i,j]):.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", color="white")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_csv_dict(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)

def eval_by_bins(y_true, y_pred, classes, values: np.ndarray, edges, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    rows = []
    acc_all = accuracy_score(y_true, y_pred)
    f1m_all = f1_score(y_true, y_pred, average="macro")
    rows.append({"bin": "OVERALL", "count": int(y_true.size), "acc": acc_all, "macro_f1": f1m_all})

    edges = np.asarray(edges, float)
    bin_labels = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges)-1)]
    for b in range(len(edges)-1):
        mask = (values >= edges[b]) & (values < edges[b+1])
        if not np.any(mask): continue
        yt = y_true[mask]; yp = y_pred[mask]
        acc = accuracy_score(yt, yp); f1m = f1_score(yt, yp, average="macro")
        cm = confusion_matrix(yt, yp, labels=list(range(K)))
        np.savetxt(out_dir / f"cm_{tag}_bin{b}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, classes, False, f"{tag} CM {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}.png")
        plot_confusion_matrix(cm, classes, True,  f"{tag} CM (row-norm) {bin_labels[b]}", out_dir / f"cm_{tag}_bin{b}_rownorm.png")
        rows.append({"bin": bin_labels[b], "count": int(mask.sum()), "acc": acc, "macro_f1": f1m})
    save_csv_dict(out_dir / f"metrics_{tag}.csv", rows, ["bin", "count", "acc", "macro_f1"])

# ---------------- Main ----------------
def main():
    if not HAVE_TORCH:
        raise RuntimeError(f"PyTorch not available: {TORCH_ERR}\nInstall with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    a = parse_args()
    base = Path(a.base)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    out_root = Path(a.out)
    run_root = out_root / (a.run_name or ("cnn_run_" + time.strftime("%Y%m%d_%H%M%S")))
    run_root.mkdir(parents=True, exist_ok=True)

    # Device & seed
    device = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(a.seed)

    # Scan dataset
    tr_files, tr_labels, tr_jsr, tr_cnr = list_split_files(base, "TRAIN", classes, a.cap_per_class)
    va_files, va_labels, va_jsr, va_cnr = list_split_files(base, "VAL", classes, a.cap_per_class)
    te_files, te_labels, te_jsr, te_cnr = list_split_files(base, "TEST", classes, a.cap_per_class)

    print(f"[DATA] Train={len(tr_files)}  Val={len(va_files)}  Test={len(te_files)}")
    print(f"[DATA] Classes={classes}")

    # Datasets & loaders
    ds_tr = JammerMatDataset(tr_files, tr_labels, a.var, a.target_len, a.fs, train=True,
                             augment=a.augment, cfo_jitter=a.cfo_jitter, seed=a.seed)
    ds_va = JammerMatDataset(va_files, va_labels, a.var, a.target_len, a.fs, train=False)
    ds_te = JammerMatDataset(te_files, te_labels, a.var, a.target_len, a.fs, train=False)

    # Class weights for CE loss
    num_classes = len(classes)
    cweights = compute_class_weights(tr_labels, num_classes).to(device)

    # Weighted sampler (optional; helpful if classes imbalanced)
    class_counts = np.bincount(tr_labels, minlength=num_classes)
    samples_weight = 1.0 / np.take(class_counts, tr_labels)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Collate: convert to torch
    def _collate(batch):
        xs = torch.tensor(np.stack([b[0] for b in batch], axis=0))  # (B, 2, N)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, sampler=sampler,
                       num_workers=a.num_workers, pin_memory=True, collate_fn=_collate, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True, collate_fn=_collate, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=a.batch_size, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True, collate_fn=_collate, drop_last=False)

    # Model / optim / sched
    model = RawIQ_CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cweights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, a.patience//2))

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf"); best_state = None; es_pat = 0

    for epoch in range(1, a.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, optimizer, device, criterion)
        va_loss, va_acc, _, _, _ = evaluate(model, dl_va, device, return_probs=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        scheduler.step(va_loss)

        print(f"[E{epoch:02d}/{a.epochs}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            es_pat = 0
        else:
            es_pat += 1
            if es_pat >= a.patience:
                print(f"[early-stop] No val improvement for {a.patience} epochs.")
                break

    # Save curves
    plot_curves(history, run_root / "curves.png")

    # Restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "config": vars(a),
        "history": history
    }, run_root / "model.pt")

    # VAL evaluation
    va_loss, va_acc, yv_true, yv_pred, _ = evaluate(model, dl_va, device, return_probs=False)
    cm_val = confusion_matrix(yv_true, yv_pred, labels=list(range(num_classes)))
    np.savetxt(run_root / "val_cm.csv", cm_val, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_val, classes, False, "CNN VAL CM", run_root / "val_cm.png")
    plot_confusion_matrix(cm_val, classes, True,  "CNN VAL CM (row-norm)", run_root / "val_cm_rownorm.png")
    rep_val = classification_report(yv_true, yv_pred, target_names=classes, digits=6, output_dict=True)
    rows_val = [{"name": k, **v} for k, v in rep_val.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "val_report.csv", rows_val, ["name","precision","recall","f1-score","support"])

    # TEST evaluation (with probs)
    te_loss, te_acc, yt_true, yt_pred, Pte = evaluate(model, dl_te, device, return_probs=True)
    np.savez_compressed(run_root / "test_preds.npz", y_true=yt_true, y_pred=yt_pred, probs=Pte, classes=np.array(classes, dtype=object))

    cm_test = confusion_matrix(yt_true, yt_pred, labels=list(range(num_classes)))
    np.savetxt(run_root / "test_cm.csv", cm_test, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm_test, classes, False, "CNN TEST CM", run_root / "test_cm.png")
    plot_confusion_matrix(cm_test, classes, True,  "CNN TEST CM (row-norm)", run_root / "test_cm_rownorm.png")
    rep_test = classification_report(yt_true, yt_pred, target_names=classes, digits=6, output_dict=True)
    rows_test = [{"name": k, **v} for k, v in rep_test.items() if isinstance(v, dict)]
    save_csv_dict(run_root / "test_report.csv", rows_test, ["name","precision","recall","f1-score","support"])

    # Per-JSR/CNR metrics (VAL/TEST)
    # Reload jsr/cnr quickly (already scanned at start)
    if va_jsr.size == len(yv_true):
        eval_by_bins(yv_true, yv_pred, classes, va_jsr, DEFAULT_JSR_BINS, run_root / "val_by_jsr", "val_JSR")
    if va_cnr.size == len(yv_true):
        eval_by_bins(yv_true, yv_pred, classes, va_cnr, DEFAULT_CNR_BINS, run_root / "val_by_cnr", "val_CNR")
    if te_jsr.size == len(yt_true):
        eval_by_bins(yt_true, yt_pred, classes, te_jsr, DEFAULT_JSR_BINS, run_root / "test_by_jsr", "test_JSR")
    if te_cnr.size == len(yt_true):
        eval_by_bins(yt_true, yt_pred, classes, te_cnr, DEFAULT_CNR_BINS, run_root / "test_by_cnr", "test_CNR")

    # Summary
    with open(run_root / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"val_loss={va_loss:.6f}  val_acc={va_acc:.6f}  val_macroF1={f1_score(yv_true, yv_pred, average='macro'):.6f}\n")
        f.write(f"test_loss={te_loss:.6f}  test_acc={te_acc:.6f}  test_macroF1={f1_score(yt_true, yt_pred, average='macro'):.6f}\n")

    # Save run metadata
    meta = {
        "classes": classes,
        "fs_hz": a.fs,
        "target_len": a.target_len,
        "num_train": len(tr_files),
        "num_val": len(va_files),
        "num_test": len(te_files),
        "device": str(device),
        "config": vars(a)
    }
    with open(run_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[cnn] done -> {run_root}")

if __name__ == "__main__":
    main()
