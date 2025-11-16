# train_encoder_supcon.py
# Self-supervised complex-IQ encoder with NT-Xent (SimCLR-style) contrastive loss.
# - Default paths allow:   python train_encoder_supcon.py
# - Use GPU if available; AMP enabled on CUDA (new torch.amp API).
# - Exports 128-D embeddings for train/val/test to artifacts/enc_run_YYYYmmdd_HHMMSS.
# - Saves loss curve (CSV + PNG), encoder .pt, and a run config JSON.

import argparse, time, json
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import decimate as sp_decimate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------- Defaults -----------------------
DEF_BASE = r"D:\datasets\zenodo\3783969\Jamming_Classifier"
DEF_OUT  = "./artifacts"
DEF_CLASSES = "NoJam,SingleAM,SingleChirp,SingleFM,DME,NB"

# ----------------------- IO helpers -----------------------
def load_iq_and_meta(path, var="GNSS_plus_Jammer_awgn", decim=8):
    """Load a complex IQ vector (or Nx2 [I,Q]) from a .mat and decimate.
       Returns (x:(2,T), jsr:float or nan, cnr:float or nan).
    """
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if var not in m:
        raise KeyError(f"{var} not found in {Path(path).name}")

    z = np.asarray(m[var]).ravel()
    if not np.iscomplexobj(z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2 and z.shape[1] == 2:
            z = z[:, 0] + 1j * z[:, 1]
        else:
            raise TypeError(f"{Path(path).name}: variable '{var}' not complex and not Nx2 real.")

    I = sp_decimate(z.real, decim, ftype="fir", zero_phase=True).astype(np.float32)
    Q = sp_decimate(z.imag, decim, ftype="fir", zero_phase=True).astype(np.float32)

    # Per-sample RMS norm (rotation-invariant magnitude)
    rms = np.sqrt((I**2 + Q**2).mean()) + 1e-12
    I /= rms; Q /= rms

    # Optional meta
    meta = m.get("meta", {})
    jsr = float(meta.get("JSR_dB", np.nan)) if isinstance(meta, dict) else np.nan
    cnr = float(meta.get("CNR_dBHz", np.nan)) if isinstance(meta, dict) else np.nan

    return np.stack([I, Q], axis=0), jsr, cnr  # (2,T), jsr, cnr


def phase_rotate(x, phi):
    I, Q = x[0], x[1]
    c = (I + 1j * Q) * np.exp(1j * phi)
    return np.stack([c.real.astype(np.float32), c.imag.astype(np.float32)], axis=0)


def freq_shift(x, delta_hz, fs):
    I, Q = x[0], x[1]
    T = I.shape[-1]
    t = np.arange(T, dtype=np.float32) / fs
    c = (I + 1j * Q) * np.exp(1j * 2 * np.pi * delta_hz * t)
    return np.stack([c.real.astype(np.float32), c.imag.astype(np.float32)], axis=0)


def augment(x, fs_eff, noise_snr_db=30.0):
    """Light GNSS-IQ augmentations: time shift, random phase, tiny CFO jitter, amp jitter, AWGN."""
    # 1) small time shift
    max_shift = max(1, int(0.02 * x.shape[-1]))
    s = np.random.randint(-max_shift, max_shift + 1)
    x = np.roll(x, s, axis=-1)
    # 2) random phase rotation
    phi = np.random.uniform(-np.pi, np.pi)
    x = phase_rotate(x, phi)
    # 3) tiny frequency offset jitter (±0.1% of fs_eff)
    df = np.random.uniform(-0.001, 0.001) * fs_eff
    x = freq_shift(x, df, fs_eff)
    # 4) amplitude jitter
    x = x * np.random.uniform(0.8, 1.2)
    # 5) AWGN
    p = np.mean(x**2)
    snr = 10**(noise_snr_db / 10.0)
    nvar = p / max(snr, 1e-6)
    x = x + np.random.normal(0, np.sqrt(nvar), size=x.shape).astype(np.float32)
    return x


# ----------------------- Dataset -----------------------
class MatDataset(Dataset):
    def __init__(self, root, classes, split, var, fs, decim, seg_len, per_class_cap=None):
        self.root = Path(root)
        self.classes = classes
        self.split = split
        self.var = var
        self.fs = float(fs)
        self.decim = int(decim)
        self.seg_len = int(seg_len)
        self.fs_eff = self.fs / self.decim

        base = self.root / {
            "train": "Image_training_database",
            "val":   "Image_validation_database",
            "test":  "Image_testing_database"
        }[split]

        self.items = []  # list of (path:str, y:int)
        for y, cname in enumerate(classes):
            files = sorted((base / cname).glob("*.mat"))
            if per_class_cap:
                files = files[:per_class_cap]
            for f in files:
                self.items.append((str(f), y))

    def __len__(self):
        return len(self.items)

    def _crop_center(self, x):
        T = x.shape[-1]
        if T == self.seg_len:
            return x
        if T < self.seg_len:
            pad = self.seg_len - T
            left = pad // 2
            right = pad - left
            return np.pad(x, ((0, 0), (left, right)), mode="constant")
        # T > seg_len
        s = (T - self.seg_len) // 2
        return x[:, s:s + self.seg_len]

    def __getitem__(self, idx):
        path, y = self.items[idx]
        x, _, _ = load_iq_and_meta(path, var=self.var, decim=self.decim)
        x = self._crop_center(x)
        v1 = augment(x, self.fs_eff)
        v2 = augment(x, self.fs_eff)
        return torch.from_numpy(v1), torch.from_numpy(v2), y


# ----------------------- Model -----------------------
class Encoder1D(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(2, 32, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 9, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, emb)
        )

    def forward(self, x):
        # x: (B, 2, T)
        h = self.feat(x).squeeze(-1)      # (B, 128)
        z = F.normalize(self.proj(h), dim=1)  # (B, emb) L2-normalized
        return h, z


def nt_xent(z1, z2, T=0.1):
    """
    InfoNCE / NT-Xent loss:
    - Build a 2B x 2B similarity matrix with positive pairs on the off-diagonal.
    - Uses dtype-safe masking to avoid fp16 overflow (no -1e9 literal).
    """
    z = torch.cat([z1, z2], dim=0)                  # (2B, D)
    # Cosine sim via dot product (vectors are normalized). Compute in fp32 for stability.
    S = torch.matmul(z.float(), z.float().t()) / T  # (2B, 2B)
    B = z1.size(0)
    # Remove self-similarity with the minimum representable value for the dtype
    neg_inf = torch.finfo(S.dtype).min              # safe for fp16/fp32
    mask = torch.eye(2 * B, device=S.device, dtype=torch.bool)
    S = S.masked_fill(mask, neg_inf)

    # For each row i, the positive index is j:
    # First B rows -> positives are rows B..2B-1; Next B rows -> positives are 0..B-1
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(S.device)
    loss = F.cross_entropy(S, labels)
    return loss


# ----------------------- Embedding export -----------------------
@torch.no_grad()
def export_split_embeddings(model, dataset, device, bs=128):
    """
    Run the encoder on the original (center-cropped, non-augmented) signals
    to extract the 128-D 'h' representation (pre-projection).
    Also tries to read JSR/CNR from the MATs to store alongside.
    """
    class Plain(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            path, y = self.base.items[i]
            x, jsr, cnr = load_iq_and_meta(path, var=self.base.var, decim=self.base.decim)
            x = self.base._crop_center(x)
            return torch.from_numpy(x), y, jsr, cnr, path

    pds = Plain(dataset)
    loader = DataLoader(
        pds, batch_size=bs, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model.eval()
    feats, ys = [], []
    jsrs, cnrs, paths = [], [], []
    for x, y, jsr, cnr, path in loader:
        x = x.to(device, non_blocking=True)
        h, _ = model(x)
        feats.append(h.detach().cpu().numpy())
        ys.append(np.asarray(y))
        jsrs.append(np.asarray(jsr, dtype=np.float64))
        cnrs.append(np.asarray(cnr, dtype=np.float64))
        paths.extend(list(path))

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0)
    jsr = np.concatenate(jsrs, axis=0)
    cnr = np.concatenate(cnrs, axis=0)
    paths = np.asarray(paths, dtype=object)
    return X, y, jsr, cnr, paths


# ----------------------- Plot helpers -----------------------
def save_loss_plots(run_dir: Path, losses: list):
    # Save CSV
    rows = [{"epoch": i + 1, "loss": float(v)} for i, v in enumerate(losses)]
    csv_path = run_dir / "loss_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("epoch,loss\n")
        for r in rows:
            f.write(f"{r['epoch']},{r['loss']:.6f}\n")

    # Save PNG
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("NT-Xent loss"); plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close()


# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Self-supervised IQ encoder (NT-Xent).")
    p.add_argument("--base", type=str, default=DEF_BASE,
                   help="Root folder with Image_training_database / Image_validation_database / Image_testing_database")
    p.add_argument("--out", type=str, default=DEF_OUT, help="Artifacts root.")
    p.add_argument("--classes", type=str, default=DEF_CLASSES,
                   help="Comma-separated class list.")
    p.add_argument("--var", type=str, default="GNSS_plus_Jammer_awgn")
    p.add_argument("--fs", type=float, default=40_920_000.0)
    p.add_argument("--decim", type=int, default=8)
    p.add_argument("--seg_len", type=int, default=4096)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cap", type=int, default=None, help="Optional cap per class per split.")

    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA (recommended).")
    p.add_argument("--seed", type=int, default=123, help="Reproducibility seed.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    out_root = Path(args.out)
    run_dir = out_root / f"enc_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Device + AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and (device.type == "cuda")
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            # fallback for slightly older PyTorch versions
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = None

    # Datasets
    tr = MatDataset(args.base, classes, "train", args.var, args.fs, args.decim, args.seg_len, args.cap)
    va = MatDataset(args.base, classes, "val",   args.var, args.fs, args.decim, args.seg_len, args.cap)
    te = MatDataset(args.base, classes, "test",  args.var, args.fs, args.decim, args.seg_len, args.cap)

    # Model
    model = Encoder1D(emb=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader_tr = DataLoader(
        tr, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Train
    losses = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for v1, v2, _ in loader_tr:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            if scaler is not None:
                # New AMP API (PyTorch ≥ 2.0); silently falls back to old if needed
                try:
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        _, z1 = model(v1)
                        _, z2 = model(v2)
                        loss = nt_xent(z1, z2, T=0.1)
                except TypeError:
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        _, z1 = model(v1)
                        _, z2 = model(v2)
                        loss = nt_xent(z1, z2, T=0.1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = nt_xent(z1, z2, T=0.1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            total += float(loss.item())

        epoch_loss = total / max(1, len(loader_tr))
        losses.append(epoch_loss)
        print(f"[ep {ep}] loss={epoch_loss:.4f}")

    # Save encoder weights
    torch.save(model.state_dict(), run_dir / "encoder.pt")

    # Loss curves
    save_loss_plots(run_dir, losses)

    # Export embeddings (pre-projection 'h', 128-D by default) + metadata if available
    Xtr, ytr, jsr_tr, cnr_tr, p_tr = export_split_embeddings(model, tr, device)
    Xva, yva, jsr_va, cnr_va, p_va = export_split_embeddings(model, va, device)
    Xte, yte, jsr_te, cnr_te, p_te = export_split_embeddings(model, te, device)
    feat_names = np.array([f"enc_f{i}" for i in range(Xtr.shape[1])], dtype=object)
    class_names = np.array(classes, dtype=object)

    def dump_split(name, X, y, jsr, cnr, paths):
        np.savez_compressed(
            run_dir / f"{name}_features.npz",
            X=X.astype(np.float32, copy=False),
            y=y.astype(np.int64, copy=False),
            class_names=class_names,
            feature_names=feat_names,
            jsr=jsr, cnr=cnr,
            paths=np.asarray(paths, dtype=object),
        )
        print(f"[OK] wrote {name}_features.npz  shape={X.shape}")

    dump_split("train", Xtr, ytr, jsr_tr, cnr_tr, p_tr)
    dump_split("val",   Xva, yva, jsr_va, cnr_va, p_va)
    dump_split("test",  Xte, yte, jsr_te, cnr_te, p_te)

    print(f"[OK] run dir -> {run_dir}")
    print("Notes:")
    print("- NT-Xent loss should **decrease** over epochs; lower is better alignment of two views.")
    print("- Embeddings are in train/val/test_features.npz as X (N x D=128) with y and class_names.")
    print("- If you later linear-probe or XGB/SVM these embeddings, keep batch size modest for your GPU (RTX 3050 4GB).")


if __name__ == "__main__":
    main()
