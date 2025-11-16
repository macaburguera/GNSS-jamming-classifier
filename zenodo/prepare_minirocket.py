from pathlib import Path
import argparse, time, json, csv, warnings
import numpy as np
import scipy.io as sio
from scipy.signal import decimate as sp_decimate

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

# -------- MiniROCKET --------
try:
    from sktime.transformations.panel.rocket import MiniRocket
    HAVE_SKTIME = True
except Exception:
    HAVE_SKTIME = False

# -------- Scoring (sklearn) --------
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="MiniROCKET with top-K feature selection (fast, memory-friendly)."
    )
    # data
    p.add_argument("--base", type=str,
                   default=r"D:\datasets\zenodo\3783969\Jamming_Classifier",
                   help="Root with Image_training_database / Image_validation_database / Image_testing_database")
    p.add_argument("--var", type=str, default="GNSS_plus_Jammer_awgn",
                   help="MAT variable with the complex IQ vector")
    p.add_argument("--fs", type=float, default=40_920_000.0, help="Sampling rate (Hz)")
    p.add_argument("--out", type=str, default="./artifacts", help="Artifacts root")
    p.add_argument("--classes", type=str,
                   default="NoJam,SingleAM,SingleChirp,SingleFM,DME,NB",
                   help="Comma-separated class order (subfolder names)")
    p.add_argument("--dry_per_class", type=int, default=600, help="Cap files per class (None=all)")

    # MiniROCKET
    p.add_argument("--mr_kernels", type=int, default=8000,
                   help="#kernels per channel (I and Q -> 2x this). Start big; we’ll select down.")
    p.add_argument("--decim", type=int, default=4, help="Decimation factor on I & Q (anti-aliased)")
    p.add_argument("--include_mag", action="store_true",
                   help="Also include |z| as a 3rd channel (more features)")

    # Selection
    p.add_argument("--target_k", type=int, default=1024, help="Final #features to keep (total)")
    p.add_argument("--subsample", type=int, default=3000, help="Rows for scoring/corr pruning")
    p.add_argument("--corr_thr", type=float, default=0.98, help="Correlation pruning threshold")
    p.add_argument("--score_methods", type=str, default="anova,trees",
                   help="Comma list among: anova,trees (defaults are fast & robust)")
    p.add_argument("--save_full", action="store_true",
                   help="Also save FULL MR features (for debugging / ablations)")

    return p.parse_args()

# ---------------- FS helpers ----------------
def now_run_dir(root: Path, prefix="mr_topk_run") -> Path:
    t = time.strftime("%Y%m%d_%H%M%S")
    out = root / f"{prefix}_{t}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def split_dirs(base: Path):
    return {
        "TRAIN": base / "Image_training_database",
        "VAL":   base / "Image_validation_database",
        "TEST":  base / "Image_testing_database",
    }

def iter_files(base_dir: Path, classes, cap=None):
    for label, cls in enumerate(classes):
        files = sorted((base_dir / cls).glob("*.mat"))
        if cap is not None: files = files[:cap]
        yield cls, label, files

# ---------------- Load IQ & meta ----------------
def load_iq_meta(path: Path, var_name: str):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    if var_name not in m:
        raise KeyError(f"{var_name} not found in {path.name}")
    z = np.asarray(m[var_name]).ravel()
    if not np.iscomplexobj(z):
        # accept Nx2 [I,Q] fallback
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2 and z.shape[1] == 2:
            z = z[:,0] + 1j*z[:,1]
        else:
            raise TypeError(f"{path.name}: IQ is not complex and not Nx2 real")
    z = z.astype(np.complex64, copy=False)
    meta = m.get("meta", {})
    jsr = float(meta.get("JSR_dB", np.nan)) if isinstance(meta, dict) else np.nan
    cnr = float(meta.get("CNR_dBHz", np.nan)) if isinstance(meta, dict) else np.nan
    return z, jsr, cnr

def decimate_iq(z: np.ndarray, q: int):
    if q <= 1:  # no-op
        return z.real.astype(np.float32), z.imag.astype(np.float32), np.abs(z).astype(np.float32)
    I = sp_decimate(z.real, q, ftype="fir", zero_phase=True)
    Q = sp_decimate(z.imag, q, ftype="fir", zero_phase=True)
    M = sp_decimate(np.abs(z), q, ftype="fir", zero_phase=True)
    return I.astype(np.float32), Q.astype(np.float32), M.astype(np.float32)

# ---------------- sktime panels ----------------
def to_nested_univariate_panel(X_2d):
    # X_2d: (n_samples, T) -> nested DataFrame with one column of pd.Series
    import pandas as pd
    return pd.DataFrame({0: [pd.Series(x) for x in X_2d]})

def build_channel_features(X_2d, n_kernels):
    if not HAVE_SKTIME:
        raise RuntimeError("sktime not installed. Try: pip install sktime numba")
    Ptr = to_nested_univariate_panel(X_2d["TRAIN"])
    Pva = to_nested_univariate_panel(X_2d["VAL"])
    Pte = to_nested_univariate_panel(X_2d["TEST"])
    mr = MiniRocket(num_kernels=n_kernels, random_state=42)
    mr.fit(Ptr)
    Ftr = mr.transform(Ptr).to_numpy(dtype=np.float32, copy=False)
    Fva = mr.transform(Pva).to_numpy(dtype=np.float32, copy=False)
    Fte = mr.transform(Pte).to_numpy(dtype=np.float32, copy=False)
    return {"TRAIN": Ftr, "VAL": Fva, "TEST": Fte}, mr

# ---------------- Scoring & selection ----------------
def safe_subsample(X, y, nmax, seed=RANDOM_STATE):
    if X.shape[0] <= nmax: return X, y
    rng = np.random.default_rng(seed)
    sel = rng.choice(X.shape[0], size=nmax, replace=False)
    return X[sel], y[sel]

def rank_desc(x: np.ndarray) -> np.ndarray:
    # Higher score -> better (rank 1)
    order = (-x).argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    return ranks

def score_anova(X, y):
    F, _ = f_classif(X, y)
    F[np.isnan(F)] = 0.0
    return F

def score_trees(X, y):
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=8, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    et.fit(X, y)
    return et.feature_importances_

def corr_prune(X, cand_idx, k, thr=0.98, subsample=3000, seed=RANDOM_STATE):
    Xs, _ = safe_subsample(X, np.zeros(X.shape[0]), nmax=subsample, seed=seed)
    mu = Xs.mean(axis=0); sd = Xs.std(axis=0); sd[sd==0] = 1.0
    Z = (Xs - mu) / sd
    kept = []
    Zk = None
    for j in cand_idx:
        z = Z[:, j:j+1]
        if Zk is None:
            kept.append(j); Zk = z
        else:
            c = (z.T @ Zk) / Z.shape[0]
            if np.max(np.abs(c)) < thr:
                kept.append(j)
                Zk = np.hstack([Zk, z])
        if len(kept) >= k: break
    return np.array(kept, dtype=int)

# ---------------- Main ----------------
def main():
    args = parse_args()
    if not HAVE_SKTIME:
        raise RuntimeError("sktime is required. Install with: pip install sktime numba")

    base = Path(args.base)
    out_root = Path(args.out)
    out_dir = now_run_dir(out_root, prefix="mr_topk_run")
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    splits = {
        "TRAIN": base / "Image_training_database",
        "VAL":   base / "Image_validation_database",
        "TEST":  base / "Image_testing_database",
    }
    print(f"[INFO] Base={base}")
    print(f"[INFO] Writing MR(topK) to: {out_dir}")

    # 1) Load & decimate per split
    arrays = {}
    for split, d in splits.items():
        I_list, Q_list, M_list = [], [], []
        y, jsr, cnr, paths = [], [], [], []
        for cls, label, files in iter_files(d, classes, cap=args.dry_per_class):
            print(f"[{split}] {cls}: {len(files)} files")
            for p in files:
                try:
                    z, j, c = load_iq_meta(p, args.var)
                    I, Q, M = decimate_iq(z, args.decim)
                    # per-sample RMS norm
                    rms = np.sqrt((I**2 + Q**2).mean()) + 1e-12
                    I = (I / rms).astype(np.float32); Q = (Q / rms).astype(np.float32)
                    if args.include_mag:
                        M = (M / (M.mean() + 1e-12)).astype(np.float32)

                    I_list.append(I); Q_list.append(Q)
                    if args.include_mag: M_list.append(M)
                    y.append(label); jsr.append(j); cnr.append(c); paths.append(str(p))
                except Exception as e:
                    print(f"  [WARN] {p.name} -> {e}")

        if not I_list:
            raise RuntimeError(f"No data for split {split} in {d}")

        X_I = np.vstack([x[np.newaxis, ...] for x in I_list])
        X_Q = np.vstack([x[np.newaxis, ...] for x in Q_list])
        arrays[split] = {
            "I": X_I, "Q": X_Q,
            "M": np.vstack([m[np.newaxis, ...] for m in M_list]) if args.include_mag and len(M_list)>0 else None,
            "y": np.asarray(y, int),
            "jsr": np.asarray(jsr, float),
            "cnr": np.asarray(cnr, float),
            "paths": np.asarray(paths, dtype=object),
        }

    # 2) MiniROCKET per channel
    channels = {}
    for ch in ["I", "Q"] + (["M"] if args.include_mag else []):
        X_2d = {split: arrays[split][ch] for split in arrays if arrays[split][ch] is not None}
        feats, _mr = build_channel_features(X_2d, n_kernels=args.mr_kernels)
        channels[ch] = feats

    # 3) Concatenate channel features
    def concat_for(split):
        blocks = [channels["I"][split], channels["Q"][split]]
        if args.include_mag: blocks.append(channels["M"][split])
        X = np.hstack(blocks).astype(np.float32)
        return X

    Xtr = concat_for("TRAIN"); ytr = arrays["TRAIN"]["y"]
    Xva = concat_for("VAL");   yva = arrays["VAL"]["y"]
    Xte = concat_for("TEST");  yte = arrays["TEST"]["y"]
    n_all = Xtr.shape[1]
    print(f"[INFO] MR dims per split: TRAIN {Xtr.shape}, VAL {Xva.shape}, TEST {Xte.shape}")

    # Optionally save FULL features (for ablations)
    if args.save_full:
        for split, X in [("train", Xtr), ("val", Xva), ("test", Xte)]:
            np.savez_compressed(
                out_dir / f"{split}_features_full.npz",
                X=X, y=arrays[split.upper()]["y"],
                class_names=np.array(classes, dtype=object),
                feature_names=np.array([f"mr_f{i}" for i in range(X.shape[1])], dtype=object),
                jsr=arrays[split.upper()]["jsr"], cnr=arrays[split.upper()]["cnr"],
                fs_hz=np.array([args.fs/args.decim], dtype=np.float64),
                paths=arrays[split.upper()]["paths"],
            )

    # 4) Score features (on TRAIN, subsampled)
    Xs, ys = (Xtr, ytr)
    if args.subsample and Xtr.shape[0] > args.subsample:
        rng = np.random.default_rng(RANDOM_STATE)
        sel = rng.choice(Xtr.shape[0], size=args.subsample, replace=False)
        Xs = Xtr[sel]; ys = ytr[sel]

    methods = [m.strip().lower() for m in args.score_methods.split(",") if m.strip()]
    scores = {}
    if "anova" in methods:
        scores["anova"] = score_anova(Xs, ys); print("[score] ANOVA done")
    if "trees" in methods:
        scores["trees"] = score_trees(Xs, ys); print("[score] ExtraTrees done")

    if not scores:
        raise RuntimeError("No scoring methods produced scores. Use --score_methods anova,trees")

    # 5) Ensemble rank: average of normalized inverse ranks
    m = n_all
    ens = np.zeros(m, float)
    for name, sc in scores.items():
        r = rank_desc(sc)         # 1..m (1 is best)
        inv = 1.0 / r
        ens += inv / inv.max()

    cand = np.argsort(-ens)  # best first

    # 6) Correlation pruning to target_k
    target_k = min(args.target_k, m)
    sel_idx = corr_prune(Xtr, cand, k=target_k, thr=args.corr_thr, subsample=args.subsample)
    sel_idx = np.asarray(sel_idx, int)
    print(f"[INFO] Selected {len(sel_idx)}/{m} MR features")

    # 7) Save REDUCED datasets
    def save_split(split_name: str, X, y):
        np.savez_compressed(
            out_dir / f"{split_name}_features.npz",
            X=X[:, sel_idx].astype(np.float32),
            y=y,
            class_names=np.array(classes, dtype=object),
            feature_names=np.array([f"mr_f{i}" for i in sel_idx], dtype=object),
            jsr=arrays[split_name.upper()]["jsr"],
            cnr=arrays[split_name.upper()]["cnr"],
            fs_hz=np.array([args.fs/args.decim], dtype=np.float64),
            paths=arrays[split_name.upper()]["paths"],
        )

    save_split("train", Xtr, ytr)
    save_split("val",   Xva, yva)
    save_split("test",  Xte, yte)

    # 8) Metadata
    meta = {
        "fs_hz": args.fs,
        "decim": args.decim,
        "mr_kernels_per_channel": args.mr_kernels,
        "channels": ["I","Q"] + (["M"] if args.include_mag else []),
        "classes": classes,
        "var_name": args.var,
        "target_k": int(target_k),
        "corr_thr": float(args.corr_thr),
        "score_methods": methods,
        "subsample": int(args.subsample),
        "selected_indices_len": int(len(sel_idx)),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.save(out_dir / "selected_indices.npy", sel_idx)

    # 9) Dump ranking CSV (optional, for inspection)
    rows = []
    for i in range(m):
        row = {"idx": i, "ensemble": float(ens[i]), "selected": int(i in set(sel_idx))}
        for k,v in scores.items():
            row[f"{k}_score"] = float(v[i])
        rows.append(row)
    rows.sort(key=lambda r: r["ensemble"], reverse=True)
    with open(out_dir / "feature_ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"[OK] Wrote top-K MR features to: {out_dir}")
    print(f"     train/val/test_features.npz with {len(sel_idx)} columns")
    if args.save_full:
        print("     (full MR features also saved as *_features_full.npz)")

if __name__ == "__main__":
    main()
