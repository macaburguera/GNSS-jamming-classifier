# select_and_fuse_features.py
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import argparse, time, csv, sys, warnings
import numpy as np

from sklearn.feature_selection import mutual_info_classif, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import dump

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

# ---------- Defaults ----------
ARTIFACTS = Path("./artifacts")
DEFAULT_K = 128                 # final target #features
DEFAULT_MIN_PREP = 32           # ensure at least this many handcrafted features
DEFAULT_SUBSAMPLE = 3000        # subsample rows for scoring & corr pruning
DEFAULT_CORR_THR = 0.98         # correlation pruning threshold
RANDOM_STATE = 42

# ---------- Helpers to find latest runs ----------
def latest_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists(): return None
    cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands: return None
    return sorted(cands, key=lambda p: p.name, reverse=True)[0]

def ensure_dir(arg: Optional[str], fallback: Optional[Path], tag: str) -> Path:
    if arg:
        d = Path(arg); 
        if not d.exists(): raise FileNotFoundError(f"{tag} not found: {d}")
        return d
    if fallback is None: raise FileNotFoundError(f"No {tag} given and no {tag} found automatically.")
    return fallback

# ---------- IO ----------
def load_split(d: Path, split: str) -> Dict[str, np.ndarray]:
    z = np.load(d / f"{split}_features.npz", allow_pickle=True)
    out = {
        "X": z["X"],
        "y": z["y"],
        "class_names": z["class_names"].tolist(),
        "feature_names": z["feature_names"].tolist(),
        "paths": z["paths"].tolist() if "paths" in z.files else None,
        "jsr": z["jsr"] if "jsr" in z.files else None,
        "cnr": z["cnr"] if "cnr" in z.files else None,
    }
    return out

def save_csv(path: Path, rows: List[Dict], header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

# ---------- Alignment & fusion ----------
def align_concat(mr: Dict, prep: Dict) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Align by paths, return fused X, y, class_names, feature_names."""
    # If paths missing anywhere, assume same order & length
    if mr["paths"] is None or prep["paths"] is None:
        assert mr["X"].shape[0] == prep["X"].shape[0], "Cannot align: paths missing and lengths differ."
        y = mr["y"]; assert np.all(mr["y"] == prep["y"]), "Labels mismatch without paths."
        Xf = np.hstack([mr["X"], prep["X"]])
        fn = [f"mr:{n}" for n in mr["feature_names"]] + [f"prep:{n}" for n in prep["feature_names"]]
        return Xf, y, mr["class_names"], fn

    # Build index via intersection of paths
    idx_m = {p:i for i,p in enumerate(mr["paths"])}
    idx_p = {p:i for i,p in enumerate(prep["paths"])}
    inter = [p for p in mr["paths"] if p in idx_p]
    if not inter: raise RuntimeError("No overlapping paths between mr_run and prep_run.")
    m_ix = np.array([idx_m[p] for p in inter], int)
    p_ix = np.array([idx_p[p] for p in inter], int)

    # Consistency checks
    y_m = mr["y"][m_ix]; y_p = prep["y"][p_ix]
    if not np.all(y_m == y_p):
        raise RuntimeError("Label mismatch after path alignment.")

    Xf = np.hstack([mr["X"][m_ix], prep["X"][p_ix]])
    fn = [f"mr:{n}" for n in mr["feature_names"]] + [f"prep:{n}" for n in prep["feature_names"]]
    return Xf, y_m, mr["class_names"], fn

# ---------- Scoring methods ----------
def rankdata_desc(x: np.ndarray) -> np.ndarray:
    """Ranks: highest score -> rank 1. Ties get average rank."""
    order = (-x).argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    # average ties
    # do a stable tie average pass
    vals = x[order]
    i = 0
    while i < len(vals):
        j = i+1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order][i:j].mean()
            ranks[order][i:j] = avg
        i = j
    return ranks

def safe_subsample(X, y, nmax, seed=RANDOM_STATE):
    if X.shape[0] <= nmax: return X, y
    rng = np.random.default_rng(seed)
    sel = rng.choice(X.shape[0], size=nmax, replace=False)
    return X[sel], y[sel]

def get_scores_mi(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)

def get_scores_anova(X, y):
    F, _ = f_classif(X, y); F[np.isnan(F)] = 0.0; return F

def get_scores_l1_logreg(X, y):
    # scale for L1 fairness
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    # multinomial L1 with saga
    clf = LogisticRegression(
        penalty="l1", solver="saga", multi_class="multinomial",
        C=1.0, max_iter=500, n_jobs=-1, random_state=RANDOM_STATE, tol=1e-3
    )
    clf.fit(Xs, y)
    coefs = np.abs(clf.coef_)              # (n_classes, n_feat)
    return coefs.mean(axis=0)              # mean abs coef

def get_scores_extratrees(X, y):
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=8, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    et.fit(X, y)
    return et.feature_importances_

# ---------- Correlation pruning ----------
def corr_prune(X, cand_idx, k, thr=DEFAULT_CORR_THR, seed=RANDOM_STATE):
    """
    Greedy keep-first: iterate candidate order and keep a feature
    if its |corr| with all kept features < thr. Uses a subsample for speed.
    """
    Xsub, _ = safe_subsample(X, np.zeros(X.shape[0]), nmax=DEFAULT_SUBSAMPLE, seed=seed)
    # standardize once
    mu = Xsub.mean(axis=0); std = Xsub.std(axis=0); std[std==0] = 1.0
    Z = (Xsub - mu) / std

    kept: List[int] = []
    Z_kept = None
    for idx in cand_idx:
        z = Z[:, idx:idx+1]  # (n,1)
        if Z_kept is None:
            kept.append(idx); Z_kept = z
        else:
            # corr = cosine similarity since Z is standardized
            # corr with each kept: (z.T @ Z_kept)/n
            c = (z.T @ Z_kept) / Z.shape[0]
            if np.max(np.abs(c)) < thr:
                kept.append(idx)
                Z_kept = np.hstack([Z_kept, z])
        if len(kept) >= k:
            break
    return np.array(kept, dtype=int)

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser("Fuse mr_run + prep_run features, rank, and select a compact set.")
    ap.add_argument("--mr_dir", type=str, default=None, help="Path to mr_run_* (MiniROCKET) folder.")
    ap.add_argument("--prep_dir", type=str, default=None, help="Path to prep_run_* (handcrafted) folder.")
    ap.add_argument("--out", type=str, default=str(ARTIFACTS), help="Artifacts root.")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Target number of features.")
    ap.add_argument("--min_prep", type=int, default=DEFAULT_MIN_PREP, help="Minimum handcrafted features to keep.")
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE, help="Row subsample for scoring/pruning.")
    ap.add_argument("--corr_thr", type=float, default=DEFAULT_CORR_THR, help="Correlation pruning threshold.")
    ap.add_argument("--weights", type=str, default="mi=1,anova=1,l1=1,trees=1",
                    help="Comma list of method=weight.")
    args = ap.parse_args()

    out_root = Path(args.out)
    run_dir = out_root / f"featselect_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input dirs (latest if not provided)
    mr_latest = latest_dir(out_root, "mr_topk_run_")
    prep_latest = latest_dir(out_root, "prep_run_")
    mr_dir = ensure_dir(args.mr_dir, mr_latest, "mr_run directory")
    prep_dir = ensure_dir(args.prep_dir, prep_latest, "prep_run directory")

    print(f"[info] mr_dir  = {mr_dir}")
    print(f"[info] prep_dir= {prep_dir}")

    # Load splits
    mr_tr, mr_va, mr_te = load_split(mr_dir, "train"), load_split(mr_dir, "val"), load_split(mr_dir, "test")
    pr_tr, pr_va, pr_te = load_split(prep_dir, "train"), load_split(prep_dir, "val"), load_split(prep_dir, "test")

    # Align + fuse per split
    Xtr, ytr, classes, feat_names = align_concat(mr_tr, pr_tr)
    Xva, yva, _, _ = align_concat(mr_va, pr_va)
    Xte, yte, _, _ = align_concat(mr_te, pr_te)

    # Keep JSR/CNR/paths from mr_run if available (they should match after alignment)
    # For saving later we only need arrays; alignment already based on paths.
    print(f"[info] Train {Xtr.shape} Val {Xva.shape} Test {Xte.shape}")
    print(f"[info] #features fused: {len(feat_names)} (mr:{sum(n.startswith('mr:') for n in feat_names)}, "
          f"prep:{sum(n.startswith('prep:') for n in feat_names)})")

    # Cast to float32 to save RAM
    Xtr = Xtr.astype(np.float32, copy=False)
    Xva = Xva.astype(np.float32, copy=False)
    Xte = Xte.astype(np.float32, copy=False)

    # Subsample TRAIN for fast scoring
    Xs, ys = safe_subsample(Xtr, ytr, nmax=args.subsample, seed=RANDOM_STATE)
    print(f"[info] scoring on subsample: {Xs.shape}")

    # Parse weights
    w = {"mi":1.0,"anova":1.0,"l1":1.0,"trees":1.0}
    for kv in args.weights.split(","):
        if not kv.strip(): continue
        k,v = kv.split("="); w[k.strip()] = float(v)

    # Compute scores (try/except to skip if any method fails)
    scores: Dict[str, np.ndarray] = {}
    try:
        scores["mi"] = get_scores_mi(Xs, ys);                 print("[score] MI done")
    except Exception as e:
        print(f"[warn] MI failed: {e}")
    try:
        scores["anova"] = get_scores_anova(Xs, ys);           print("[score] ANOVA done")
    except Exception as e:
        print(f"[warn] ANOVA failed: {e}")
    try:
        scores["l1"] = get_scores_l1_logreg(Xs, ys);          print("[score] L1-LogReg done")
    except Exception as e:
        print(f"[warn] L1-LogReg failed: {e}")
    try:
        scores["trees"] = get_scores_extratrees(Xs, ys);       print("[score] ExtraTrees done")
    except Exception as e:
        print(f"[warn] ExtraTrees failed: {e}")

    if not scores:
        print("[fatal] No scoring method succeeded."); sys.exit(1)

    # Rank per method, ensemble by weighted average of normalized inverse ranks
    m = Xtr.shape[1]
    rankmat = []
    for name, s in scores.items():
        r = rankdata_desc(s)          # 1..m (1 is best)
        inv = 1.0 / r                 # higher is better
        rankmat.append(w.get(name, 0.0) * (inv / inv.max()))
    ens = np.zeros(m, float)
    for rr in rankmat: ens += rr

    # Get candidates sorted by ensemble
    cand = np.argsort(-ens)  # best first

    # Enforce minimum handcrafted quota
    prep_mask = np.array([n.startswith("prep:") for n in feat_names])
    prep_cand = [i for i in cand if prep_mask[i]]
    mr_cand   = [i for i in cand if not prep_mask[i]]

    # Start with top min_prep handcrafted
    selected_seed = prep_cand[:min(args.min_prep, len(prep_cand))]
    seed_set = set(selected_seed)
    # Fill remainder by correlation-pruned pass over remaining candidates
    remaining = [i for i in cand if i not in seed_set]
    target_k = min(args.k, m)
    ordered = selected_seed + remaining

    sel_idx = corr_prune(Xtr, ordered, k=target_k, thr=args.corr_thr, seed=RANDOM_STATE)

    sel_names = [feat_names[i] for i in sel_idx]
    n_prep_sel = sum(n.startswith("prep:") for n in sel_names)
    n_mr_sel   = len(sel_names) - n_prep_sel
    print(f"[info] selected {len(sel_idx)} features  (prep:{n_prep_sel}, mr:{n_mr_sel})")

    # ---- Save reduced NPZs ----
    def save_split(split_name: str, X, y, classes, sel_idx, src_dir: Path, out_dir: Path):
        z = np.load(src_dir / f"{split_name}_features.npz", allow_pickle=True)
        jsr = z.get("jsr", None); cnr = z.get("cnr", None)
        paths = z.get("paths", None)
        np.savez_compressed(
            out_dir / f"{split_name}_features.npz",
            X=X[:, sel_idx],
            y=y,
            jsr=jsr, cnr=cnr,
            class_names=np.array(classes, dtype=object),
            feature_names=np.array([feat_names[i] for i in sel_idx], dtype=object),
            paths=paths
        )

    fused_dir = run_dir / "fused_selected"
    fused_dir.mkdir(parents=True, exist_ok=True)

    # We will carry meta from mr_dir for jsr/cnr/paths (already aligned)
    save_split("train", Xtr, ytr, classes, sel_idx, mr_dir, fused_dir)
    save_split("val",   Xva, yva, classes, sel_idx, mr_dir, fused_dir)
    save_split("test",  Xte, yte, classes, sel_idx, mr_dir, fused_dir)

    # ---- Save ranking table ----
    rows = []
    for i in range(len(feat_names)):
        row = {
            "idx": i,
            "feature": feat_names[i],
            "source": "prep" if feat_names[i].startswith("prep:") else "mr",
            "ensemble_score": float(ens[i]),
            "selected": int(i in set(sel_idx))
        }
        for name in ["mi","anova","l1","trees"]:
            if name in scores:
                row[f"{name}_score"] = float(scores[name][i])
        rows.append(row)
    rows.sort(key=lambda r: r["ensemble_score"], reverse=True)
    save_csv(run_dir / "feature_ranking.csv",
             rows,
             ["idx","feature","source","ensemble_score","selected","mi_score","anova_score","l1_score","trees_score"])

    # ---- Quick sanity: simple val macro-F1 with a tiny logistic regression (fast) ----
    try:
        clf = LogisticRegression(max_iter=500, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(Xtr[:, sel_idx], ytr)
        yhat = clf.predict(Xva[:, sel_idx])
        f1m = f1_score(yva, yhat, average="macro")
        with open(run_dir / "quick_val_check.txt", "w", encoding="utf-8") as f:
            f.write(f"LogReg macro-F1 on VAL with {len(sel_idx)} features: {f1m:.6f}\n")
        print(f"[check] LogReg macro-F1 on VAL = {f1m:.4f}")
    except Exception as e:
        print(f"[check] quick validation skipped: {e}")

    # ---- Save indices for reproducibility ----
    np.save(run_dir / "selected_indices.npy", sel_idx)
    print(f"[done] Wrote reduced datasets to: {fused_dir}")
    print(f"[done] Ranking CSV: {run_dir / 'feature_ranking.csv'}")
    print(f"[done] Selected indices: {run_dir / 'selected_indices.npy'}")

if __name__ == "__main__":
    main()
