#!/usr/bin/env python3
# scan_incidents_only_context_noplots.py

"""
Blockwise SBF inference ONLY around incidents (context windows), ONLY for folders present in incidents.txt.

Key changes vs your current script:
- Scans ONLY folders referenced by incidents.txt (e.g., 25143, 25144, ...).
- Processes ONLY blocks inside +/- (INCIDENT_CONTEXT_TOTAL_MINUTES/2) around each incident time.
- Does NOT generate spectrogram plots during scanning.
- Writes block_index (BBSamples index within the file) so a separate script can re-open the SBF and plot later.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# -------- SBF parser ----------
try:
    from sbf_parser import load  # pip install sbf-parser
except ImportError as e:
    raise SystemExit("Install first:  pip install sbf-parser") from e

# -------- PyTorch ----------
try:
    import torch
except ImportError as e:
    raise SystemExit("Install first:  pip install torch") from e


# =============================================================================
#                                   CONFIG
# =============================================================================

@dataclass
class Config:
    ROOT_DIR: Path = Path(r"E:\Roadtest")
    PROJECT_DIR: Path = Path(
        r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\1. DTU\4. Fall 2025\gnss jamming\GNSS-jamming-classifier"
    )
    OUT_ROOT: Path = Path(r"E:\Roadtest\scan_outputs")
    MODEL_PT: Path = Path(r"..\artifacts\finetuned_DL\finetune_spec_20251216_161529\model_finetuned.pt")

    # Incidents file
    INCIDENTS_TXT: Path = ROOT_DIR / "incidents.txt"

    # ---- context window around each incident ----
    # Total window length. Default 10 => +/- 5 minutes.
    INCIDENT_CONTEXT_TOTAL_MINUTES: float = 5.0

    # If True: process ONLY blocks that fall inside any context window (recommended)
    PROCESS_ONLY_CONTEXT: bool = False

    # If dt_start is missing and we cannot time-filter:
    # - True => skip those blocks (strict)
    # - False => process anyway (but they won't be marked in_context)
    REQUIRE_TIME_TAGS_FOR_CONTEXT_FILTER: bool = False

    # Logging
    WRITE_FULL_PREDICTIONS: bool = False
    WRITE_CONTEXT_PREDICTIONS: bool = True

    # Detection threshold and merging
    DETECT_PROB: float = 0.80
    MERGE_GAP_S: float = 2.0

    # LO filter
    PROCESS_ONLY_LO_HZ: Optional[int] = None

    # Runtime
    DEVICE: str = "auto"
    SECONDS_PER_BLOCK: float = 0.0  # global pace: 1 processed block per second

    # Debug
    LOG_FIRST_N_BBSAMPLE_KEYS: int = 3
    LOG_FIRST_N_BLOCK_LENS: int = 5


CFG = Config()


# =============================================================================
#                              INCIDENT PARSING
# =============================================================================

_INCIDENT_LINE_RE = re.compile(
    r"""^\s*(?P<date>\d{2}/\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+[:\t ]+(?P<folder>\d{5})\s*:.*$"""
)

def parse_incidents(path: Path) -> Dict[str, List[datetime]]:
    """
    incidents.txt line format:
      23/05-2025 17:39:58  25143 : ...
    """
    out: Dict[str, List[datetime]] = {}
    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or raw == "...":
            continue
        m = _INCIDENT_LINE_RE.match(raw)
        if not m:
            continue

        d = m.group("date")  # dd/mm-YYYY
        t = m.group("time")
        folder = m.group("folder")

        dd = d.split("/")[0]
        mm = d.split("/")[1].split("-")[0]
        yyyy = d.split("-")[1]
        dt = datetime.strptime(f"{dd}/{mm}/{yyyy} {t}", "%d/%m/%Y %H:%M:%S")
        out.setdefault(folder, []).append(dt)

    for k in list(out.keys()):
        out[k] = sorted(out[k])
    return out

def build_context_windows(times: List[datetime], total_minutes: float) -> List[Tuple[datetime, datetime]]:
    if not times:
        return []
    half = timedelta(seconds=float(total_minutes) * 60.0 / 2.0)
    return [(t - half, t + half) for t in times]

def in_any_window(dt: datetime, windows: List[Tuple[datetime, datetime]]) -> bool:
    return any(a <= dt <= b for (a, b) in windows)


# =============================================================================
#                         SBF IQ EXTRACT (BBSamples)
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
#                             MODEL + PREPROC LOADING
# =============================================================================

def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # torch>=2.1
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_model_and_preproc(model_path: Path):
    bundle = torch_load_bundle(model_path)
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

    C = bundle.get("config", None)
    if not isinstance(C, dict):
        C = {}
    config = dict(C)

    # Import your training module for the exact same preprocessing helpers
    sys.path.insert(0, str(CFG.PROJECT_DIR))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_eval_cnn_spectrogram as TE  # noqa: E402

    model_kind = str(config.get("MODEL", "se_cnn"))
    in_ch = 3 if str(preproc.get("spec_mode", "logpow")) == "logpow_phase3" else 1
    K = len(classes)

    if model_kind == "cnn":
        model = TE.SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=False)
    elif model_kind == "se_cnn":
        model = TE.SpecCNN2D(in_ch=in_ch, num_classes=K, use_se=True)
    elif model_kind == "vit":
        model = TE.SpecViT(
            in_ch=in_ch,
            num_classes=K,
            patch=int(config.get("VIT_PATCH", 8)),
            embed_dim=int(config.get("VIT_EMBED_DIM", 192)),
            depth=int(config.get("VIT_DEPTH", 6)),
            heads=int(config.get("VIT_HEADS", 6)),
            mlp_ratio=float(config.get("VIT_MLP_RATIO", 4.0)),
            dropout=float(config.get("VIT_DROPOUT", 0.1)),
        )
    else:
        raise ValueError(f"Unknown MODEL={model_kind}")

    state = bundle.get("model_state", None) or bundle.get("state_dict", None)
    if state is None:
        raise RuntimeError("Checkpoint bundle missing model_state/state_dict.")
    model.load_state_dict(state, strict=False)

    return model, classes, preproc, config, TE


# =============================================================================
#                               SBF TIME EXTRACTION
# =============================================================================

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

def _get_key_ci(d: Dict[str, Any], name: str) -> Optional[Any]:
    if name in d:
        return d[name]
    lname = name.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == lname:
            return v
    return None

def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[Any]]:
    for k in keys:
        v = _get_key_ci(d, k)
        if v is not None:
            return k, v
    return None, None

def extract_block_datetime(info: Dict[str, Any]) -> Optional[datetime]:
    wk_key, wk_val = _first_present(info, ("WNc", "WN", "Week", "GPSWeek", "GpsWeek", "WeekNumber"))
    tow_key, tow_val = _first_present(info, ("TOW", "Tow", "TimeOfWeek", "TowMs", "TOW_ms", "tow_ms", "TOWmsec", "TowMsec"))

    if wk_val is not None and tow_val is not None:
        try:
            wk = int(wk_val)
            tow = float(tow_val)

            if tow_key is not None and ("ms" in tow_key.lower() or "msec" in tow_key.lower()):
                tow = tow / 1000.0
            elif tow > 700_000:
                tow = tow / 1000.0

            if tow < 0 or tow > 7 * 24 * 3600 + 1:
                return None

            return GPS_EPOCH + timedelta(weeks=wk, seconds=tow)
        except Exception:
            return None

    return None


# =============================================================================
#                           DETECTION MERGING
# =============================================================================

@dataclass
class DetSeg:
    folder: str
    file: str
    lo_hz: int
    t_start: Optional[datetime]
    t_end: Optional[datetime]
    offset_start_s: float
    offset_end_s: float
    p_intf_max: float
    pred_mode: str

def _gap_seconds(cur: DetSeg, nxt_t: Optional[datetime], nxt_off_s: float) -> float:
    if cur.t_end is not None and nxt_t is not None:
        return (nxt_t - cur.t_end).total_seconds()
    return float(nxt_off_s) - float(cur.offset_end_s)


# =============================================================================
#                                 MAIN
# =============================================================================

@dataclass
class StreamState:
    fs_hz: float
    sample_count: int = 0
    t0_dt: Optional[datetime] = None
    t0_sample: int = 0
    saw_time_tags: bool = False


def main() -> int:
    out_root = CFG.OUT_ROOT.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "logs.txt"
    pred_path = run_dir / "predictions.csv"
    pred_ctx_path = run_dir / "predictions_context_10min.csv"
    det_path = run_dir / "detections.csv"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    incidents_by_folder = parse_incidents(CFG.INCIDENTS_TXT)
    total_inc = sum(len(v) for v in incidents_by_folder.values())
    log(f"[INCIDENTS] folders={len(incidents_by_folder)} total_entries={total_inc}")
    log(f"[CONTEXT] total_minutes={CFG.INCIDENT_CONTEXT_TOTAL_MINUTES} => +/-{CFG.INCIDENT_CONTEXT_TOTAL_MINUTES/2:.1f} min")
    log(f"[CONTEXT] process_only_context={CFG.PROCESS_ONLY_CONTEXT} require_time_tags={CFG.REQUIRE_TIME_TAGS_FOR_CONTEXT_FILTER}")

    # ONLY scan folders that exist AND appear in incidents.txt
    folders: List[Path] = []
    for folder_id in sorted(incidents_by_folder.keys()):
        p = (CFG.ROOT_DIR / folder_id)
        if p.is_dir():
            folders.append(p)
    log(f"[SCAN] folders={len(folders)} (from incidents.txt only)")

    device = resolve_device(CFG.DEVICE)
    model, classes, preproc, config, TE = load_model_and_preproc(CFG.MODEL_PT)
    model.to(device)
    model.eval()

    # Model preproc (must match training)
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

    log(f"[MODEL] {CFG.MODEL_PT}")
    log(f"[CLASSES] {classes}")
    log(f"[MODEL-PREPROC] target_len={target_len} nfft={nfft} win={win} hop={hop} mode={spec_mode} norm={spec_norm} fftshift={fftshift}")
    log(f"[DEVICE] {device}")
    log(f"[PACE] seconds_per_block={CFG.SECONDS_PER_BLOCK}")
    log("[PLOTS] disabled during scan; use the separate plotter script.")

    prob_fields = [f"p_{c}" for c in classes]
    pred_fields = [
        "folder", "file", "lo_hz",
        "block_index",          # <= BBSamples index within the file (counts ALL BBSamples blocks)
        "t_utc", "offset_s", "n_samp",
        "in_context",
        "pred", "p_pred", "p_intf",
    ] + prob_fields

    # detections merging state
    cur_seg: Dict[Tuple[str, str, int], DetSeg] = {}
    finished_segs: List[DetSeg] = []

    # Open CSVs
    f_pred = pred_path.open("w", newline="", encoding="utf-8") if CFG.WRITE_FULL_PREDICTIONS else None
    w_pred = csv.DictWriter(f_pred, fieldnames=pred_fields) if f_pred else None
    if w_pred:
        w_pred.writeheader()

    f_ctx = pred_ctx_path.open("w", newline="", encoding="utf-8") if CFG.WRITE_CONTEXT_PREDICTIONS else None
    w_ctx = csv.DictWriter(f_ctx, fieldnames=pred_fields) if f_ctx else None
    if w_ctx:
        w_ctx.writeheader()

    # Global tick pacing (stable 1Hz schedule)
    next_tick: Optional[float] = None

    try:
        for folder in folders:
            folder_id = folder.name
            incident_times = incidents_by_folder.get(folder_id, [])
            ctx_windows = build_context_windows(incident_times, CFG.INCIDENT_CONTEXT_TOTAL_MINUTES)
            log(f"[FOLDER] {folder_id} incidents={len(incident_times)} ctx_windows={len(ctx_windows)}")

            # Find SBF files in this folder
            sbf_files: List[Path] = []
            for g in ("*.25_", "*.25"):
                sbf_files.extend(folder.glob(g))
            sbf_files = sorted({p.resolve() for p in sbf_files if p.is_file()})
            if not sbf_files:
                log(f"  [WARN] no SBF files found in {folder_id}")
                continue

            for sbf in sbf_files:
                log(f"  [FILE] {sbf.name}")

                streams: Dict[int, StreamState] = {}
                logged_keys = 0
                logged_lens = 0

                n_blocks_total = 0
                n_blocks_processed = 0
                n_blocks_in_context = 0

                bb_index_in_file = 0  # counts ALL BBSamples blocks in this file

                with sbf.open("rb") as f:
                    for block_type, info in load(f):
                        if block_type != "BBSamples":
                            continue

                        bb_index_in_file += 1  # IMPORTANT: increment for every BBSamples block

                        if logged_keys < int(CFG.LOG_FIRST_N_BBSAMPLE_KEYS):
                            log(f"    [BBSamples keys] {sorted([str(k) for k in info.keys()])}")
                            logged_keys += 1

                        Fs = int(info.get("SampleFreq", 0))
                        LO = int(info.get("LOFreq", 0))
                        samples = info.get("Samples", None)
                        if Fs <= 0 or samples is None:
                            continue

                        if CFG.PROCESS_ONLY_LO_HZ is not None and LO != int(CFG.PROCESS_ONLY_LO_HZ):
                            continue

                        words = as_u16_array(samples)
                        I8, Q8 = unpack_iq_int8(words)
                        N = int(I8.size)
                        if N <= 0:
                            continue

                        n_blocks_total += 1
                        if logged_lens < int(CFG.LOG_FIRST_N_BLOCK_LENS):
                            log(f"    [BBSamples] idx={bb_index_in_file} LO={LO} Fs={Fs} N={N} (≈{(N/float(Fs))*1e6:.2f} µs)")
                            logged_lens += 1

                        st = streams.get(LO)
                        if st is None:
                            streams[LO] = StreamState(fs_hz=float(Fs))
                            st = streams[LO]
                        else:
                            if abs(float(Fs) - float(st.fs_hz)) > 1e-6:
                                st.fs_hz = float(Fs)

                        fs_used = float(Fs) if use_npz_fs else fs_default

                        # Timestamp
                        block_dt = extract_block_datetime(info)
                        if block_dt is not None:
                            st.saw_time_tags = True
                            if st.t0_dt is None:
                                st.t0_dt = block_dt
                                st.t0_sample = int(st.sample_count)
                            else:
                                pred_dt = st.t0_dt + timedelta(seconds=float(st.sample_count - st.t0_sample) / fs_used)
                                if abs((block_dt - pred_dt).total_seconds()) > 2.0:
                                    st.t0_dt = block_dt
                                    st.t0_sample = int(st.sample_count)

                        dt_start: Optional[datetime] = block_dt
                        if dt_start is None and st.t0_dt is not None:
                            dt_start = st.t0_dt + timedelta(seconds=float(st.sample_count - st.t0_sample) / fs_used)

                        # Context membership
                        in_ctx = False
                        if ctx_windows and dt_start is not None:
                            in_ctx = in_any_window(dt_start, ctx_windows)

                        # Process only context blocks
                        if CFG.PROCESS_ONLY_CONTEXT:
                            if dt_start is None:
                                if CFG.REQUIRE_TIME_TAGS_FOR_CONTEXT_FILTER:
                                    st.sample_count += N
                                    continue
                                # else: cannot filter reliably -> treat as out of context
                                if ctx_windows:
                                    st.sample_count += N
                                    continue
                            else:
                                if not in_ctx:
                                    st.sample_count += N
                                    continue

                        # True offset in signal time
                        offset_s = float(st.sample_count) / fs_used
                        offset_s_end = offset_s + (float(N) / fs_used)
                        dt_end = (dt_start + timedelta(seconds=float(N) / fs_used)) if dt_start else None

                        # Pace scheduling start for this processed block
                        now = time.perf_counter()
                        if float(CFG.SECONDS_PER_BLOCK) > 0.0:
                            if next_tick is None:
                                next_tick = now
                            next_tick = next_tick + float(CFG.SECONDS_PER_BLOCK)

                        # Build complex block
                        z_raw = (I8.astype(np.float32) / 128.0) + 1j * (Q8.astype(np.float32) / 128.0)
                        z_raw = z_raw.astype(np.complex64, copy=False)

                        # Enforce model length
                        z_in = TE.center_crop_or_pad(z_raw, target_len)
                        z_in = TE.normalize_iq(z_in.copy())

                        # Model spectrogram
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

                        # Inference (single block)
                        xt = torch.from_numpy(S[None, ...]).to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            logits = model(xt)
                            if isinstance(logits, (tuple, list)):
                                logits = logits[0]
                            pr = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)

                        pred_i = int(np.argmax(pr))
                        pred = classes[pred_i] if pred_i < len(classes) else str(pred_i)
                        p_pred = float(pr[pred_i])
                        p_nojam = float(pr[0]) if pr.size > 0 else float("nan")
                        p_intf = float(1.0 - p_nojam) if not math.isnan(p_nojam) else float("nan")

                        row = {
                            "folder": folder_id,
                            "file": sbf.name,
                            "lo_hz": LO,
                            "block_index": str(bb_index_in_file),
                            "t_utc": dt_start.isoformat() if dt_start else "",
                            "offset_s": f"{offset_s:.6f}",
                            "n_samp": str(N),
                            "in_context": "1" if in_ctx else "0",
                            "pred": pred,
                            "p_pred": f"{p_pred:.6f}",
                            "p_intf": f"{p_intf:.6f}",
                        }
                        for k, c in enumerate(classes):
                            row[f"p_{c}"] = f"{float(pr[k]) if k < pr.size else float('nan'):.6f}"

                        if w_pred is not None:
                            w_pred.writerow(row)
                            f_pred.flush()

                        if w_ctx is not None:
                            w_ctx.writerow(row)
                            f_ctx.flush()
                            n_blocks_in_context += 1

                        n_blocks_processed += 1

                        # Update merged detections
                        if (not math.isnan(p_intf)) and (p_intf >= float(CFG.DETECT_PROB)):
                            key = (folder_id, sbf.name, int(LO))
                            cur = cur_seg.get(key)
                            if cur is None:
                                cur_seg[key] = DetSeg(
                                    folder=folder_id,
                                    file=sbf.name,
                                    lo_hz=int(LO),
                                    t_start=dt_start,
                                    t_end=dt_end,
                                    offset_start_s=float(offset_s),
                                    offset_end_s=float(offset_s_end),
                                    p_intf_max=float(p_intf),
                                    pred_mode=pred,
                                )
                            else:
                                gap = _gap_seconds(cur, dt_start, float(offset_s))
                                if gap <= float(CFG.MERGE_GAP_S):
                                    cur.offset_end_s = max(cur.offset_end_s, float(offset_s_end))
                                    if cur.t_end is not None and dt_end is not None:
                                        cur.t_end = max(cur.t_end, dt_end)
                                    elif cur.t_end is None:
                                        cur.t_end = dt_end
                                    if float(p_intf) >= float(cur.p_intf_max):
                                        cur.p_intf_max = float(p_intf)
                                        cur.pred_mode = pred
                                else:
                                    finished_segs.append(cur)
                                    cur_seg[key] = DetSeg(
                                        folder=folder_id,
                                        file=sbf.name,
                                        lo_hz=int(LO),
                                        t_start=dt_start,
                                        t_end=dt_end,
                                        offset_start_s=float(offset_s),
                                        offset_end_s=float(offset_s_end),
                                        p_intf_max=float(p_intf),
                                        pred_mode=pred,
                                    )

                        # advance sample counter by true block length
                        st.sample_count += N

                        # Pace limiter
                        if float(CFG.SECONDS_PER_BLOCK) > 0.0 and next_tick is not None:
                            sleep_s = next_tick - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            else:
                                next_tick = time.perf_counter()

                # diagnostics per file
                for lo_hz, st in streams.items():
                    dur_s = float(st.sample_count) / float(st.fs_hz) if st.fs_hz > 0 else float("nan")
                    if st.t0_dt is not None:
                        t_end = st.t0_dt + timedelta(seconds=float(st.sample_count - st.t0_sample) / float(st.fs_hz))
                        log(f"    [LO {lo_hz}] samples={st.sample_count} fs={st.fs_hz:.3f} dur≈{dur_s:.3f}s "
                            f"t0={st.t0_dt.isoformat()} t_end≈{t_end.isoformat()}")
                    else:
                        log(f"    [LO {lo_hz}] samples={st.sample_count} fs={st.fs_hz:.3f} dur≈{dur_s:.3f}s t0=?")

                log(f"    blocks_total_seen={n_blocks_total} processed={n_blocks_processed} written={n_blocks_in_context}")

        # finalize open detection segments
        finished_segs.extend(cur_seg.values())
        cur_seg.clear()

        # write detections.csv
        with det_path.open("w", newline="", encoding="utf-8") as f_det:
            w_det = csv.DictWriter(
                f_det,
                fieldnames=[
                    "folder", "file", "lo_hz",
                    "t_start", "t_end",
                    "offset_start_s", "offset_end_s",
                    "p_intf_max", "pred_mode",
                ],
            )
            w_det.writeheader()
            for seg in finished_segs:
                w_det.writerow({
                    "folder": seg.folder,
                    "file": seg.file,
                    "lo_hz": seg.lo_hz,
                    "t_start": seg.t_start.isoformat() if seg.t_start else "",
                    "t_end": seg.t_end.isoformat() if seg.t_end else "",
                    "offset_start_s": f"{float(seg.offset_start_s):.6f}",
                    "offset_end_s": f"{float(seg.offset_end_s):.6f}",
                    "p_intf_max": f"{float(seg.p_intf_max):.6f}",
                    "pred_mode": seg.pred_mode,
                })

    finally:
        if f_pred is not None:
            f_pred.close()
        if f_ctx is not None:
            f_ctx.close()

    log(f"[DONE] run_dir={run_dir}")
    if CFG.WRITE_FULL_PREDICTIONS:
        log(f"[DONE] {pred_path}")
    if CFG.WRITE_CONTEXT_PREDICTIONS:
        log(f"[DONE] {pred_ctx_path}")
    log(f"[DONE] {det_path}")
    log(f"[DONE] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
