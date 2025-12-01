# sbf_multiband_spectrograms_simple_multifile.py
#
# Sample IQ from ALL Septentrio SBF files in a folder (including subfolders)
# every X seconds, for EACH LO/band logged in the BBSamples block,
# and save spectrograms.
#
# Extra:
# - Reads navigation blocks (PVTGeodetic / PVTGeodetic2 / PVTCartesian / PVTCartesian2
#   + DOP / DOP2 + PosCovGeodetic / PosCovGeodetic2 + MeasEpoch).
# - For each BBSamples snap, attaches the most recent nav data within
#   NAV_MAX_AGE_SEC *in the past* and stamps on every spectrogram:
#       * Sol=YES/NO   (valid solution or not)
#       * Lat / Lon / height (even if position is logged in PVTCartesian)
#       * Nr of satellites, PVT Mode, Error
#       * HDOP / VDOP / PDOP   (properly scaled: centi-DOP → DOP)
#       * Position 1σ std-dev (σE, σN, σU) from PosCovGeodetic
#       * CNR (C/N0) median over all channels decoded from MeasEpoch
#
# Uses: https://github.com/septentrio-gnss/SbfParser

from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import stft
from datetime import datetime, timedelta, timezone
import math

from sbf_parser import SbfParser

# ============================ CONFIG ============================

SBF_DIR = r"E:\Roadtest"
OUT_ROOT = r"E:\Roadtest\Plots"

# One snap & plot every SAMPLE_PERIOD_SEC (UTC) *per LO/band*
SAMPLE_PERIOD_SEC = 30
MAX_SAMPLES_PER_FILE = 6000

# Snap duration (microseconds) used for the spectrograms
SNAP_DUR_US = 33.0

# Optional decimation of IQ before spectrogram
DECIM = 1
NPERSEG_STFT = 128
NOVERLAP_STFT = 96
DPI_FIG = 120
VMIN_DB = -80
VMAX_DB = -20
REMOVE_DC = True
CHUNK_BYTES = 1_000_000

# Max time offset (s) to associate nav blocks to a BBSamples snap
NAV_MAX_AGE_SEC = 1.0

# SBF DOP block (HDOP/VDOP/PDOP) is in centi-DOP units (0.01)
DOP_SCALE = 0.01

# --------- MeasEpoch → CN0 decoding ---------
# In MeasEpochChannelType1 (SB1), CN0 is a 1-byte field.
# Layout (byte indices inside each SB1):
#   0  rx_channel
#   1  type (SigIdxLo in lower 5 bits)
#   2  sv_id
#   3  misc
#   4–7   code_lsb (uint32)
#   8–11  doppler  (int32)
#   12–13 carrier_lsb (uint16)
#   14    carrier_msb (int8)
#   15    cn0 (uint8)  <-- HERE
#   16–17 lock_time (uint16)
#   18    obs_info (uint8)
#   19    n2 (uint8)
#
# According to the Septentrio SBF manual:
#   C/N0 [dB-Hz] = CN0 * 0.25                 if signal number in {1, 2}
#   C/N0 [dB-Hz] = CN0 * 0.25 + 10           otherwise
# where "signal number" is the lower 5 bits of Type (SigIdxLo).
CN0_OFFSET_SB1 = 15  # byte index of CN0 within each Type1 sub-block
CN0_LSB_DB = 0.25    # 0.25 dB-Hz per LSB

# For "summary" stats we ignore very weak / marginal channels.
# If no CN0 is above this threshold, we fall back to all valid CN0.
CN0_MIN_FOR_STATS_DBHZ = 25.0

# ===============================================================

EPS = 1e-20
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (stable since 2017-01-01)


# ---------- time helpers ----------

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600)
    m = int((tsec % 3600) // 60)
    s = tsec - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)


def extract_time_labels(infos: dict) -> Tuple[int, float, str, str, str, datetime]:
    """
    Generic time extractor for any SBF block.

    Tries both WNc/WN and TOW/Tow/TOW_ms/Tow_ms.
    Handles TOW in seconds or milliseconds.
    """
    # Week
    if "WNc" in infos:
        wnc = int(infos["WNc"])
    elif "WN" in infos:
        wnc = int(infos["WN"])
    else:
        wnc = -1

    # TOW
    tow_keys = ["TOW", "Tow", "TOW_ms", "Tow_ms"]
    tow_raw: float = 0.0
    for k in tow_keys:
        if k in infos:
            tow_raw = float(infos[k])
            break

    # If TOW is bigger than a GPS week in seconds, treat as ms
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw

    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt


# -------------------- SBF IQ decode --------------------

def decode_bbsamples_iq(infos: dict):
    """
    Decode BBSamples into complex IQ + fs + LO frequency (Hz).

    Assumes:
      - 'Samples' is a bytes buffer with interleaved I/Q, int8.
      - 'N' is the number of complex samples.
      - 'SampleFreq' is in Hz.
      - 'LOFreq' is in Hz.
    """
    if "Samples" not in infos or "N" not in infos:
        return None, None, None

    buf = infos["Samples"]
    N = int(infos.get("N", 0))

    arr = np.frombuffer(buf, dtype=np.int8)
    if arr.size != 2 * N:
        return None, None, None

    I = arr[0::2].astype(np.float32) / 128.0
    Q = arr[1::2].astype(np.float32) / 128.0
    x = I + 1j * Q

    fs = float(infos.get("SampleFreq", 1.0))
    flo = float(infos.get("LOFreq", 0.0))
    return x, fs, flo


# -------------------- band naming helper --------------------

def band_name_from_lo(flo_hz: float) -> str:
    if flo_hz <= 0:
        return "Unknown"
    if 1.57e9 <= flo_hz <= 1.60e9:
        return "L1_E1"
    if 1.20e9 <= flo_hz <= 1.26e9:
        return "L2_E5b"
    if 1.16e9 <= flo_hz <= 1.20e9:
        return "L5_E5ab"
    return f"LO_{flo_hz / 1e6:.1f}MHz"


# -------------------- CN0 helpers (MeasEpoch payload) --------------------

def decode_measepoch_cn0(infos: dict) -> Optional[np.ndarray]:
    """
    Decode CN0 array from a MeasEpoch 'infos' dict produced by SbfParser.

    We only use Type1 channels (N1, SB1Length), and we apply the Septentrio
    scaling:

        C/N0 = CN0 * 0.25                  if signal number in {1, 2}
        C/N0 = CN0 * 0.25 + 10             otherwise

    where 'signal number' is the lower 5 bits of the Type byte (SigIdxLo).

    Returns:
        cn0_dBHz : np.ndarray of shape (N1,) with NaN for invalid entries,
                   or None if decoding is not possible.
    """
    payload = infos.get("payload")
    if payload is None:
        return None

    N1 = int(infos.get("N1", 0))
    sb1_len = int(infos.get("SB1Length", 0))

    if N1 <= 0 or sb1_len <= 0:
        return None

    buf = np.frombuffer(payload, dtype=np.uint8)
    expected_len = N1 * sb1_len
    if buf.size < expected_len:
        # At least the Type1 region must be present
        return None

    cn0_vals = np.full(N1, np.nan, dtype=float)

    for i in range(N1):
        base = i * sb1_len

        idx_type = base + 1                # Type byte
        idx_cn0 = base + CN0_OFFSET_SB1    # CN0 byte

        if idx_type >= buf.size or idx_cn0 >= buf.size:
            continue

        raw_cn0 = int(buf[idx_cn0])
        if raw_cn0 == 0 or raw_cn0 == 255:
            # 0 => no measurement, 255 => cannot be computed
            continue

        type_byte = int(buf[idx_type])
        sig_idx_lo = type_byte & 0x1F      # lower 5 bits = signal number (SigIdxLo)

        cn0_dbhz = raw_cn0 * CN0_LSB_DB
        if sig_idx_lo not in (1, 2):
            cn0_dbhz += 10.0

        cn0_vals[i] = cn0_dbhz

    if not np.isfinite(cn0_vals).any():
        return None

    return cn0_vals


def cn0_info_for_band(meas_infos: Optional[dict], flo_hz: float) -> Optional[dict]:
    """
    Given a MeasEpoch infos dict and an LO, compute CN0 statistics.

    For ahora no se separa por banda: band stats == overall stats,
    pero calculamos las estadísticas principales solo con canales
    "fuertes" (C/N0 >= CN0_MIN_FOR_STATS_DBHZ). Si no hay ninguno,
    usamos todos los CN0 válidos.
    """
    if meas_infos is None:
        return None

    cn0_arr = decode_measepoch_cn0(meas_infos)
    if cn0_arr is None:
        return None

    cn0 = cn0_arr.astype(float)
    mask_valid = np.isfinite(cn0) & (cn0 > 0.0)
    if not mask_valid.any():
        return None

    cn0_valid = cn0[mask_valid]

    # Canales fuertes para stats "band"
    mask_strong = cn0_valid >= CN0_MIN_FOR_STATS_DBHZ
    if mask_strong.any():
        cn0_band = cn0_valid[mask_strong]
    else:
        cn0_band = cn0_valid  # fallback

    info = {
        # Stats sobre todos los canales válidos
        "med_all": float(np.median(cn0_valid)),
        "mean_all": float(np.mean(cn0_valid)),
        "max_all": float(np.max(cn0_valid)),
        "n_all": int(cn0_valid.size),
        # Stats sobre canales fuertes (más comparables al SBF viewer)
        "med_band": float(np.median(cn0_band)),
        "max_band": float(np.max(cn0_band)),
        "n_band": int(cn0_band.size),
    }
    return info


# -------------------- ECEF → Geodetic helper --------------------

def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Convert ECEF (X,Y,Z) [m] → (lat, lon, h) in WGS-84.

    Returns (lat_deg, lon_deg, h_m) or (None, None, None) if input is bad.
    """
    try:
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)

        lon = math.atan2(y, x)
        p = math.sqrt(x * x + y * y)

        if p < 1e-6 and abs(z) < 1e-6:
            return None, None, None

        lat = math.atan2(z, p * (1.0 - e2))  # initial guess
        h = 0.0

        for _ in range(10):
            sin_lat = math.sin(lat)
            N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
            h = p / math.cos(lat) - N
            lat = math.atan2(z, p * (1.0 - e2 * N / (N + h)))

        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - N

        lat_deg = math.degrees(lat)
        lon_deg = math.degrees(lon)

        return lat_deg, lon_deg, h
    except Exception:
        return None, None, None


# -------------------- nav / accuracy text --------------------

def format_nav_lines(
    pvt_info: Optional[dict],
    dop_info: Optional[dict],
    posstd_info: Optional[dict],
    cn0_info: Optional[dict],
) -> str:
    """
    Build a concise string with Sol flag + PVT + DOP + PosStd + CNR info.
    """

    parts = []

    # ---------- solution status ----------
    sol_ok = False
    if pvt_info is not None:
        mode = pvt_info.get("mode")
        err = pvt_info.get("error")
        nsat = pvt_info.get("nr_sv")
        h = pvt_info.get("height")

        if (mode is not None and mode != 0) and (err in (0, None)):
            if nsat is None or (0 < nsat < 64):
                if h is None or abs(h) < 10000:
                    sol_ok = True

    parts.append(f"Sol={'YES' if sol_ok else 'NO'}")

    # ---------- PVT ----------
    if pvt_info is None:
        parts.append("PVT: none")
    else:
        lat = pvt_info.get("lat")
        lon = pvt_info.get("lon")
        h = pvt_info.get("height")
        nr_sv = pvt_info.get("nr_sv")
        mode = pvt_info.get("mode")
        err = pvt_info.get("error")

        sub = []
        if sol_ok and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            sub.append(f"Lat={lat:.6f}, Lon={lon:.6f}")
        if isinstance(h, (int, float)) and abs(h) < 10000:
            sub.append(f"h={h:.1f} m")
        if isinstance(nr_sv, (int, float)) and 0 < nr_sv < 256:
            sub.append(f"Nsat={int(nr_sv)}")
        if mode is not None:
            sub.append(f"Mode={mode}")
        if err is not None:
            sub.append(f"Err={err}")

        parts.append("PVT: " + (", ".join(sub) if sub else "present"))

    # ---------- DOP ----------
    if dop_info is not None:
        hdop = dop_info.get("hdop")
        vdop = dop_info.get("vdop")
        pdop = dop_info.get("pdop")
        dop_sub = []
        if isinstance(hdop, (int, float)):
            dop_sub.append(f"HDOP={hdop:.2f}")
        if isinstance(vdop, (int, float)):
            dop_sub.append(f"VDOP={vdop:.2f}")
        if isinstance(pdop, (int, float)):
            dop_sub.append(f"PDOP={pdop:.2f}")
        if dop_sub:
            parts.append("DOP: " + ", ".join(dop_sub))

    # ---------- Pos std-dev (σE, σN, σU in meters) ----------
    if posstd_info is not None:
        sE = posstd_info.get("sE")
        sN = posstd_info.get("sN")
        sU = posstd_info.get("sU")
        sub = []
        if isinstance(sE, (int, float)):
            sub.append(f"σE={sE:.2f} m")
        if isinstance(sN, (int, float)):
            sub.append(f"σN={sN:.2f} m")
        if isinstance(sU, (int, float)):
            sub.append(f"σU={sU:.2f} m")
        if sub:
            parts.append("PosStd: " + ", ".join(sub))

    # ---------- CNR / CN0 ----------
    if cn0_info is not None and cn0_info.get("n_all", 0) > 0:
        med = cn0_info.get("med_band") or cn0_info.get("med_all")
        if isinstance(med, (int, float)):
            parts.append(f"CNR (C/N0): med={med:.1f} dB-Hz")

    return "" if not parts else " | " + " | ".join(parts)


# ---------------------- plotting ----------------------

def plot_and_save(
    sample_idx: int,
    block_idx: int,
    x: np.ndarray,
    fs: float,
    flo: float,
    band_name: str,
    wnc: int,
    tow_s: float,
    tow_hms: str,
    utc_hms: str,
    utc_iso: str,
    utc_dt: datetime,
    out_dir: Path,
    pvt_info: Optional[dict] = None,
    dop_info: Optional[dict] = None,
    posstd_info: Optional[dict] = None,
    cn0_info: Optional[dict] = None,
) -> Optional[Path]:
    if x.size < 8:
        return None

    xx = x - np.mean(x) if REMOVE_DC else x

    nperseg_eff = min(int(NPERSEG_STFT), len(xx))
    noverlap_eff = min(int(NOVERLAP_STFT), max(0, nperseg_eff - 1))

    f, t, Z = stft(
        xx,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    if t.size < 2:
        nperseg_eff = max(16, min(len(xx) // 4, nperseg_eff))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = stft(
            xx,
            fs=fs,
            window="hann",
            nperseg=nperseg_eff,
            noverlap=noverlap_eff,
            return_onesided=False,
            boundary=None,
            padded=False,
        )

    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    tt = np.arange(len(xx), dtype=np.float32) / fs
    I = np.real(xx)
    Q = np.imag(xx)

    T_snap_used_us = (len(xx) / fs) * 1e6

    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    if t.size >= 2:
        pcm = ax1.pcolormesh(
            t,
            f,
            S_dB,
            shading="auto",
            vmin=VMIN_DB,
            vmax=VMAX_DB,
        )
        fig.colorbar(pcm, ax=ax1, label="dB")
    else:
        im = ax1.imshow(
            S_dB,
            aspect="auto",
            origin="lower",
            extent=[
                0.0,
                max(1.0 / fs, nperseg_eff / fs),
                f[0],
                f[-1],
            ],
            vmin=VMIN_DB,
            vmax=VMAX_DB,
        )
        fig.colorbar(im, ax=ax1, label="dB")

    band_name_label = band_name_from_lo(flo)
    lo_str = (
        f"LO={flo / 1e6:.3f} MHz ({band_name_label})"
        if flo and flo > 0
        else "LO=Unknown"
    )

    nav_line = format_nav_lines(pvt_info, dop_info, posstd_info, cn0_info)

    title = (
        f"Sample #{sample_idx} (BBSamples #{block_idx}) | GPS week {wnc} | "
        f"TOW {tow_s:.3f}s ({tow_hms}) | UTC {utc_hms} | {lo_str}\n"
        f"fs={fs / 1e6:.3f} Msps, T_snap={T_snap_used_us:.1f} µs, "
        f"nperseg={nperseg_eff}, noverlap={noverlap_eff}{nav_line}"
    )
    ax1.set_title(title)
    ax1.set_ylabel("Baseband freq [Hz]")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(tt, I, linewidth=0.7, label="I")
    ax2.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.legend(loc="upper right")
    ax2.text(
        0.01,
        0.02,
        utc_iso,
        transform=ax2.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    safe_band = band_name.replace(" ", "_")

    out_path = out_dir / (
        f"spec_{utc_dt.strftime('%H%M%S')}"
        f"_S{sample_idx:05d}_{safe_band}.png"
    )
    fig.savefig(out_path, dpi=DPI_FIG, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ======================= PER-FILE PROCESSOR =======================

def process_single_sbf(sbf_path: Path, out_root: Path, root_dir: Path):
    """
    Process one SBF file:
      - create output folder OUT_ROOT/<relative_path_without_ext>/
      - detect fs from first BBSamples block
      - choose nsnap = fs * SNAP_DUR_US (µs)
      - parse nav blocks while streaming
      - for each selected BBSamples, associate most recent nav data
      - save spectrograms + I/Q plots with full nav annotation.
    """

    rel = sbf_path.relative_to(root_dir)
    rel_no_ext = rel.with_suffix("")
    out_dir = out_root / rel_no_ext
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print(f"SBF file       : {sbf_path}")
    print(f"Output dir     : {out_dir}")
    print(f"Sampling every : {SAMPLE_PERIOD_SEC} s (UTC) PER LO/band")
    print(f"Max samples    : {MAX_SAMPLES_PER_FILE} per file (all bands)")
    print(f"Config T_snap  : {SNAP_DUR_US:.1f} µs (requested)")
    print("============================================================")

    parser = SbfParser()
    block_i = -1
    saved = 0

    file_fs: Optional[float] = None
    nsnap: Optional[int] = None

    next_sample_t_by_lo: Dict[float, Optional[datetime]] = {}

    last_pvt: Optional[dict] = None
    last_dop: Optional[dict] = None
    last_poscov: Optional[dict] = None
    last_meas: Optional[dict] = None

    printed_meas_keys = False
    printed_poscov_keys = False
    printed_pvt_keys = False
    printed_dop_keys = False

    with open(sbf_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break

            for blk, infos in parser.parse(chunk):
                # ---------- nav blocks ----------

                # PVTGeodetic / PVTGeodetic2
                if blk in ("PVTGeodetic", "PVTGeodetic2"):
                    if not printed_pvt_keys:
                        print(f"PVT-like block '{blk}' keys:", list(infos.keys()))
                        printed_pvt_keys = True

                    wnc_p, tow_p, _, _, _, utc_dt_p = extract_time_labels(infos)
                    last_pvt = {
                        "wnc": wnc_p,
                        "tow": tow_p,
                        "utc_dt": utc_dt_p,
                        "lat": infos.get("Lat"),
                        "lon": infos.get("Lon"),
                        "height": infos.get("Height")
                        or infos.get("Alt")
                        or infos.get("AltEll"),
                        "mode": infos.get("Mode"),
                        "error": infos.get("Error"),
                        "nr_sv": infos.get("NrSV"),
                    }
                    continue

                # PVTCartesian / PVTCartesian2: convert ECEF → geodetic
                if blk in ("PVTCartesian", "PVTCartesian2"):
                    if not printed_pvt_keys:
                        print(f"PVT-like block '{blk}' keys:", list(infos.keys()))
                        printed_pvt_keys = True

                    wnc_p, tow_p, _, _, _, utc_dt_p = extract_time_labels(infos)

                    X = infos.get("X")
                    Y = infos.get("Y")
                    Z = infos.get("Z")
                    lat = lon = h = None
                    if isinstance(X, (int, float)) and isinstance(Y, (int, float)) and isinstance(Z, (int, float)):
                        lat, lon, h = ecef_to_geodetic(float(X), float(Y), float(Z))

                    last_pvt = {
                        "wnc": wnc_p,
                        "tow": tow_p,
                        "utc_dt": utc_dt_p,
                        "lat": lat,
                        "lon": lon,
                        "height": h,
                        "mode": infos.get("Mode"),
                        "error": infos.get("Error"),
                        "nr_sv": infos.get("NrSV"),
                        # extra (not used in title but might be handy)
                        "X": X,
                        "Y": Y,
                        "Z": Z,
                    }
                    continue

                # DOP / DOP2
                if blk in ("DOP", "DOP2"):
                    if not printed_dop_keys:
                        print(f"DOP-like block '{blk}' keys:", list(infos.keys()))
                        printed_dop_keys = True

                    wnc_d, tow_d, *_ = extract_time_labels(infos)

                    def d(val):
                        return float(val) * DOP_SCALE if val is not None else None

                    hdop_raw = infos.get("HDOP")
                    vdop_raw = infos.get("VDOP")
                    pdop_val = infos.get("PDOP")
                    if pdop_val is None:
                        pdop_val = (
                            infos.get("PDOP3D") or infos.get("PDOP_3D")
                        )

                    last_dop = {
                        "wnc": wnc_d,
                        "tow": tow_d,
                        "hdop": d(hdop_raw),
                        "vdop": d(vdop_raw),
                        "pdop": d(pdop_val),
                    }
                    continue

                # PosCovGeodetic / PosCovGeodetic2
                if blk in ("PosCovGeodetic", "PosCovGeodetic2"):
                    if not printed_poscov_keys:
                        print(
                            "PosCov-like block '%s' keys:" % blk,
                            list(infos.keys()),
                        )
                        printed_poscov_keys = True

                    wnc_pc, tow_pc, *_ = extract_time_labels(infos)

                    cov_latlat = infos.get("Cov_latlat")
                    cov_lonlon = infos.get("Cov_lonlon")
                    cov_hgthgt = infos.get("Cov_hgthgt")

                    def s(val):
                        return (
                            float(np.sqrt(val))
                            if isinstance(val, (int, float)) and val >= 0
                            else None
                        )

                    last_poscov = {
                        "wnc": wnc_pc,
                        "tow": tow_pc,
                        "sE": s(cov_lonlon),  # approx σE
                        "sN": s(cov_latlat),  # approx σN
                        "sU": s(cov_hgthgt),  # σU
                    }
                    continue

                # MeasEpoch (for CN0)
                if blk == "MeasEpoch":
                    if not printed_meas_keys:
                        print(
                            "Meas-like block 'MeasEpoch' keys:",
                            list(infos.keys()),
                        )
                        printed_meas_keys = True

                    wnc_m, tow_m, *_ = extract_time_labels(infos)

                    # Keep full infos dict INCLUDING 'payload'
                    infos_copy = dict(infos)
                    infos_copy["wnc"] = wnc_m
                    infos_copy["tow"] = tow_m

                    last_meas = infos_copy
                    continue

                # ---------- BBSamples ----------
                if blk != "BBSamples":
                    continue
                block_i += 1

                x, fs_raw, flo = decode_bbsamples_iq(infos)
                if x is None or flo is None:
                    continue

                if file_fs is None:
                    file_fs = fs_raw
                    print(
                        f"Detected raw fs for {sbf_path.name}: "
                        f"{file_fs / 1e6:.6f} Msps"
                    )
                else:
                    if abs(fs_raw - file_fs) > 1e-3 * max(file_fs, 1.0):
                        print(
                            f"WARNING: fs changed within file {sbf_path.name}: "
                            f"{file_fs / 1e6:.6f} Msps -> {fs_raw / 1e6:.6f} Msps "
                            f"(block #{block_i})"
                        )

                if DECIM > 1:
                    x = x[::DECIM]
                    fs_eff = fs_raw / DECIM
                else:
                    fs_eff = fs_raw

                if nsnap is None:
                    snap_dur_sec = SNAP_DUR_US * 1e-6
                    nsnap = int(round(fs_eff * snap_dur_sec))
                    if nsnap < 8:
                        nsnap = 8
                    print(
                        f"For file {sbf_path.name}: "
                        f"effective fs={fs_eff / 1e6:.6f} Msps -> "
                        f"nsnap={nsnap} samples "
                        f"(T_snap≈{nsnap / fs_eff * 1e6:.1f} µs)"
                    )

                if nsnap > x.size:
                    if saved == 0:
                        print(
                            f"NOTE: requested nsnap={nsnap} > block size={x.size} "
                            f"for first snap; using full block length instead "
                            f"(T_snap≈{x.size / fs_eff * 1e6:.1f} µs)."
                        )
                    x_snap = x
                else:
                    x_snap = x[:nsnap]

                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = (
                    extract_time_labels(infos)
                )

                if flo not in next_sample_t_by_lo or next_sample_t_by_lo[
                    flo
                ] is None:
                    stride = max(1, int(SAMPLE_PERIOD_SEC))
                    floor = utc_dt.replace(
                        second=(utc_dt.second // stride) * stride,
                        microsecond=0,
                    )
                    if floor > utc_dt:
                        floor -= timedelta(seconds=stride)
                    next_sample_t_by_lo[flo] = floor + timedelta(
                        seconds=stride
                    )

                ns_time = next_sample_t_by_lo[flo]
                if ns_time is None:
                    ns_time = utc_dt
                    next_sample_t_by_lo[flo] = ns_time

                if utc_dt < ns_time:
                    continue
                while utc_dt >= ns_time + timedelta(
                    seconds=SAMPLE_PERIOD_SEC
                ):
                    ns_time += timedelta(seconds=SAMPLE_PERIOD_SEC)
                next_sample_t_by_lo[flo] = ns_time

                band_name = band_name_from_lo(flo)

                # ----- associate nav info (must be in the past) -----
                pvt_for_snap = None
                if (
                    last_pvt is not None
                    and last_pvt.get("wnc") == wnc
                ):
                    pvt_tow = last_pvt.get("tow")
                    if isinstance(pvt_tow, (int, float)):
                        dt = tow_s - pvt_tow
                        if 0.0 <= dt <= NAV_MAX_AGE_SEC:
                            pvt_for_snap = last_pvt

                dop_for_snap = None
                if (
                    last_dop is not None
                    and last_dop.get("wnc") == wnc
                ):
                    dop_tow = last_dop.get("tow")
                    if isinstance(dop_tow, (int, float)):
                        dt = tow_s - dop_tow
                        if 0.0 <= dt <= NAV_MAX_AGE_SEC:
                            dop_for_snap = last_dop

                posstd_for_snap = None
                if (
                    last_poscov is not None
                    and last_poscov.get("wnc") == wnc
                ):
                    pc_tow = last_poscov.get("tow")
                    if isinstance(pc_tow, (int, float)):
                        dt = tow_s - pc_tow
                        if 0.0 <= dt <= NAV_MAX_AGE_SEC:
                            posstd_for_snap = last_poscov

                meas_for_snap = None
                if (
                    last_meas is not None
                    and last_meas.get("wnc") == wnc
                ):
                    m_tow = last_meas.get("tow")
                    if isinstance(m_tow, (int, float)):
                        dt = tow_s - m_tow
                        if 0.0 <= dt <= NAV_MAX_AGE_SEC:
                            meas_for_snap = last_meas

                cn0_for_snap = cn0_info_for_band(meas_for_snap, flo)

                saved += 1
                out_path = plot_and_save(
                    sample_idx=saved,
                    block_idx=block_i,
                    x=x_snap,
                    fs=fs_eff,
                    flo=flo,
                    band_name=band_name,
                    wnc=wnc,
                    tow_s=tow_s,
                    tow_hms=tow_hms,
                    utc_hms=utc_hms,
                    utc_iso=utc_iso,
                    utc_dt=utc_dt,
                    out_dir=out_dir,
                    pvt_info=pvt_for_snap,
                    dop_info=dop_for_snap,
                    posstd_info=posstd_for_snap,
                    cn0_info=cn0_for_snap,
                )

                print(
                    f"[{saved}] {utc_iso} | {band_name} | "
                    f"fs={fs_eff / 1e6:.3f} Msps | "
                    f"T_snap≈{len(x_snap) / fs_eff * 1e6:.1f} µs | "
                    f"LO={flo / 1e6:.3f} MHz | "
                    f"PVT={'yes' if pvt_for_snap else 'no'} | "
                    f"DOP={'yes' if dop_for_snap else 'no'} | "
                    f"PosStd={'yes' if posstd_for_snap else 'no'} | "
                    f"CN0/CNR={'yes' if cn0_for_snap else 'no'} -> {out_path}"
                )

                next_sample_t_by_lo[flo] = ns_time + timedelta(
                    seconds=SAMPLE_PERIOD_SEC
                )

                if saved >= MAX_SAMPLES_PER_FILE:
                    print(
                        "Reached MAX_SAMPLES_PER_FILE; stopping this file."
                    )
                    return

    print(
        f"Done with {sbf_path.name}. Saved {saved} spectrograms (all bands)."
    )


# ======================= MAIN =======================

def main():
    sbf_dir = Path(SBF_DIR)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    if not sbf_dir.is_dir():
        raise RuntimeError(f"SBF_DIR is not a directory: {sbf_dir}")

    # Support both classic .sbf and Jammertest-style .25_ files
    sbf_files = sorted(list(sbf_dir.rglob("*.sbf")) +
                       list(sbf_dir.rglob("*.25_")))

    print(
        "====== SBF MULTI-FILE → SPECTROGRAMS "
        "(MULTI-BAND + PVT/DOP/PosStd/CNR) ======"
    )
    print(f"Input root  : {sbf_dir}")
    print(f"Output root : {out_root}")
    print(f"Found {len(sbf_files)} SBF-like files (*.sbf + *.25_) (including subfolders).\n")

    if not sbf_files:
        print("No SBF files found. Nothing to do.")
        return

    for i, sbf_path in enumerate(sbf_files, start=1):
        print(f"\n=== [{i}/{len(sbf_files)}] Processing file: {sbf_path} ===")
        process_single_sbf(sbf_path, out_root, sbf_dir)

    print("\nAll files processed. Done.")


if __name__ == "__main__":
    main()
