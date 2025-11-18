# sbf_iq_perblock_loglabel_standalone.py
# Save one spectrogram every 30 s from a Septentrio SBF, annotate with jammer label
# parsed from a plain-text logbook in LOCAL time. No CLI params—edit variables below.

from pathlib import Path
import re
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from sbf_parser import SbfParser
from datetime import datetime, timedelta, timezone

# ---------------------- USER VARIABLES ----------------------
SBF_PATH = r"D:\datasets\Jammertest2023_Day1\Altus04 - 50m\alt04024.sbf"
OUT_DIR = r"D:\datasets\Jammertest2023_Day1\plots\out_specs_alt04024_labeled_30s"
LOGBOOK_PATH = r"D:\datasets\Jammertest2023_Day1\Testlog 23.09.18.txt"   # the plain-text block you pasted
LOCAL_DATE = "2023-09-18"              # date of the test in LOCAL time (YYYY-MM-DD) ← set this
LOCAL_UTC_OFFSET = 2.0                 # LOCAL minus UTC (e.g., 2 for CEST)
SAVE_EVERY_SEC = 30.0                  # save at :00, :30, :00, ...
DECIM = 1                             # decimate complex IQ by this factor before STFT
NPERSEG = 64                          # STFT window (tuned for N≈500 after decim=4)
NOVERLAP = 56                          # STFT overlap
REMOVE_DC = True                       # subtract mean (reduces center stripe)
VMIN_DB = -80                          # fix spectrogram color scale (set None to auto)
VMAX_DB = -20
CHUNK_BYTES = 1_000_000                # read size for streaming the SBF
MAX_IMAGES = None                      # cap number of saved images; None = no cap
# ------------------------------------------------------------

EPS = 1e-12
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (stable since 2017-01-01)

Interval = Tuple[datetime, datetime, str]  # (UTC start, UTC end, label)

# ---------------------- time helpers -----------------------
def gps_week_tow_to_utc(wn: int, tow_s: float) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=GPS_MINUS_UTC)

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600)
    m = int((tsec % 3600) // 60)
    s = tsec - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def extract_time_labels(infos):
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0))
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw  # TOW often ms in BBSamples
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

# ---------------------- SBF decode -------------------------
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

# ---------------------- logbook parsing --------------------
def parse_plaintext_logbook(path: str, local_date: str, local_utc_offset_hours: float) -> List[Interval]:
    """
    Parses lines like:
      '16:00 - Test was started - no jamming'
      '16:05 - Jammer u1.1 was turned on'
      '16:25 - Jammer was turned off but then on again - some confusion about what jammer to use.'
      '17:00 - Jammer turned off'
    Returns UTC intervals with labels: 'NO JAM', 'u1.1', 's1.2', 'UNKNOWN/CONFUSION', etc.
    """
    # Load all (time, raw_text)
    time_re = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*-\s*(.+)$')
    events = []  # list of (local_dt, label_for_next_interval)
    base = datetime.strptime(local_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    offs = timedelta(hours=float(local_utc_offset_hours))

    def label_from_text(txt: str) -> str:
        t = txt.strip().lower()
        if "no jamming" in t or "turned off" in t and "confusion" not in t:
            return "NO JAM"
        if "confusion" in t:
            return "UNKNOWN/CONFUSION"
        # turned on → try to extract jammer code
        m = re.search(r"jammer\s+(.+?)\s+was turned on", txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"jammer\s+(.+?)\s+turned on", txt, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return "UNKNOWN"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line)
            if not m:
                continue
            hh, mm, rest = m.groups()
            hh = int(hh); mm = int(mm)
            local_dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            lbl = label_from_text(rest)
            # store as the label active FROM this time onward
            events.append((local_dt, lbl))

    # Sort & build intervals (local → UTC)
    events.sort(key=lambda e: e[0])
    intervals: List[Interval] = []
    for i in range(len(events)):
        start_local, lbl = events[i]
        end_local = events[i+1][0] if i+1 < len(events) else (start_local + timedelta(minutes=60))
        intervals.append((start_local - offs, end_local - offs, lbl))
    return intervals

def label_for_time(intervals: List[Interval], t_utc: datetime) -> Optional[str]:
    for a,b,label in intervals:
        if a <= t_utc < b:
            return label
    return None

# ---------------------- plotting --------------------------
def plot_and_save(block_idx, x, fs, wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt,
                  label, out_dir, nperseg=128, noverlap=96, dpi=120,
                  remove_dc=False, vmin=None, vmax=None):
    N = x.size
    if N < 8:
        return None
    if remove_dc:
        x = x - np.mean(x)

    nperseg_eff = min(int(nperseg), N)
    noverlap_eff = min(int(noverlap), max(0, nperseg_eff - 1))

    f, t, Z = stft(x, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                   return_onesided=False, boundary=None, padded=False)
    if t.size < 2:
        nperseg_eff = max(32, min(N//4, nperseg_eff))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = stft(x, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff,
                       return_onesided=False, boundary=None, padded=False)

    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    tt = np.arange(N, dtype=np.float32) / fs
    I = np.real(x); Q = np.imag(x)

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

    jam_txt = f" | Jammer: {label}" if label else ""
    title = (f"Spectrogram (BBSamples #{block_idx})  |  GPS week {wnc}  |  "
             f"TOW {tow_s:.3f}s ({tow_hms})  |  UTC {utc_hms}{jam_txt}\n"
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
    fname_label = (label.replace(" ", "_").replace("/", "-") if label else "nolabel")
    out_path = Path(out_dir) / f"spec_{utc_dt.strftime('%H%M%S')}_{fname_label}_blk{block_idx:06d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------- main flow --------------------------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load intervals from the plain-text logbook
    intervals = parse_plaintext_logbook(LOGBOOK_PATH, LOCAL_DATE, LOCAL_UTC_OFFSET)
    print(f"Loaded {len(intervals)} intervals from logbook.")
    for a,b,lbl in intervals:
        print(f"  {a.strftime('%H:%M:%S')}Z → {b.strftime('%H:%M:%S')}Z : {lbl}")

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
                    x = x[::DECIM]
                    fs = fs / DECIM

                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)

                # Gate: save the first block at/after each :00/:30 boundary (UTC)
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

                label = label_for_time(intervals, utc_dt)

                out_path = plot_and_save(
                    block_idx=block_i, x=x, fs=fs,
                    wnc=wnc, tow_s=tow_s, tow_hms=tow_hms,
                    utc_hms=utc_hms, utc_iso=utc_iso, utc_dt=utc_dt,
                    label=label, out_dir=OUT_DIR,
                    nperseg=NPERSEG, noverlap=NOVERLAP,
                    remove_dc=REMOVE_DC, vmin=VMIN_DB, vmax=VMAX_DB
                )
                saved += 1
                print(f"[{saved}] {out_path}")
                next_save_t = next_save_t + timedelta(seconds=SAVE_EVERY_SEC)

                if MAX_IMAGES is not None and saved >= MAX_IMAGES:
                    return

if __name__ == "__main__":
    main()
