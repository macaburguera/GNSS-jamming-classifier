# sbf_iq_perblock_30s.py
# One spectrogram per 30-second UTC slot from a Septentrio SBF file (IQ together).

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from sbf_parser import SbfParser
from datetime import datetime, timedelta, timezone

EPS = 1e-12
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC = 18.0  # seconds (stable since 2017-01-01)

# -------------------- time helpers --------------------

def seconds_to_hms(tsec: float) -> str:
    tsec = float(tsec) % 86400.0
    h = int(tsec // 3600); m = int((tsec % 3600) // 60)
    s = tsec - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def gps_week_tow_to_utc(wn: int, tow_s: float, gps_minus_utc: float = GPS_MINUS_UTC) -> datetime:
    dt_gps = GPS_EPOCH + timedelta(weeks=int(wn), seconds=float(tow_s))
    return dt_gps - timedelta(seconds=gps_minus_utc)

def extract_time_labels(infos):
    wnc = int(infos.get("WNc", -1))
    tow_raw = float(infos.get("TOW", 0))
    tow_s = tow_raw / 1000.0 if tow_raw > 604800.0 else tow_raw  # ms vs s
    tow_hms = seconds_to_hms(tow_s)
    utc_dt = gps_week_tow_to_utc(wnc, tow_s)
    utc_hms = utc_dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.sss
    utc_iso = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    return wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt

# -------------------- data decode --------------------

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

# -------------------- plotting --------------------

def plot_and_save(block_idx, x, fs, wnc, tow_s, tow_hms, utc_hms, utc_iso,
                  out_dir, nperseg=128, noverlap=96, dpi=120, show=False):
    N = x.size
    if N < 8:
        return None

    nperseg_eff = min(int(nperseg), N)
    noverlap_eff = min(int(noverlap), max(0, nperseg_eff - 1))

    def do_stft(nw, ov):
        f, t, Z = stft(x, fs=fs, window="hann",
                       nperseg=nw, noverlap=ov,
                       return_onesided=False, boundary=None, padded=False)
        return f, t, Z

    f, t, Z = do_stft(nperseg_eff, noverlap_eff)
    if t.size < 2:
        nperseg_eff = max(32, min(N // 4, nperseg_eff, 256))
        noverlap_eff = int(0.9 * nperseg_eff)
        f, t, Z = do_stft(nperseg_eff, noverlap_eff)

    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S_dB = 20.0 * np.log10(np.abs(Z) + EPS)

    tt = np.arange(N, dtype=np.float32) / fs
    I = np.real(x); Q = np.imag(x)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2, 1, 1)

    if t.size >= 2:
        pcm = ax1.pcolormesh(t, f, S_dB, shading="auto")
        fig.colorbar(pcm, ax=ax1, label="dB")
    else:
        t0, t1 = 0.0, max(1.0 / fs, nperseg_eff / fs)
        im = ax1.imshow(S_dB, aspect="auto", origin="lower",
                        extent=[t0, t1, f[0], f[-1]])
        fig.colorbar(im, ax=ax1, label="dB")

    title = (f"Spectrogram (BBSamples #{block_idx})  |  GPS week {wnc}  |  "
             f"TOW {tow_s:.3f}s ({tow_hms})  |  UTC {utc_hms}\n"
             f"nperseg={nperseg_eff}, noverlap={noverlap_eff}")
    ax1.set_title(title)
    ax1.set_ylabel("Frequency [Hz]")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(tt, I, linewidth=0.7, label="I")
    ax2.plot(tt, Q, linewidth=0.7, alpha=0.85, label="Q")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.legend(loc="upper right")
    ax2.text(0.01, 0.02, utc_iso, transform=ax2.transAxes, fontsize=8,
             ha="left", va="bottom")

    fig.tight_layout()
    # filename includes TOW in ms and UTC wall-clock (safe for Windows)
    tow_ms = int(round(tow_s * 1000.0))
    out_name = f"spec_block{block_idx:06d}_tow{tow_ms:010d}.png"
    out_path = Path(out_dir) / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="One spectrogram per 30-second UTC slot (IQ together).")
    ap.add_argument("--sbf_path", default=r"C:\Users\macab\OneDrive - Danmarks Tekniske Universitet\Geopositioning and Navigation - Jammertest 2023\23.09.19 - Jammertest 2023 - Day 2\alt02 - reference during kraken test at location 2 for smartphone comparison.sbf")
    ap.add_argument("--out-dir", default="out_spectrograms")
    ap.add_argument("--max-blocks", type=int, default=100, help="Max figures to save")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N BBSamples blocks")
    ap.add_argument("--stride", type=int, default=1, help="Take every k-th BBSamples block")
    ap.add_argument("--decim", type=int, default=1, help="Integer decimation before STFT")
    ap.add_argument("--nperseg", type=int, default=128)
    ap.add_argument("--noverlap", type=int, default=96)
    ap.add_argument("--slot-secs", type=int, default=30, help="Slot length in seconds (default 30)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--chunk-bytes", type=int, default=65536)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    parser = SbfParser()
    block_i = -1
    saved = 0
    last_slot_idx = None  # integer floor(utc_timestamp / slot_secs)

    with open(args.sbf_path, "rb") as f:
        while True:
            chunk = f.read(args.chunk_bytes)
            if not chunk:
                break
            for blk, infos in parser.parse(chunk):
                if blk != "BBSamples":
                    continue
                block_i += 1
                if block_i < args.skip or (block_i - args.skip) % args.stride != 0:
                    continue

                x, fs = decode_bbsamples_iq(infos)
                if x is None:
                    continue
                if args.decim > 1:
                    x = x[::args.decim]
                    fs = fs / args.decim

                # times
                wnc, tow_s, tow_hms, utc_hms, utc_iso, utc_dt = extract_time_labels(infos)

                # ---- 30-second gating (aligned to :00 and :30) ----
                slot_idx = int(utc_dt.timestamp() // args.slot_secs)
                if last_slot_idx is not None and slot_idx == last_slot_idx:
                    continue  # already saved a figure for this slot
                last_slot_idx = slot_idx

                out_path = plot_and_save(
                    block_idx=block_i, x=x, fs=fs,
                    wnc=wnc, tow_s=tow_s, tow_hms=tow_hms,
                    utc_hms=utc_hms, utc_iso=utc_iso,
                    out_dir=args.out_dir,
                    nperseg=args.nperseg, noverlap=args.noverlap,
                    dpi=args.dpi, show=args.show
                )
                if out_path:
                    saved += 1
                    print(f"[{saved}] {out_path}")
                if saved >= args.max_blocks:
                    return

if __name__ == "__main__":
    main()
