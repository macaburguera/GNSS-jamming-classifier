# sbf_session_stats.py
import sys, numpy as np
from sbf_parser import load

GPS_WEEK_SEC = 604800.0

def gpstime_secs(wnc, tow):
    # SBF TOW is usually in milliseconds; handle both ms/s defensively.
    tow = float(tow)
    if tow > 1e6:  # clearly ms-scale
        tow = tow / 1000.0
    return wnc * GPS_WEEK_SEC + tow

def main(path):
    t_first = None
    t_last  = None
    per_lo = {}  # LO -> dict(Fs, samples, t_first, t_last, blocks)

    with open(path, 'rb') as f:
        for blk, info in load(f):
            if blk != "BBSamples":
                continue
            Fs = int(info.get("SampleFreq", 0))
            LO = int(info.get("LOFreq", 0))
            N  = int(info.get("N", 0))  # samples in this burst
            WNc = int(info.get("WNc", info.get("WN", 0)))
            TOW = info.get("TOW", 0)
            t = gpstime_secs(WNc, TOW)

            if t_first is None or t < t_first: t_first = t
            if t_last  is None or t > t_last:  t_last  = t

            s = per_lo.setdefault(LO, {"Fs": Fs, "samples": 0, "t0": None, "t1": None, "blocks": 0})
            s["samples"] += N
            s["blocks"]  += 1
            s["t0"] = t if s["t0"] is None or t < s["t0"] else s["t0"]
            s["t1"] = t if s["t1"] is None or t > s["t1"] else s["t1"]

    if t_first is None:
        print("No BBSamples found.")
        return

    sess = t_last - t_first
    print(f"Session duration (BBSamples span): {sess:.1f} s  (~{sess/3600:.2f} h)")
    for LO, s in sorted(per_lo.items()):
        Fs = s["Fs"]
        agg_sec = s["samples"] / Fs if Fs else 0.0
        span_lo = (s["t1"] - s["t0"]) if (s["t0"] and s["t1"]) else 0.0
        duty = (agg_sec / span_lo * 100.0) if span_lo > 0 else 0.0
        print(f"  LO={LO} Hz | Fs={Fs} Hz | blocks={s['blocks']:,} | "
              f"aggregate IQ={agg_sec:.3f} s | LO span={span_lo:.1f} s | duty≈{duty:.4f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sbf_session_stats.py <file.sbf>")
        sys.exit(1)
    main(sys.argv[1])
