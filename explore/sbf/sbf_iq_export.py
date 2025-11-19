import os, sys, numpy as np
import argparse

try:
    from sbf_parser import load  # pip install sbf-parser
except ImportError:
    print("Install first:  pip install sbf-parser")
    sys.exit(1)

def as_u16_array(x):
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(x, dtype='<u2')
    return np.asarray(x, dtype=np.uint16)

def unpack_iq_int8(words_u16):
    I8 = ((words_u16 >> 8) & 0xFF).astype(np.int8)
    Q8 = (words_u16 & 0xFF).astype(np.int8)
    return I8, Q8

def export_streams(path, do_plot=False):
    base = os.path.splitext(os.path.basename(path))[0]

    # Accumulate per LO (Hz)
    streams = {}  # LO -> dict(Fs, I8_chunks, Q8_chunks, n)
    nblocks = 0

    with open(path, 'rb') as f:
        for block_type, info in load(f):
            if block_type != "BBSamples":
                continue
            Fs = int(info.get("SampleFreq", 0))
            LO = int(info.get("LOFreq", 0))
            samples = info.get("Samples", None)
            if Fs == 0 or samples is None:
                continue
            words = as_u16_array(samples)
            I8, Q8 = unpack_iq_int8(words)
            s = streams.setdefault(LO, {"Fs": Fs, "I8": [], "Q8": [], "n": 0})
            s["I8"].append(I8)
            s["Q8"].append(Q8)
            s["n"] += I8.size
            nblocks += 1

    if not streams:
        print("No BBSamples found in this SBF. (Then there is no IQ in this file.)")
        return

    print(f"Found {nblocks} BBSamples blocks across {len(streams)} LO groups.")
    for LO, s in streams.items():
        Fs = s["Fs"]
        I8 = np.concatenate(s["I8"]) if s["I8"] else np.empty(0, np.int8)
        Q8 = np.concatenate(s["Q8"]) if s["Q8"] else np.empty(0, np.int8)
        n = I8.size
        dur = n / Fs if Fs else 0.0

        # Write interleaved int8 IQ (.c8)
        iq_i8 = np.empty(n*2, dtype=np.int8)
        iq_i8[0::2] = I8
        iq_i8[1::2] = Q8
        out_c8 = f"{base}_LO{LO/1e6:.3f}MHz_Fs{Fs/1e6:.3f}MHz.c8"
        iq_i8.tofile(out_c8)

        # Write float32 complex (.fc32)
        iq_f = I8.astype(np.float32)/128.0 + 1j*(Q8.astype(np.float32)/128.0)
        out_fc32 = f"{base}_LO{LO/1e6:.3f}MHz_Fs{Fs/1e6:.3f}MHz.fc32"
        iq_f.astype(np.complex64).tofile(out_fc32)

        print(f"- LO={LO} Hz | Fs={Fs} Hz | samples={n} (~{dur:.1f}s) → {out_c8}  &  {out_fc32}")

        if do_plot:
            import matplotlib.pyplot as plt
            Nshow = min(2000, n)
            t = np.arange(Nshow) / Fs
            plt.figure(); plt.plot(t, iq_f.real[:Nshow], label="I"); plt.plot(t, iq_f.imag[:Nshow], label="Q", alpha=0.7)
            plt.xlabel("Time [s]"); plt.ylabel("Norm. amp"); plt.title(f"Time domain  LO={LO/1e6:.3f} MHz  Fs={Fs/1e6:.3f} MHz"); plt.grid(True); plt.legend()

            Nfft = 1
            while Nfft < min(262144, n): Nfft <<= 1
            x = iq_f[:Nfft] if n >= Nfft else iq_f
            X = np.fft.fftshift(np.fft.fft(x))
            f = np.fft.fftshift(np.fft.fftfreq(x.size, d=1.0/Fs))
            mag = 20*np.log10(np.maximum(np.abs(X), 1e-12))
            plt.figure(); plt.plot(f, mag)
            plt.xlabel("Frequency [Hz]"); plt.ylabel("Magnitude [dB]"); plt.title(f"Spectrum (N={x.size})  LO={LO/1e6:.3f} MHz"); plt.grid(True)
            plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract raw baseband IQ from Septentrio SBF (BBSamples).")
    ap.add_argument("sbf", help="path to .sbf file")
    ap.add_argument("--plot", action="store_true", help="show time/spectrum plots")
    args = ap.parse_args()
    export_streams(args.sbf, do_plot=args.plot)
