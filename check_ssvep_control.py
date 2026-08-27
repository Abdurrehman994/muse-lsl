"""
check_ssvep_control.py — pre-flight GO/NO-GO gate for a shared-flicker (SSVEP)
positive-control recording, BEFORE running the full pipeline.py.

Motivation
----------
A shared-flicker session is only a valid positive control if the flicker
actually drove a steady-state response. Several sessions in this project
looked structurally fine (both headsets, shared marker, 100% "clean") yet
contained NO detectable SSVEP at all -- so a null coupling result from them
would have been a null about the *recording*, not the pipeline's sensitivity.
This script catches that in seconds, so you don't spend a full pipeline run
(pool null over every recording, several minutes) discovering the input was
dead.

It also reports whether the recording is running hot (railing near the Muse's
+/-1000 uV limit) and recommends an --artifact-threshold, because the one
known-good SSVEP session in this project rails and needs the pipeline's
default 500 uV threshold loosened or 100% of its data gets rejected.

What it checks, per headset
---------------------------
  1. SSVEP drive: per-channel SNR at the flicker frequency and its 2nd
     harmonic (peak power in a narrow band vs flanking noise bands),
     computed on the stimulus-locked (post-onset) segment. TP9/TP10 are
     flagged as the occipital-nearest channels (best SSVEP pickup on Muse).
  2. Amplitude health: % of samples railed (>= 990 uV on the raw signal)
     and the distribution of 0.5s peak-to-peak amplitude AFTER a 1-40 Hz
     band-pass (matching pipeline.preprocess), used to recommend a
     threshold and predict the clean fraction the pipeline would keep.

Verdict
-------
  GO   -- best-channel SSVEP SNR >= --snr-pass on BOTH headsets. Prints the
          ready-to-run pipeline.py command (with a recommended threshold).
  WEAK -- SSVEP present but marginal on at least one headset.
  NO-GO- no detectable SSVEP on a headset: do not use as a positive control;
          re-check fixation / screen brightness / electrode contact / that
          the flicker clip actually played fullscreen at the right rate.

Usage
-----
  python check_ssvep_control.py recordings/<stamp>_D1_A1.csv recordings/<stamp>_D1_6F.csv
  python check_ssvep_control.py --session 20260811_113201      # auto-find the two CSVs
  python check_ssvep_control.py a.csv b.csv --stim-hz 8.0      # non-default flicker rate
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, detrend

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
OCCIPITAL = {"TP9", "TP10"}   # Muse's rear electrodes -- closest to visual cortex
RAIL_UV = 990.0               # matches pipeline.detect_bad_channels
PIPELINE_DEFAULT_THRESHOLD = 500.0


def find_session_csvs(session):
    """Given a stamp like 20260811_113201, return its two device CSVs."""
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
    hits = sorted(glob.glob(os.path.join(rec, f"{session}_*.csv")))
    hits = [h for h in hits if not h.endswith("_calibration.png")]
    if len(hits) < 2:
        raise SystemExit(f"ERROR: expected 2 CSVs for session {session}, found {len(hits)}: {hits}")
    return hits[:2]


def load_stimulus_onset(csv_path):
    """rel_time_s of the first stimulus marker in the _markers.json sidecar, or None."""
    sidecar = csv_path.replace(".csv", "_markers.json")
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        markers = json.load(f)
    for m in markers:
        if "stimulus" in m.get("marker", "").lower():
            return float(m["rel_time_s"])
    return float(markers[0]["rel_time_s"]) if markers else None


def load_post_onset(csv_path):
    """Return (df_post_onset, fs, onset_s). Trims to the stimulus marker if present."""
    df = pd.read_csv(csv_path)
    tcol = "time_s" if "time_s" in df.columns else "lsl_timestamp"
    missing = [c for c in [tcol] + CH_NAMES if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {csv_path} missing columns: {missing}")
    ts = df[tcol].values
    fs = 1.0 / float(np.median(np.diff(ts)))
    onset = load_stimulus_onset(csv_path)
    if onset is not None:
        df = df[ts >= onset].reset_index(drop=True)
    return df, fs, onset


def snr_at(f, freqs, pxx, bw=0.5, noise=(1.0, 3.0)):
    """Peak power in [f-bw, f+bw] over mean power in the flanking noise bands."""
    sig = pxx[(freqs >= f - bw) & (freqs <= f + bw)].max()
    lo = pxx[(freqs >= f - noise[1]) & (freqs <= f - noise[0])]
    hi = pxx[(freqs >= f + noise[0]) & (freqs <= f + noise[1])]
    flank = np.concatenate([lo, hi])
    return float(sig / flank.mean()) if flank.size else float("nan")


def bandpass(x, fs, lo=1.0, hi=40.0):
    """1-40 Hz zero-phase band-pass, matching pipeline.preprocess."""
    hi = min(hi, fs / 2 * 0.95)
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def window_ptp_max(data_uv, fs, window_s=0.5, step_s=0.1):
    """Per-window max-over-channels peak-to-peak (uV), mirroring the pipeline's
    sliding-window artifact detector (a window is rejected if ANY channel's
    ptp exceeds the threshold)."""
    n = data_uv.shape[1]
    win = max(1, int(round(window_s * fs)))
    step = max(1, int(round(step_s * fs)))
    out = []
    for start in range(0, max(1, n - win + 1), step):
        seg = data_uv[:, start:start + win]
        out.append(float((seg.max(axis=1) - seg.min(axis=1)).max()))
    return np.array(out)


def clean_fraction_at(win_ptp, threshold):
    """Fraction of sliding windows that would survive a given threshold."""
    return float((win_ptp <= threshold).mean())


def analyze_headset(csv_path, stim_hz, snr_pass):
    df, fs, onset = load_post_onset(csv_path)
    dur = len(df) / fs
    clean_frac = float((~df["is_gap"].astype(bool)).mean()) if "is_gap" in df else 1.0
    name = os.path.basename(csv_path)

    print(f"\n=== {name} ===")
    onset_str = f"{onset:.2f}s" if onset is not None else "NONE (no marker!)"
    print(f"  fs={fs:.0f}Hz  post-onset={dur:.0f}s  onset={onset_str}  gap-clean={clean_frac:.1%}")
    if onset is None:
        print("  WARNING: no stimulus marker -- analyzing whole recording, not stimulus-locked.")

    raw_uv = df[CH_NAMES].values.T.astype(float)      # rail check on raw
    filt_uv = np.vstack([bandpass(detrend(x), fs) for x in raw_uv])  # SNR + ptp on filtered

    # --- SSVEP drive per channel ---
    print(f"  SSVEP @ {stim_hz:g}Hz (fundamental) / {2*stim_hz:g}Hz (harmonic):")
    best_snr = 0.0
    best_ch = None
    for ch, x in zip(CH_NAMES, filt_uv):
        freqs, pxx = welch(x, fs=fs, nperseg=int(fs * 4))
        s1 = snr_at(stim_hz, freqs, pxx)
        s2 = snr_at(2 * stim_hz, freqs, pxx)
        occ = "  <- occipital" if ch in OCCIPITAL else ""
        railed = float((np.abs(raw_uv[CH_NAMES.index(ch)]) >= RAIL_UV).mean())
        rail_str = f"  railed={railed:.0%}" if railed > 0.005 else ""
        print(f"    {ch:5s} SNR={s1:6.1f} (harm {s2:5.1f}){rail_str}{occ}")
        if s1 > best_snr:
            best_snr, best_ch = s1, ch

    # --- amplitude / threshold recommendation ---
    win_ptp = window_ptp_max(filt_uv, fs)
    p95, p99 = np.percentile(win_ptp, [95, 99])
    # smallest round threshold keeping >=90% of windows, floored at the pipeline default
    rec = max(PIPELINE_DEFAULT_THRESHOLD, np.ceil(p95 / 250.0) * 250.0)
    at_default = clean_fraction_at(win_ptp, PIPELINE_DEFAULT_THRESHOLD)
    at_rec = clean_fraction_at(win_ptp, rec)
    print(f"  amplitude: 0.5s-window ptp p95={p95:.0f}uV p99={p99:.0f}uV")
    print(f"    clean@{PIPELINE_DEFAULT_THRESHOLD:.0f}uV(default)={at_default:.0%}"
          f"   clean@{rec:.0f}uV(rec)={at_rec:.0%}")

    if best_snr >= snr_pass:
        verdict = "GO"
    elif best_snr >= 2.0:
        verdict = "WEAK"
    else:
        verdict = "NO-GO"
    print(f"  --> {verdict}  (best SSVEP SNR={best_snr:.1f} on {best_ch})")
    return {"verdict": verdict, "best_snr": best_snr, "rec_threshold": rec,
            "clean_at_default": at_default}


def main():
    p = argparse.ArgumentParser(
        description="GO/NO-GO gate for a shared-flicker SSVEP positive-control recording.")
    p.add_argument("csv", nargs="*", help="the two device CSVs (or use --session)")
    p.add_argument("--session", help="session stamp, e.g. 20260811_113201; auto-finds both CSVs")
    p.add_argument("--stim-hz", type=float, default=6.0, help="flicker reversal rate (default 6)")
    p.add_argument("--snr-pass", type=float, default=3.0,
                   help="min best-channel SSVEP SNR for a GO (default 3.0)")
    args = p.parse_args()

    if args.session:
        csvs = find_session_csvs(args.session)
    elif len(args.csv) >= 2:
        csvs = args.csv[:2]
    else:
        p.error("give two CSV paths or --session <stamp>")

    print(f"Pre-flight SSVEP check  (flicker={args.stim_hz:g}Hz, GO if best SNR>={args.snr_pass:g})")
    results = [analyze_headset(c, args.stim_hz, args.snr_pass) for c in csvs]

    verdicts = [r["verdict"] for r in results]
    print("\n" + "=" * 60)
    if all(v == "GO" for v in verdicts):
        thr = max(r["rec_threshold"] for r in results)
        overall = "GO"
        note = f"valid positive control. Run the pipeline with --artifact-threshold {thr:.0f}:"
        cmd = (f"    python pipeline.py {csvs[0]} {csvs[1]} \\\n"
               f"           --stim-hz {args.stim_hz:g} --pool-dir recordings/ "
               f"--artifact-threshold {thr:.0f}")
    elif any(v == "NO-GO" for v in verdicts):
        overall = "NO-GO"
        note = ("NO detectable SSVEP on at least one headset -- NOT a valid positive control.\n"
                "  Do NOT run the pipeline on this as a sensitivity test. Re-check: subject "
                "fixating the\n  flicker? screen bright/close enough? TP9/TP10 contact? clip "
                "actually fullscreen at the right rate?")
        cmd = None
    else:
        overall = "WEAK"
        note = ("SSVEP marginal -- usable but not a strong control. Consider re-recording with "
                "better fixation/contact.")
        cmd = None

    print(f"OVERALL: {overall} -- {note}")
    if cmd:
        print(cmd)
    print("=" * 60)
    sys.exit(0 if overall == "GO" else 1)


if __name__ == "__main__":
    main()
