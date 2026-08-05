"""
check_artifact_calibration.py — check continuous_bad_mask() against known,
self-marked artifacts on REAL Muse hardware (the synthetic-signal test in
test_artifact_rejection.py validates the detector's logic in isolation;
this closes the gap to actual sensor noise).

Protocol (single headset, no partner needed):

  Terminal 1:  muselsl stream --address <MAC> --name Muse
  Terminal 2:  python stimulus_marker.py calibrate 20
  Terminal 3:  python record_single.py 180 Muse

  1. Start the recording, then sit still for the first ~20-30s (clean
     baseline — expect ~0% flagged here).
  2. For each deliberate artifact, press Enter in Terminal 2 the instant
     you do it, then wait ~2s before the next one so they don't overlap:
       - a few blinks
       - a jaw clench
       - a big head turn
       - pulling the headband / touching an electrode
     (all 20 markers don't need to be used -- press Ctrl+C in Terminal 2
     once you've covered the artifact types you care about)
  3. When the recording finishes, run this script on the resulting CSV:

       python check_artifact_calibration.py recordings/<stamp>_Muse.csv

Reads the `_markers.json` sidecar that record_single.py writes next to the
CSV (one entry per Enter-press in stimulus_marker.py), reports whether the
~1s window after each marker got flagged bad, reports the overall %
flagged, and saves a plot (traces + shaded bad regions + marker lines) so
you can eyeball whether the shading actually lines up with when you moved.

Usage:
  python check_artifact_calibration.py recordings/<stamp>_Muse.csv
  python check_artifact_calibration.py recordings/<stamp>_Muse.csv --check-window 1.5
  python check_artifact_calibration.py recordings/<stamp>_Muse.csv --artifact-threshold 400
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import CH_NAMES, continuous_bad_mask, load_csv_to_raw, preprocess


def mask_to_spans(mask, sfreq):
    """Boolean per-sample array -> list of (start_s, end_s) contiguous True runs."""
    spans = []
    in_span = False
    start = 0
    for i, bad in enumerate(mask):
        if bad and not in_span:
            start = i
            in_span = True
        elif not bad and in_span:
            spans.append((start / sfreq, i / sfreq))
            in_span = False
    if in_span:
        spans.append((start / sfreq, len(mask) / sfreq))
    return spans


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path")
    p.add_argument("--artifact-window", type=float, default=0.5)
    p.add_argument("--artifact-step", type=float, default=0.1)
    p.add_argument("--artifact-threshold", type=float, default=500.0)
    p.add_argument("--check-window", type=float, default=1.0,
                   help="seconds after each marker to check for overlap with "
                        "the bad mask (default 1.0 -- real artifacts like a "
                        "jaw clench outlast a single 500ms window)")
    p.add_argument("--out", default=None,
                   help="plot output path (default: <csv_path>_calibration.png)")
    args = p.parse_args()

    sidecar = args.csv_path.replace(".csv", "_markers.json")
    if not os.path.exists(sidecar):
        raise SystemExit(
            f"ERROR: no marker sidecar found at {sidecar}\n"
            "Run stimulus_marker.py alongside record_single.py and press "
            "Enter at each deliberate artifact so they get timestamped."
        )
    with open(sidecar) as f:
        markers = json.load(f)
    if not markers:
        raise SystemExit(f"ERROR: {sidecar} has no markers in it.")

    raw, fs = load_csv_to_raw(args.csv_path, "CAL")
    nyq = fs / 2
    raw_pp = preprocess(raw, h_freq=min(40.0, nyq * 0.95))
    mask = continuous_bad_mask(raw_pp, window_s=args.artifact_window,
                                step_s=args.artifact_step,
                                threshold_uv=args.artifact_threshold)

    print("=" * 60)
    print("ARTIFACT CALIBRATION CHECK")
    print("=" * 60)
    print(f"  window={args.artifact_window}s  step={args.artifact_step}s  "
          f"threshold={args.artifact_threshold} uV")
    print(f"  Overall: {100 * mask.mean():.1f}% of the recording flagged bad")
    print()
    print(f"  {'marker':<20} {'time (s)':>10}   flagged in +{args.check_window:.1f}s window")
    print("  " + "-" * 56)

    check_n = int(round(args.check_window * fs))
    for m in markers:
        t = float(m["rel_time_s"])
        start = int(round(t * fs))
        end = min(len(mask), start + check_n)
        if start >= len(mask):
            pct = float("nan")
        else:
            pct = 100 * mask[start:end].mean()
        # any non-trivial overlap counts as "caught" -- real artifacts are
        # often shorter than --check-window, so a strict majority-overlap
        # bar would mislabel a correctly-detected-but-brief artifact as missed
        flag = "FLAGGED" if pct > 10 else "not flagged"
        print(f"  {m['marker']:<20} {t:>10.2f}   {pct:5.1f}%  {flag}")

    out_path = args.out or args.csv_path.replace(".csv", "_calibration.png")
    data_uv = raw_pp.get_data() * 1e6
    t = np.arange(data_uv.shape[1]) / fs

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, ch in enumerate(CH_NAMES):
        ax.plot(t, data_uv[i] + i * 400, lw=0.5, label=ch)
    for start_s, end_s in mask_to_spans(mask, fs):
        ax.axvspan(start_s, end_s, color="red", alpha=0.15, lw=0)
    for m in markers:
        mt = float(m["rel_time_s"])
        ax.axvline(mt, color="black", ls="--", lw=0.8)
        ax.text(mt, ax.get_ylim()[1], m["marker"], rotation=90,
                fontsize=7, va="top", ha="right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("uV (offset per channel)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.set_title(f"Artifact calibration check — red = flagged bad ({os.path.basename(args.csv_path)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print()
    print(f"  Plot saved: {out_path}")
    print("  Check that the red shading lines up with the marker lines --")
    print("  flagged stretches with no nearby marker are false positives;")
    print("  markers with no red nearby are missed artifacts.")


if __name__ == "__main__":
    main()
