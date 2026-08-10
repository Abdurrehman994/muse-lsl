"""
analyze_pattern_segments.py -- per-segment inter-brain PLV breakdown for a
combined multi-pattern stimulus video (see combine_clips.py), e.g.
stimuli/checker_combined.mp4 + stimuli/checker_combined_segments.json.

checker_combined.mp4 concatenates several checkerboard PATTERNS (grid,
radial, rings, wedges, stripes) all at the same reversal frequency. Rather
than betting the whole positive-control recording on one pattern, this
script slices ONE recording by the segments sidecar and reports inter-brain
PLV/circ-r for each pattern separately -- so you can see which pattern (if
any) gives the strongest, most reliable SSVEP-driven positive control on
Muse's frontal/temporal montage before standardizing on one for future
sessions.

Reuses the exact same validated building blocks as pipeline.py's default
continuous path (bad-channel exclusion, continuous artifact masking,
adjusted circular correlation) -- this is NOT a separate implementation,
just those functions applied within each segment's time window instead of
the whole recording.

Scope note: this reports DESCRIPTIVE per-segment PLV/circ-r to compare
patterns, not per-segment significance testing (each 30-60s segment is too
short for a well-powered surrogate/pseudo-pair null on its own -- see
combine_clips.py's docstring). For a statistically validated positive-
control verdict, run the WHOLE recording through pipeline.py with
--stim-hz/--pool-dir as usual; use this script to decide which pattern to
standardize on, not as the significance test itself.

Usage:
  python analyze_pattern_segments.py A.csv B.csv stimuli/checker_combined_segments.json
  python analyze_pattern_segments.py A.csv B.csv stimuli/checker_combined_segments.json --stim-hz 6.0
"""
import argparse
import json

import numpy as np

from pipeline import (
    load_stimulus_onset, load_csv_to_raw, preprocess, continuous_bad_mask,
    prefilter_raw_for_band, analytic_signal, plv_masked, circ_corr_adjusted_masked,
)


def load_segments(path):
    with open(path) as f:
        return json.load(f)


def window_mask(n_samples, fs, start_s, end_s):
    mask = np.zeros(n_samples, dtype=bool)
    start = max(0, int(round(start_s * fs)))
    end = min(n_samples, int(round(end_s * fs)))
    if end > start:
        mask[start:end] = True
    return mask


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    p.add_argument("segments_json",
                    help="sidecar written by combine_clips.py, e.g. "
                         "stimuli/checker_combined_segments.json")
    p.add_argument("--stim-hz", type=float, default=6.0,
                    help="reversal frequency shared by every segment (default 6.0)")
    p.add_argument("--stim-bandwidth", type=float, default=1.0,
                    help="full width in Hz of the band around --stim-hz (default 1.0)")
    p.add_argument("--artifact-window", type=float, default=0.5)
    p.add_argument("--artifact-step", type=float, default=0.1)
    p.add_argument("--artifact-threshold", type=float, default=500.0)
    p.add_argument("--artifact-pad", type=float, default=0.3)
    p.add_argument("--ica", action="store_true")
    args = p.parse_args()

    segments = load_segments(args.segments_json)
    band = (args.stim_hz - args.stim_bandwidth / 2, args.stim_hz + args.stim_bandwidth / 2)

    print("=" * 70)
    print(f"Per-segment breakdown  ({args.stim_hz:g} Hz +/- {args.stim_bandwidth/2:g} Hz)")
    print("=" * 70)
    onset_a = load_stimulus_onset(args.csv_a)
    onset_b = load_stimulus_onset(args.csv_b)
    if onset_a is None or onset_b is None:
        raise SystemExit(
            "ERROR: both recordings need a stimulus_start marker for segment "
            "alignment (see play_stimulus.py) -- at least one is missing.")

    raw_a, fs_a = load_csv_to_raw(args.csv_a, "A", onset_s=onset_a)
    raw_b, fs_b = load_csv_to_raw(args.csv_b, "B", onset_s=onset_b)
    if fs_a != fs_b:
        raise SystemExit(f"ERROR: sampling rates differ ({fs_a} vs {fs_b})")

    nyq = fs_a / 2
    h_freq = min(40.0, nyq * 0.95)
    raw_a_pp = preprocess(raw_a, h_freq=h_freq, use_ica=args.ica, subject_label="A")
    raw_b_pp = preprocess(raw_b, h_freq=h_freq, use_ica=args.ica, subject_label="B")

    bad_a = continuous_bad_mask(raw_a_pp, window_s=args.artifact_window,
                                 step_s=args.artifact_step,
                                 threshold_uv=args.artifact_threshold,
                                 pad_s=args.artifact_pad)
    bad_b = continuous_bad_mask(raw_b_pp, window_s=args.artifact_window,
                                 step_s=args.artifact_step,
                                 threshold_uv=args.artifact_threshold,
                                 pad_s=args.artifact_pad)
    n_common = min(len(bad_a), len(bad_b))
    good_mask = ~(bad_a[:n_common] | bad_b[:n_common])
    print(f"whole-recording jointly clean: {good_mask.sum()/fs_a:.1f}s / "
          f"{n_common/fs_a:.1f}s ({100*good_mask.mean():.1f}%)\n")

    raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
    raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
    analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
    analytic_b = analytic_signal(raw_b_band, n_samples=n_common)

    header = f"{'segment':<22}{'window':<16}{'usable':>8}{'PLV':>8}{'max':>8}{'circ-r':>9}"
    print(header)
    print("-" * len(header))
    results = []
    for seg in segments:
        seg_win = window_mask(n_common, fs_a, seg["start_s"], seg["end_s"])
        seg_good = seg_win & good_mask
        usable_s = seg_good.sum() / fs_a
        if seg_good.sum() < int(fs_a * 1.0):  # need at least ~1s of clean data
            print(f"{seg['name']:<22}{seg['start_s']:>5.0f}-{seg['end_s']:<9.0f}"
                  f"{usable_s:>7.1f}s   -- too little clean data --")
            continue
        plv = plv_masked(analytic_a, analytic_b, seg_good)
        cc = circ_corr_adjusted_masked(analytic_a, analytic_b, seg_good)
        results.append((seg["name"], plv.mean(), cc.mean()))
        print(f"{seg['name']:<22}{seg['start_s']:>5.0f}-{seg['end_s']:<9.0f}"
              f"{usable_s:>7.1f}s{plv.mean():>8.3f}{plv.max():>8.3f}{cc.mean():>9.3f}")

    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\nStrongest pattern by mean PLV: {best[0]}  (PLV={best[1]:.3f})")
        print("Reminder: this is descriptive (pattern comparison), not a "
              "significance test -- validate the winning pattern's whole-"
              "recording result with pipeline.py --stim-hz --pool-dir.")


if __name__ == "__main__":
    main()
