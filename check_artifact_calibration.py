"""
check_artifact_calibration.py — check the artifact detectors against known,
self-marked artifacts on REAL Muse hardware.

The synthetic tests (test_artifact_rejection.py, test_ocular_detection.py)
validate detector logic in isolation against injected ground truth. This
script closes the gap to actual sensor noise, and — since the detectors are
no longer a single fixed threshold — tells you WHICH detector catches WHICH
kind of artifact on a real head.

Protocol (single headset, no partner needed):

  Terminal 1:  muselsl stream --address <MAC> --name Muse
  Terminal 2:  python stimulus_marker.py calibrate 20
  Terminal 3:  python record_single.py 180 Muse

  1. Start the recording, then sit still for the first ~20-30s (clean
     baseline — expect ~0% flagged here).
  2. For each deliberate artifact, press Enter in Terminal 2 the instant
     you do it, then wait ~2s before the next one so they don't overlap:
       - a few blinks              -> should fire the BLINK VELOCITY detector
       - a few hard left/right eye movements
                                   -> should fire the SACCADE channel
       - a jaw clench              -> should fire peak-to-peak, NOT blink
                                      velocity (it is too long to pass the
                                      duration test)
       - a big head turn           -> peak-to-peak
       - pulling the headband      -> peak-to-peak
     Doing them in that order makes the table below easy to read. Note in
     Terminal 2's order which marker was which; the sidecar only records
     timing, not what you did.
  3. When the recording finishes, run this script on the resulting CSV:

       python check_artifact_calibration.py recordings/<stamp>_Muse.csv

What to look for. A detector that fires on everything is as useless as one
that fires on nothing, so read the per-marker table against the "% of
recording flagged" line: a detector flagging 30% of the recording will
"catch" most markers by chance. The blink-velocity detector should be
selective — catching blinks and skipping the jaw clench — and its blink
count should work out to a plausible 10-20 blinks/minute.

Usage:
  python check_artifact_calibration.py recordings/<stamp>_Muse.csv
  python check_artifact_calibration.py rec.csv --reference mastoid
  python check_artifact_calibration.py rec.csv --ocular-k 3
  python check_artifact_calibration.py rec.csv --check-window 1.5
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline as P


def mask_to_spans(mask, sfreq):
    """Boolean per-sample array -> list of (start_s, end_s) contiguous True runs."""
    return [(s / sfreq, e / sfreq) for s, e in P.mask_runs(mask)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path")
    p.add_argument("--artifact-window", type=float, default=0.5)
    p.add_argument("--artifact-step", type=float, default=0.1)
    p.add_argument("--artifact-threshold", type=float, default=500.0,
                   help="fixed EEG peak-to-peak threshold in uV (default 500) "
                        "-- the original detector, kept for comparison")
    p.add_argument("--reference", choices=["average", "mastoid"],
                   default="average")
    p.add_argument("--ocular-k", type=float, default=5.0,
                   help="k for the per-participant ocular thresholds "
                        "(threshold = median + k*MAD; default 5)")
    p.add_argument("--blink-min-dur", type=float, default=0.05)
    p.add_argument("--blink-max-dur", type=float, default=0.6)
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
    # utf-8-sig: some sidecars carry a BOM, which plain json.load rejects
    with open(sidecar, encoding="utf-8-sig") as f:
        markers = json.load(f)
    if not markers:
        raise SystemExit(f"ERROR: {sidecar} has no markers in it.")

    raw, fs = P.load_csv_to_raw(args.csv_path, "CAL")
    nyq = fs / 2
    h_freq = min(40.0, nyq * 0.95)
    raw_pp = P.preprocess(raw, h_freq=h_freq, subject_label="CAL",
                          reference=args.reference)
    oc = P.make_ocular_channels(raw, subject_label="CAL",
                                bads=raw_pp.info["bads"])
    n = raw_pp.n_times

    # --- the detectors, each on its own so they can be compared -----------
    masks = {}
    masks["eeg-amp"] = P.continuous_bad_mask(
        raw_pp, window_s=args.artifact_window, step_s=args.artifact_step,
        threshold_uv=args.artifact_threshold, pad_s=0.0)

    thr = P.ocular_thresholds(oc, window_s=args.artifact_window,
                              step_s=args.artifact_step, k=args.ocular_k)
    for name in ("blink", "saccade"):
        trace = getattr(oc, name)
        if trace is not None and thr.get(name):
            masks[f"{name}-p2p"] = P.ptp_event_mask(
                trace[:n], oc.fs, thr[name], args.artifact_window,
                args.artifact_step)
    n_blinks = 0
    if oc.blink is not None and thr.get("blink_velocity"):
        m, n_blinks, _ = P.velocity_event_mask(
            oc.blink[:n], oc.fs, thr["blink_velocity"],
            args.blink_min_dur, args.blink_max_dur)
        masks["blink-vel"] = m

    dur_s = n / fs
    print("=" * 72)
    print("ARTIFACT CALIBRATION CHECK")
    print("=" * 72)
    print(f"  file: {os.path.basename(args.csv_path)}   {dur_s:.0f}s @ {fs:.0f} Hz")
    print(f"  reference={args.reference}   ocular band "
          f"{oc.l_freq:g}-{oc.h_freq:g} Hz   k={args.ocular_k:g}")
    if raw_pp.info["bads"]:
        print(f"  WARNING channels flagged bad: {raw_pp.info['bads']}")
    print()

    print("  Per-participant thresholds estimated from THIS recording:")
    for name in ("blink", "saccade"):
        if thr.get(name) is not None:
            print(f"    {name + ' p2p':<18} {thr[name]:8.1f} uV")
        else:
            print(f"    {name + ' p2p':<18} {'unavailable':>8} "
                  "(electrodes flagged bad)")
    if thr.get("blink_velocity"):
        print(f"    {'blink velocity':<18} {thr['blink_velocity']:8.0f} uV/s"
              f"   -> {n_blinks} blinks "
              f"({60 * n_blinks / dur_s:.1f}/min; typical is 10-20)")
    if thr.get("blink") is not None or thr.get("saccade") is not None:
        bits = " ".join(
            f"--{nm}-threshold-a {thr[nm]:.0f}"
            for nm in ("blink", "saccade") if thr.get(nm) is not None)
        print(f"    to reuse these for subject A:  {bits}")
    print()

    print("  Fraction of the whole recording each detector flags:")
    print("  (a detector flagging a large fraction will 'catch' markers by")
    print("   chance -- read the table below against these numbers)")
    for name, m in masks.items():
        print(f"    {name:<14} {100 * m.mean():5.1f}%")
    print()

    # --- per-marker breakdown --------------------------------------------
    names = list(masks)
    header = f"  {'marker':<18} {'time (s)':>9}  " + "  ".join(
        f"{nm:^11}" for nm in names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    check_n = int(round(args.check_window * fs))
    caught = {nm: 0 for nm in names}
    n_checked = 0
    for mk in markers:
        t = float(mk["rel_time_s"])
        start = int(round(t * fs))
        if start >= n:
            continue
        n_checked += 1
        end = min(n, start + check_n)
        cells = []
        for nm in names:
            # any non-trivial overlap counts as "caught" -- real artifacts are
            # often shorter than --check-window, so a strict majority-overlap
            # bar would mislabel a correctly-detected-but-brief artifact as
            # missed
            pct = 100 * masks[nm][start:end].mean()
            hit = pct > 10
            caught[nm] += hit
            cells.append(f"{pct:4.0f}% {'HIT ' if hit else '  . '}")
        print(f"  {mk['marker']:<18} {t:>9.2f}  " + "  ".join(
            f"{c:^11}" for c in cells))

    print()
    print(f"  Caught (of {n_checked} markers):  " + "   ".join(
        f"{nm}={caught[nm]}/{n_checked}" for nm in names))
    print()

    # --- plot -------------------------------------------------------------
    out_path = args.out or args.csv_path.replace(".csv", "_calibration.png")
    data_uv = raw_pp.get_data() * 1e6
    t = np.arange(n) / fs
    ocular_rows = [(nm, tr) for nm, tr in (("blink", oc.blink),
                                           ("saccade", oc.saccade))
                   if tr is not None]

    fig, axes = plt.subplots(1 + len(ocular_rows), 1, sharex=True,
                             figsize=(15, 4.5 + 2.2 * len(ocular_rows)),
                             gridspec_kw={"height_ratios":
                                          [3] + [1.6] * len(ocular_rows)})
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for i, ch in enumerate(raw_pp.ch_names):
        ax.plot(t, data_uv[i][:n] + i * 400, lw=0.5, label=ch.split("_")[-1])
    for start_s, end_s in mask_to_spans(masks["eeg-amp"], fs):
        ax.axvspan(start_s, end_s, color="red", alpha=0.15, lw=0)
    ax.set_ylabel("uV (offset)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.set_title(f"EEG ({args.reference} reference) — red = flagged by the "
                 f"fixed {args.artifact_threshold:.0f} uV peak-to-peak detector",
                 fontsize=10, loc="left")

    colors = {"blink": "#0072B2", "saccade": "#D55E00"}
    for ax, (nm, trace) in zip(axes[1:], ocular_rows):
        ax.plot(t, trace[:n], lw=0.5, color=colors[nm])
        if thr.get(nm):
            for sign in (+1, -1):
                ax.axhline(sign * thr[nm] / 2.0, color=colors[nm], ls="--",
                           lw=0.8, alpha=0.7)
        shade = f"{nm}-p2p"
        if nm == "blink" and "blink-vel" in masks:
            for s0, s1 in mask_to_spans(masks["blink-vel"], fs):
                ax.axvspan(s0, s1, color="#009E73", alpha=0.30, lw=0)
        if shade in masks:
            for s0, s1 in mask_to_spans(masks[shade], fs):
                ax.axvspan(s0, s1, color="red", alpha=0.12, lw=0)
        ax.set_ylabel(f"{nm} (uV)")
        extra = ("   green = blink-velocity detections"
                 if nm == "blink" and "blink-vel" in masks else "")
        ax.set_title(f"{nm} channel — dashed = +/- half the p2p threshold, "
                     f"red = p2p detections{extra}", fontsize=9, loc="left")

    for mk in markers:
        for ax in axes:
            ax.axvline(float(mk["rel_time_s"]), color="black", ls="--", lw=0.8)
    axes[0].set_xlim(0, dur_s)
    for mk in markers:
        axes[0].text(float(mk["rel_time_s"]), axes[0].get_ylim()[1],
                     mk["marker"], rotation=90, fontsize=7, va="top",
                     ha="right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Artifact calibration — {os.path.basename(args.csv_path)}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # --- per-marker zoom grid --------------------------------------------
    # The full-length figure above shows overall coverage, but at 300s a
    # 300ms blink is a hairline and shading blurs together, so it cannot
    # answer the question the protocol is actually asking: did the detector
    # fire on THIS thing I did. One small window per marker can.
    zoom_path = out_path.replace(".png", "_markers.png")
    shown = [mk for mk in markers if float(mk["rel_time_s"]) * fs < n][:12]
    if shown and ocular_rows:
        ncol = min(4, len(shown))
        nrow = int(np.ceil(len(shown) / ncol))
        figz, axz = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                                 squeeze=False)
        half = int(round(2.0 * fs))
        for idx, mk in enumerate(shown):
            ax = axz[idx // ncol][idx % ncol]
            centre = int(round(float(mk["rel_time_s"]) * fs))
            z0, z1 = max(0, centre - half), min(n, centre + half)
            tz = np.arange(z0, z1) / fs
            scale = max(np.percentile(np.abs(tr), 99) for _, tr in ocular_rows) or 1.0
            for j, (nm, tr) in enumerate(ocular_rows):
                ax.plot(tz, tr[z0:z1] + j * 4.0 * scale, lw=0.7,
                        color=colors[nm])
            for nm, colour, alpha in (("blink-vel", "#009E73", 0.30),
                                      ("blink-p2p", "red", 0.12),
                                      ("saccade-p2p", "#D55E00", 0.12)):
                if nm in masks:
                    for s0, s1 in mask_to_spans(masks[nm][z0:z1], fs):
                        ax.axvspan(z0 / fs + s0, z0 / fs + s1, color=colour,
                                   alpha=alpha, lw=0)
            ax.axvline(float(mk["rel_time_s"]), color="black", ls="--", lw=1.0)
            ax.set_yticks([j * 4.0 * scale for j in range(len(ocular_rows))])
            ax.set_yticklabels([nm for nm, _ in ocular_rows], fontsize=7)
            ax.set_title(f"{mk['marker']}  t={float(mk['rel_time_s']):.1f}s",
                         fontsize=8, loc="left")
            ax.tick_params(labelsize=7)
            ax.margins(x=0)
        for idx in range(len(shown), nrow * ncol):
            axz[idx // ncol][idx % ncol].axis("off")
        figz.suptitle("Per-marker zoom (+/-2s)   black = your marker press, "
                      "green = blink-velocity, red = peak-to-peak",
                      fontsize=11, fontweight="bold")
        figz.tight_layout(rect=(0, 0, 1, 0.95))
        figz.savefig(zoom_path, dpi=120)
        plt.close(figz)
        if len(shown) < len(markers):
            print(f"  (zoom grid shows the first {len(shown)} of "
                  f"{len(markers)} markers)")

    print(f"  Plot saved: {out_path}")
    if shown and ocular_rows:
        print(f"  Zoom grid:  {zoom_path}")
    print("  Black dashed lines are your marker presses. Check that the")
    print("  shading lines up with them: shaded stretches far from any marker")
    print("  are false positives, markers with nothing shaded are misses.")
    print("  If the blink-velocity (green) spans also cover your jaw clench,")
    print("  the duration test is too loose -- lower --blink-max-dur.")


if __name__ == "__main__":
    main()
