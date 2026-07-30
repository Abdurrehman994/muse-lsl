"""
compare_conditions.py — contrast inter-brain synchrony between two task
conditions recorded within ONE continuous session, segmented via block
markers. Works with any block-design task script's markers -- e.g.
cooperative_task.py's ALONE vs COOPERATIVE, or video_task.py's ALONE vs
VIDEO -- see BLOCK_BOUNDARY_MARKERS below; it just needs exactly two
distinct condition labels present in the recording's _markers.json.

Uses the same continuous artifact-rejection / PLV / circular-correlation
machinery as pipeline.py's default path (see pipeline.py, changelog point 6):
no fixed epochs -- a sliding-window artifact mask per subject, Hilbert
transform on the full continuous band-passed recording, and only the
jointly-clean SAMPLES that also fall inside a given condition's block(s) are
used for that condition's PLV / circular-correlation sum.

Significance for the ALONE-vs-COOPERATIVE difference comes from a
BLOCK-level permutation test: the recording is split into blocks by the
marker timestamps (each block belongs to one condition), and the null
distribution for the PLV/circ-corr difference is built by reshuffling which
blocks count as ALONE vs COOPERATIVE (preserving each block's internal
data), NOT by shuffling individual samples -- adjacent EEG samples are
autocorrelated, so a sample-level shuffle would be anti-conservative (the
same class of bias --epoch-len used to have, see pipeline.py point 5).
This means --reps in whichever task script you used matters: with only 1
block per condition there are only 2 possible block-to-label assignments,
so the permutation test has essentially no resolution. Aim for --reps 3+.

Usage:
    python compare_conditions.py recordings/<stamp>_A.csv recordings/<stamp>_B.csv
    python compare_conditions.py a.csv b.csv --bands theta alpha
    python compare_conditions.py a.csv b.csv --contrast-perm 500
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (
    CH_NAMES,
    FREQ_BANDS,
    analytic_signal,
    circ_corr_masked,
    continuous_bad_mask,
    fdr_bh,
    load_csv_to_raw,
    plv_masked,
    prefilter_raw_for_band,
    preprocess,
)

# Block-boundary markers recognized across all block-design task scripts
# (cooperative_task.py's ALONE_start/COOPERATIVE_start, video_task.py's
# ALONE_start/VIDEO_start). An exact-match registry rather than a generic
# "ends with _start" check, since cooperative_task.py also fires a
# COOPERATIVE_trial_start marker once per quiz question (not once per
# block) -- treating that as a third condition would break the
# exactly-2-labels assumption below. Add new task scripts' block markers
# here as they're built.
BLOCK_BOUNDARY_MARKERS = {"ALONE_start", "COOPERATIVE_start", "VIDEO_start"}


def load_all_markers(csv_path):
    """Every marker event in the _markers.json sidecar, sorted by time."""
    sidecar = csv_path.replace(".csv", "_markers.json")
    if not os.path.exists(sidecar):
        return []
    with open(sidecar) as f:
        markers = json.load(f)
    return sorted(markers, key=lambda m: m["rel_time_s"])


def condition_label(marker_name):
    """'ALONE_start' -> 'ALONE', or None if this isn't a recognized
    block-boundary marker (see BLOCK_BOUNDARY_MARKERS)."""
    if marker_name in BLOCK_BOUNDARY_MARKERS:
        return marker_name[: -len("_start")]
    return None


def build_condition_segments(markers, recording_end_s):
    """
    Turn a sorted marker list into (condition_label, start_s, end_s) blocks --
    each condition marker starts a block that runs until the next marker (or
    the end of the recording, for the last block).
    """
    block_markers = [m for m in markers if condition_label(m["marker"]) is not None]
    segments = []
    for i, m in enumerate(block_markers):
        start = float(m["rel_time_s"])
        end = (float(block_markers[i + 1]["rel_time_s"])
               if i + 1 < len(block_markers) else recording_end_s)
        segments.append((condition_label(m["marker"]), start, end))
    return segments


def block_sample_ranges(segments, label, fs, n_samples):
    """List of (start_sample, end_sample) for every block belonging to `label`."""
    out = []
    for seg_label, start, end in segments:
        if seg_label != label:
            continue
        s = max(0, int(round(start * fs)))
        e = min(n_samples, int(round(end * fs)))
        if e > s:
            out.append((s, e))
    return out


def mask_from_blocks(n_samples, blocks):
    """Boolean mask, True inside any of the given (start, end) sample ranges."""
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in blocks:
        mask[s:e] = True
    return mask


def run_condition_contrast(csv_a, csv_b, bands, artifact_window, artifact_step,
                            artifact_threshold, n_perm, correction, use_ica,
                            out_dir, seed=0):
    print("="*60)
    print("LOADING")
    print("="*60)
    raw_a, fs_a = load_csv_to_raw(csv_a, "A", onset_s=None)
    raw_b, fs_b = load_csv_to_raw(csv_b, "B", onset_s=None)
    if fs_a != fs_b:
        print(f"  WARNING: sampling rates differ ({fs_a} vs {fs_b})")
    fs = fs_a

    markers = load_all_markers(csv_a)
    if not markers:
        print(f"  ERROR: no _markers.json sidecar found for {csv_a} -- "
              "this script needs block markers from cooperative_task.py.")
        return
    recording_end_s = min(len(raw_a.times), len(raw_b.times)) / fs
    segments = build_condition_segments(markers, recording_end_s)
    if not segments:
        print("  ERROR: markers found, but none matched ALONE/COOPERATIVE "
              "-- was this recording run with cooperative_task.py?")
        return
    labels_present = sorted(set(s[0] for s in segments))
    print(f"  Found {len(segments)} condition blocks: "
          f"{ {lab: sum(1 for s in segments if s[0] == lab) for lab in labels_present} }")

    print()
    print("="*60)
    print("PREPROCESSING + CONTINUOUS ARTIFACT DETECTION")
    print("="*60)
    nyq = fs / 2
    h_freq = min(40.0, nyq * 0.95)
    raw_a_pp = preprocess(raw_a, h_freq=h_freq, use_ica=use_ica, subject_label="A")
    raw_b_pp = preprocess(raw_b, h_freq=h_freq, use_ica=use_ica, subject_label="B")

    bad_a = continuous_bad_mask(raw_a_pp, window_s=artifact_window,
                                 step_s=artifact_step, threshold_uv=artifact_threshold)
    bad_b = continuous_bad_mask(raw_b_pp, window_s=artifact_window,
                                 step_s=artifact_step, threshold_uv=artifact_threshold)
    n_common = min(len(bad_a), len(bad_b))
    bad_a, bad_b = bad_a[:n_common], bad_b[:n_common]
    base_good = ~(bad_a | bad_b)
    print(f"  Jointly clean: {base_good.sum() / fs:.1f}s / {n_common / fs:.1f}s "
          f"({100 * base_good.mean():.1f}%)")

    blocks_by_label = {
        lab: block_sample_ranges(segments, lab, fs, n_common) for lab in labels_present
    }
    for lab, blocks in blocks_by_label.items():
        n_reps = len(blocks)
        total_s = sum(e - s for s, e in blocks) / fs
        print(f"  {lab}: {n_reps} block(s), {total_s:.1f}s total")
        if n_reps < 3:
            print(f"     NOTE: only {n_reps} block(s) -- the block-level "
                  "permutation test below will have very coarse resolution. "
                  "Consider --reps 3+ in cooperative_task.py next time.")

    if len(labels_present) != 2:
        print(f"  ERROR: need exactly 2 condition labels to contrast, found "
              f"{labels_present}")
        return
    label1, label2 = labels_present

    os.makedirs(out_dir, exist_ok=True)
    summary_lines = [f"Condition contrast: {label1} vs {label2}"]
    summary_lines.append(f"  A: {csv_a}")
    summary_lines.append(f"  B: {csv_b}")
    summary_lines.append("")

    print()
    print("="*60)
    print("PLV + CIRCULAR CORRELATION PER CONDITION PER BAND")
    print("="*60)
    rng = np.random.default_rng(seed)

    for band_name in bands:
        if band_name not in FREQ_BANDS:
            print(f"  skipping unknown band: {band_name}")
            continue
        band = FREQ_BANDS[band_name]
        if band[1] >= nyq:
            print(f"  skipping {band_name}: above Nyquist ({nyq:.1f} Hz)")
            continue

        raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
        raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
        analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
        analytic_b = analytic_signal(raw_b_band, n_samples=n_common)

        cond_mask = {
            lab: base_good & mask_from_blocks(n_common, blocks_by_label[lab])
            for lab in labels_present
        }
        for lab in labels_present:
            if cond_mask[lab].sum() == 0:
                print(f"  {band_name}/{lab}: 0 clean samples -- skipping band")
                break
        else:
            plv1 = plv_masked(analytic_a, analytic_b, cond_mask[label1])
            plv2 = plv_masked(analytic_a, analytic_b, cond_mask[label2])
            cc1 = circ_corr_masked(analytic_a, analytic_b, cond_mask[label1])
            cc2 = circ_corr_masked(analytic_a, analytic_b, cond_mask[label2])

            line = (f"  {band_name:6s}: PLV {label1}={plv1.mean():.3f}  "
                    f"{label2}={plv2.mean():.3f}  diff={plv1.mean() - plv2.mean():+.3f}")
            print(line); summary_lines.append(line)
            line = (f"  {band_name:6s}: circ-r {label1}={cc1.mean():.3f}  "
                    f"{label2}={cc2.mean():.3f}  diff={cc1.mean() - cc2.mean():+.3f}")
            print(line); summary_lines.append(line)

            # -- block-level permutation null for the PLV/circ-corr difference --
            all_blocks = [(lab, s, e) for lab in labels_present
                          for s, e in blocks_by_label[lab]]
            n1 = len(blocks_by_label[label1])
            plv_null_diffs, cc_null_diffs = [], []
            for _ in range(n_perm):
                perm = rng.permutation(len(all_blocks))
                pseudo1_blocks = [all_blocks[i][1:] for i in perm[:n1]]
                pseudo2_blocks = [all_blocks[i][1:] for i in perm[n1:]]
                m1 = base_good & mask_from_blocks(n_common, pseudo1_blocks)
                m2 = base_good & mask_from_blocks(n_common, pseudo2_blocks)
                if m1.sum() == 0 or m2.sum() == 0:
                    continue
                p1 = plv_masked(analytic_a, analytic_b, m1)
                p2 = plv_masked(analytic_a, analytic_b, m2)
                plv_null_diffs.append(p1 - p2)
                c1 = circ_corr_masked(analytic_a, analytic_b, m1)
                c2 = circ_corr_masked(analytic_a, analytic_b, m2)
                cc_null_diffs.append(c1 - c2)

            if plv_null_diffs:
                plv_null_diffs = np.array(plv_null_diffs)
                cc_null_diffs = np.array(cc_null_diffs)
                plv_diff = plv1 - plv2
                cc_diff = cc1 - cc2
                plv_p = (np.abs(plv_null_diffs) >= np.abs(plv_diff)[None, :, :]).mean(axis=0)
                cc_p = (np.abs(cc_null_diffs) >= np.abs(cc_diff)[None, :, :]).mean(axis=0)

                if correction == "fdr":
                    plv_sig, _ = fdr_bh(plv_p)
                    cc_sig, _ = fdr_bh(cc_p)
                else:
                    plv_sig = plv_p < 0.05
                    cc_sig = cc_p < 0.05
                line = (f"     PLV diff significant pairs ({len(plv_null_diffs)} "
                        f"block-perms, {correction}): {int(plv_sig.sum())}/{plv_diff.size}")
                print(line); summary_lines.append(line)
                line = (f"     circ-r diff significant pairs ({len(cc_null_diffs)} "
                        f"block-perms, {correction}): {int(cc_sig.sum())}/{cc_diff.size}")
                print(line); summary_lines.append(line)

                np.save(os.path.join(out_dir, f"plv_{label1}_{band_name}.npy"), plv1)
                np.save(os.path.join(out_dir, f"plv_{label2}_{band_name}.npy"), plv2)
                np.save(os.path.join(out_dir, f"plv_diff_{band_name}.npy"), plv_diff)
                plot_diff_matrix(plv_diff, band_name, label1, label2, "PLV",
                                  os.path.join(out_dir, f"plv_diff_{band_name}.png"),
                                  sig_mask=plv_sig, vlim=1.0)
                plot_diff_matrix(cc_diff, band_name, label1, label2, "circ-r",
                                  os.path.join(out_dir, f"circ_corr_diff_{band_name}.png"),
                                  sig_mask=cc_sig, vlim=2.0)
            else:
                print("     block-level permutation produced no usable draws "
                      "-- too few clean samples per pseudo-condition")

    with open(os.path.join(out_dir, "contrast_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print()
    print(f"Wrote outputs to: {out_dir}")


def plot_diff_matrix(diff, band_name, label1, label2, metric_name, out_path,
                      sig_mask=None, vlim=1.0):
    vmax = max(0.02, float(np.abs(diff).max()))
    vmax = min(vmax, vlim)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(CH_NAMES)))
    ax.set_yticks(range(len(CH_NAMES)))
    ax.set_xticklabels([f"B:{c}" for c in CH_NAMES])
    ax.set_yticklabels([f"A:{c}" for c in CH_NAMES])
    ax.set_title(f"{metric_name} diff ({label1} - {label2}) — {band_name}")
    plt.colorbar(im, ax=ax, label=f"Δ {metric_name}")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            val = diff[i, j]
            star = "*" if sig_mask is not None and sig_mask[i, j] else ""
            ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    p.add_argument("--bands", nargs="+", default=list(FREQ_BANDS.keys()))
    p.add_argument("--artifact-window", type=float, default=0.5)
    p.add_argument("--artifact-step", type=float, default=0.1)
    p.add_argument("--artifact-threshold", type=float, default=500.0)
    p.add_argument("--contrast-perm", type=int, default=500,
                   help="block-level permutations for the condition contrast "
                        "null distribution (default 500)")
    p.add_argument("--correction", choices=["fdr", "none"], default="fdr")
    p.add_argument("--ica", action="store_true")
    p.add_argument("--out-dir", default="out/condition_contrast")
    args = p.parse_args()

    run_condition_contrast(
        args.csv_a, args.csv_b, args.bands,
        args.artifact_window, args.artifact_step, args.artifact_threshold,
        args.contrast_perm, args.correction, args.ica, args.out_dir,
    )


if __name__ == "__main__":
    main()
