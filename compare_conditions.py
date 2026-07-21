"""
compare_conditions.py — compare inter-brain synchrony between two conditions
(e.g. "same video" vs "different video") using the same preprocessing / PLV
machinery as pipeline.py, run once per condition, side by side.

Usage:
  python compare_conditions.py \\
      --cond1 same_video recordings/<stampA>_A.csv recordings/<stampA>_B.csv \\
      --cond2 diff_video recordings/<stampB>_A.csv recordings/<stampB>_B.csv \\
      --surrogate 100
"""
import argparse

from pipeline import (
    FREQ_BANDS,
    circular_corr_manual,
    epoch_with_gap_rejection,
    load_csv_to_raw,
    load_stimulus_onset,
    plv_manual,
    preprocess,
    surrogate_plv_distribution,
)


def run_condition(csv_a, csv_b, epoch_len, overlap, amp_thresh, bands,
                   n_surrogate, use_ica):
    onset_a = load_stimulus_onset(csv_a)
    onset_b = load_stimulus_onset(csv_b)
    if onset_a is None or onset_b is None:
        onset_a = onset_b = None

    raw_a, fs_a = load_csv_to_raw(csv_a, "A", onset_s=onset_a)
    raw_b, fs_b = load_csv_to_raw(csv_b, "B", onset_s=onset_b)
    nyq = fs_a / 2
    h_freq = min(40.0, nyq * 0.95)
    raw_a_pp = preprocess(raw_a, h_freq=h_freq, use_ica=use_ica, subject_label="A")
    raw_b_pp = preprocess(raw_b, h_freq=h_freq, use_ica=use_ica, subject_label="B")

    epochs_a = epoch_with_gap_rejection(raw_a_pp, epoch_len, overlap,
                                        amplitude_uv=amp_thresh or 1e9)
    epochs_b = epoch_with_gap_rejection(raw_b_pp, epoch_len, overlap,
                                        amplitude_uv=amp_thresh or 1e9)
    n_ep = min(len(epochs_a), len(epochs_b))
    if n_ep == 0:
        return None
    epochs_a, epochs_b = epochs_a[:n_ep], epochs_b[:n_ep]

    results = {}
    for band_name in bands:
        band = FREQ_BANDS[band_name]
        if band[1] >= nyq:
            continue
        plv = plv_manual(epochs_a, epochs_b, band)
        cc = circular_corr_manual(epochs_a, epochs_b, band)
        entry = {
            "n_epochs": n_ep,
            "plv_mean": plv.mean(), "plv_max": plv.max(),
            "cc_mean": cc.mean(), "cc_min": cc.min(), "cc_max": cc.max(),
        }
        if n_surrogate > 0:
            null = surrogate_plv_distribution(epochs_a, epochs_b, band, n_surrogate)
            p = (null >= plv[None, :, :]).mean(axis=0)
            entry["n_sig"] = int((p < 0.05).sum())
            entry["n_pairs"] = plv.size
        results[band_name] = entry
    return results


def print_comparison(label1, res1, label2, res2, bands):
    print()
    header = f"{'band':6s} {'metric':14s} {label1:>16s} {label2:>16s}"
    print(header)
    print("-" * len(header))
    for band in bands:
        r1, r2 = res1.get(band), res2.get(band)
        if r1 is None or r2 is None:
            continue
        print(f"{band:6s} {'PLV mean':14s} {r1['plv_mean']:16.3f} {r2['plv_mean']:16.3f}")
        print(f"{'':6s} {'circ-r mean':14s} {r1['cc_mean']:16.3f} {r2['cc_mean']:16.3f}")
        if "n_sig" in r1 and "n_sig" in r2:
            sig1 = f"{r1['n_sig']}/{r1['n_pairs']}"
            sig2 = f"{r2['n_sig']}/{r2['n_pairs']}"
            print(f"{'':6s} {'sig pairs':14s} {sig1:>16s} {sig2:>16s}")
    print()
    print(f"{label1}: {res1[bands[0]]['n_epochs']} epochs   "
          f"{label2}: {res2[bands[0]]['n_epochs']} epochs")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond1", nargs=3, metavar=("LABEL", "CSV_A", "CSV_B"), required=True)
    p.add_argument("--cond2", nargs=3, metavar=("LABEL", "CSV_A", "CSV_B"), required=True)
    p.add_argument("--epoch-len", type=float, default=2.0)
    p.add_argument("--epoch-overlap", type=float, default=1.0)
    p.add_argument("--amplitude-threshold", type=float, default=150.0)
    p.add_argument("--bands", nargs="+", default=list(FREQ_BANDS.keys()))
    p.add_argument("--surrogate", type=int, default=0,
                   help="surrogate permutations per condition (0=off)")
    p.add_argument("--ica", action="store_true")
    args = p.parse_args()

    label1, csv_a1, csv_b1 = args.cond1
    label2, csv_a2, csv_b2 = args.cond2

    print(f"Running condition '{label1}': {csv_a1}  +  {csv_b1}")
    res1 = run_condition(csv_a1, csv_b1, args.epoch_len, args.epoch_overlap,
                          args.amplitude_threshold, args.bands, args.surrogate, args.ica)
    print(f"Running condition '{label2}': {csv_a2}  +  {csv_b2}")
    res2 = run_condition(csv_a2, csv_b2, args.epoch_len, args.epoch_overlap,
                          args.amplitude_threshold, args.bands, args.surrogate, args.ica)

    if res1 is None or res2 is None:
        print("One of the conditions had no surviving epochs — cannot compare.")
        return

    print_comparison(label1, res1, label2, res2, args.bands)


if __name__ == "__main__":
    main()
