"""
pipeline.py — hyperscanning pipeline for two-Muse recordings.

Takes the two CSVs produced by record_both.py and runs the full chain:

    raw CSV   ->   MNE Raw
                   |
                   v
              annotations from is_gap column   (BAD_gap, MNE will skip these)
                   |
                   v
              bandpass filter 1-40 Hz, average reference
                   |
                   v
              fixed-length epochs (2s default), gap epochs auto-rejected
                   |
                   v
              PLV per frequency band (theta, alpha, beta) via HyPyP
                   |
                   v
              inter-brain connectivity matrices + plots

Outputs land in `out/<timestamp>/`:
    plv_<band>.npy              raw 8x8 connectivity matrices per band
    plv_interbrain_<band>.png   4x4 inter-brain matrices (the main result)
    raw_with_gaps.png           sanity check: signal + gap markers
    psd.png                     power spectrum per subject (QC)
    summary.txt                 numerical summary

Usage:
    python pipeline.py <subject_a.csv> <subject_b.csv>
    python pipeline.py a.csv b.csv --epoch-len 2.0 --epoch-overlap 0.5
    python pipeline.py a.csv b.csv --bands alpha             # one band only
    python pipeline.py a.csv b.csv --surrogate 100           # null distribution
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

from scipy.signal import hilbert, butter, filtfilt


CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
FREQ_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


# ============================================================
# 1. LOADING
# ============================================================

def load_stimulus_onset(csv_path):
    """
    Look for a _markers.json sidecar next to the CSV (written by record_single.py).
    Returns the rel_time_s of the first stimulus marker, or None if no sidecar found.
    """
    sidecar = csv_path.replace(".csv", "_markers.json")
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        markers = json.load(f)
    for m in markers:
        if "stimulus" in m.get("marker", "").lower():
            return float(m["rel_time_s"])
    if markers:
        return float(markers[0]["rel_time_s"])
    return None


def load_csv_to_raw(csv_path, subject_label, onset_s=None):
    """
    CSV (lsl_timestamp + 4 channels + is_gap) -> MNE Raw with BAD_gap annotations.

    Each channel gets prefixed with the subject label (e.g. 'A_TP9') so when we
    later concatenate the two subjects' data into one Raw for HyPyP, we can tell
    them apart.
    """
    df = pd.read_csv(csv_path)
    # accept either lsl_timestamp (record_both.py) or time_s (record_single.py)
    ts_col = "lsl_timestamp" if "lsl_timestamp" in df.columns else "time_s"
    required = [ts_col] + CH_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    # IBS stimulus alignment: trim to [onset, end] and re-zero timestamps
    if onset_s is not None:
        df = df[df[ts_col] >= onset_s].copy().reset_index(drop=True)
        df[ts_col] = df[ts_col] - onset_s
        print(f"     stimulus alignment: trimmed to onset at {onset_s:.3f}s")

    # infer sampling rate from timestamps
    dt = np.diff(df[ts_col].values)
    fs = 1.0 / float(np.median(dt))
    fs_rounded = round(fs)
    if abs(fs - fs_rounded) > 0.5:
        print(f"  WARNING: {csv_path} has irregular sampling (~{fs:.2f} Hz)")
    fs = float(fs_rounded)

    # build MNE Raw
    data = df[CH_NAMES].values.T  # shape (n_channels, n_samples), microvolts
    # MNE wants volts; convert from microvolts
    data = data * 1e-6
    # if there are NaN gaps, MNE filtering will explode - fill with 0 for now,
    # the annotations below tell downstream code to ignore those segments anyway
    nan_mask_per_channel = np.isnan(data)
    data = np.where(nan_mask_per_channel, 0.0, data)

    ch_names_prefixed = [f"{subject_label}_{c}" for c in CH_NAMES]
    info = mne.create_info(ch_names=ch_names_prefixed, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # attach standard 10-20 electrode coordinates so MNE knows where TP9/AF7/
    # AF8/TP10 sit on the scalp (enables spatial-color PSD plots, topomaps,
    # and any future montage-aware steps). Renaming is needed because the
    # montage's built-in names ("TP9") don't match our subject-prefixed ones
    # ("A_TP9") -- MNE matches montage points to raw channels by exact name.
    montage = mne.channels.make_standard_montage("standard_1020")
    rename_map = {f"{subject_label}_{c}": c for c in CH_NAMES}
    raw.rename_channels(rename_map)
    raw.set_montage(montage, on_missing="warn", verbose=False)
    raw.rename_channels({v: k for k, v in rename_map.items()})

    # annotations from is_gap column
    if "is_gap" in df.columns:
        gap = df["is_gap"].values.astype(bool)
    else:
        # fall back to any-NaN-across-channels
        gap = nan_mask_per_channel.any(axis=0)

    onsets, durations = gap_runs_to_annotations(gap, fs,
                                                start_time=df[ts_col].iloc[0])
    if len(onsets) > 0:
        anns = mne.Annotations(
            onset=onsets - df[ts_col].iloc[0],   # relative to raw start
            duration=durations,
            description=["BAD_gap"] * len(onsets),
        )
        raw.set_annotations(anns)

    clean_frac = float((~gap).mean())
    print(f"  {subject_label}: {csv_path}")
    print(f"     samples={len(df)}  fs={fs:.0f} Hz  duration={len(df)/fs:.1f}s")
    print(f"     clean fraction={clean_frac:.1%}  gap annotations={len(onsets)}")
    return raw, fs


def gap_runs_to_annotations(gap_mask, fs, start_time=0.0):
    """Turn a boolean is_gap array into (onsets, durations) absolute-time arrays."""
    if not gap_mask.any():
        return np.array([]), np.array([])
    # find run boundaries
    g = gap_mask.astype(int)
    edges = np.diff(np.concatenate(([0], g, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    onsets = start_time + starts / fs
    durations = (ends - starts) / fs
    return onsets, durations


# ============================================================
# 2. PREPROCESS + EPOCH
# ============================================================

def remove_blink_component(raw, subject_label, random_state=42):
    """
    Best-effort blink removal via ICA.

    Muse has no dedicated EOG channel, so the frontal channel (AF7 or AF8,
    most exposed to blinks) is used as a proxy target for
    ICA.find_bads_eog(). Only 4 EEG channels are available in total, so
    separation is weak compared to a real multi-channel montage — at most
    the single most blink-correlated component is removed, to avoid
    discarding real brain signal along with the artifact.
    """
    proxy_ch = next((ch for ch in raw.ch_names if ch.endswith("AF7") or ch.endswith("AF8")), None)
    if proxy_ch is None:
        print(f"     {subject_label}: ICA skipped — no frontal channel found for blink proxy")
        return raw

    ica = ICA(max_iter="auto", random_state=random_state)
    ica.fit(raw, verbose=False)

    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=proxy_ch, verbose=False)
    except Exception as e:
        print(f"     {subject_label}: ICA blink-detection via {proxy_ch} failed ({e}) — no components removed")
        return raw

    if not eog_indices:
        print(f"     {subject_label}: ICA found no clear blink component — no components removed")
        return raw

    ica.exclude = eog_indices[:1]
    print(f"     {subject_label}: ICA removed component {ica.exclude} (blink-correlated via {proxy_ch})")
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


def preprocess(raw, l_freq=1.0, h_freq=40.0, use_ica=False, subject_label=""):
    """Bandpass + optional ICA blink removal + average reference."""
    raw = raw.copy()
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    if use_ica:
        raw = remove_blink_component(raw, subject_label)
    # average reference across that subject's channels only
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def epoch_with_gap_rejection(raw, epoch_len_s, overlap_s, amplitude_uv=150.0):
    """
    Fixed-length epochs, rejecting any epoch that:
      - overlaps a BAD_gap annotation (lost BLE packets), OR
      - has peak-to-peak amplitude > amplitude_uv on any channel (muscle artifact)

    150 µV is a standard threshold for consumer EEG. Lower it (e.g. 100) for
    stricter rejection; raise it (e.g. 200) if too many epochs are lost.
    """
    events = mne.make_fixed_length_events(
        raw, duration=epoch_len_s, overlap=overlap_s
    )
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=epoch_len_s - 1.0 / raw.info["sfreq"],
        baseline=None, preload=True, reject_by_annotation=True,
        reject={"eeg": amplitude_uv * 1e-6},
        verbose=False,
    )
    return epochs


# ============================================================
# 3. PLV
# ============================================================

def plv_manual(epochs_a, epochs_b, band):
    """
    Compute PLV between every channel pair (one from A, one from B).

    Returns: (n_chan_a, n_chan_b) matrix of PLVs averaged across epochs.

    Pure-numpy implementation so you can read it and trust it. HyPyP does the
    same thing under the hood but through several layers of abstraction.
    """
    fs = epochs_a.info["sfreq"]
    lo, hi = band

    # equalize epoch counts: drop whichever has more
    n_ep = min(len(epochs_a), len(epochs_b))
    if n_ep == 0:
        return None
    data_a = epochs_a.get_data()[:n_ep]   # shape (n_ep, n_chan_a, n_samples)
    data_b = epochs_b.get_data()[:n_ep]

    # bandpass filter to target band
    nyq = fs / 2
    if hi >= nyq:
        hi = nyq * 0.99  # avoid butter complaining
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")

    def phase(x):
        # x: (n_ep, n_chan, n_samples) -> phases same shape
        xf = filtfilt(b, a, x, axis=-1)
        return np.angle(hilbert(xf, axis=-1))

    phi_a = phase(data_a)
    phi_b = phase(data_b)

    n_chan_a = phi_a.shape[1]
    n_chan_b = phi_b.shape[1]
    plv = np.zeros((n_chan_a, n_chan_b))

    for i in range(n_chan_a):
        for j in range(n_chan_b):
            # PLV per epoch then average across epochs
            d = phi_a[:, i, :] - phi_b[:, j, :]    # (n_ep, n_samples)
            plv_per_ep = np.abs(np.mean(np.exp(1j * d), axis=-1))
            plv[i, j] = plv_per_ep.mean()
    return plv


def circular_corr_manual(epochs_a, epochs_b, band):
    """
    Jammalamadaka & SenGupta circular correlation between every channel pair.

    Unlike PLV (which collapses phase difference to a magnitude), circular
    correlation preserves sign: +1 = perfectly in-phase, -1 = perfectly
    anti-phase, 0 = no consistent relationship.

    Within each epoch we compute:
        r = Σ sin(φ_A - μ_A) sin(φ_B - μ_B)
            / sqrt( Σ sin²(φ_A - μ_A) · Σ sin²(φ_B - μ_B) )
    then average r across epochs.

    Returns: (n_chan_a, n_chan_b) matrix in [-1, 1], or None if no epochs.
    """
    fs = epochs_a.info["sfreq"]
    lo, hi = band

    n_ep = min(len(epochs_a), len(epochs_b))
    if n_ep == 0:
        return None
    data_a = epochs_a.get_data()[:n_ep]
    data_b = epochs_b.get_data()[:n_ep]

    nyq = fs / 2
    if hi >= nyq:
        hi = nyq * 0.99
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")

    def phase(x):
        xf = filtfilt(b, a, x, axis=-1)
        return np.angle(hilbert(xf, axis=-1))

    phi_a = phase(data_a)   # (n_ep, n_chan_a, n_samples)
    phi_b = phase(data_b)

    n_chan_a = phi_a.shape[1]
    n_chan_b = phi_b.shape[1]
    cc = np.zeros((n_chan_a, n_chan_b))

    for i in range(n_chan_a):
        for j in range(n_chan_b):
            r_per_ep = np.zeros(n_ep)
            for ep in range(n_ep):
                pa = phi_a[ep, i, :]
                pb = phi_b[ep, j, :]
                # circular means
                mu_a = np.arctan2(np.mean(np.sin(pa)), np.mean(np.cos(pa)))
                mu_b = np.arctan2(np.mean(np.sin(pb)), np.mean(np.cos(pb)))
                sa = np.sin(pa - mu_a)
                sb = np.sin(pb - mu_b)
                denom = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
                r_per_ep[ep] = np.sum(sa * sb) / denom if denom > 0 else 0.0
            cc[i, j] = r_per_ep.mean()
    return cc


def surrogate_plv_distribution(epochs_a, epochs_b, band, n_surrogates, seed=0):
    """
    Build a null distribution by shuffling epoch order between subjects.

    If real PLV > 95th percentile of this distribution, the coupling is unlikely
    to be due to chance alone. Standard hyperscanning practice.
    """
    rng = np.random.default_rng(seed)
    n_ep = min(len(epochs_a), len(epochs_b))
    nulls = []
    for k in range(n_surrogates):
        perm = rng.permutation(n_ep)
        # shuffle B's epochs
        epochs_b_shuffled = epochs_b.copy()
        epochs_b_shuffled._data = epochs_b_shuffled._data[perm]
        nulls.append(plv_manual(epochs_a, epochs_b_shuffled, band))
    return np.array(nulls)  # shape (n_surrogates, n_chan_a, n_chan_b)


# ============================================================
# 4. PLOTS
# ============================================================

def plot_raw_with_gaps(raw_a, raw_b, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, raw, label in zip(axes, [raw_a, raw_b], ["A", "B"]):
        data = raw.get_data() * 1e6  # back to microvolts
        t = np.arange(data.shape[1]) / raw.info["sfreq"]
        for i, ch in enumerate(raw.ch_names):
            ax.plot(t, data[i] + i * 100, lw=0.5, label=ch.split("_")[-1])
        # shade gap annotations
        for ann in raw.annotations:
            if "BAD" in ann["description"]:
                ax.axvspan(ann["onset"], ann["onset"] + ann["duration"],
                           color="red", alpha=0.15, lw=0)
        ax.set_ylabel(f"Subject {label}\n(uV, offset)")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw signal with BAD_gap segments highlighted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_psd(raw_a, raw_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, raw, label in zip(axes, [raw_a, raw_b], ["A", "B"]):
        fmax = min(40.0, raw.info["sfreq"] / 2 * 0.99)
        psd = raw.compute_psd(fmin=1, fmax=fmax, verbose=False)
        psd.plot(axes=ax, show=False)
        ax.set_title(f"Subject {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_plv_matrix(plv, band_name, out_path, surrogate_p=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(plv, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CH_NAMES)))
    ax.set_yticks(range(len(CH_NAMES)))
    ax.set_xticklabels([f"B:{c}" for c in CH_NAMES])
    ax.set_yticklabels([f"A:{c}" for c in CH_NAMES])
    ax.set_title(f"Inter-brain PLV — {band_name}")
    plt.colorbar(im, ax=ax, label="PLV")
    for i in range(plv.shape[0]):
        for j in range(plv.shape[1]):
            val = plv[i, j]
            star = ""
            if surrogate_p is not None and surrogate_p[i, j] < 0.05:
                star = "*"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_circ_corr_matrix(cc, band_name, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CH_NAMES)))
    ax.set_yticks(range(len(CH_NAMES)))
    ax.set_xticklabels([f"B:{c}" for c in CH_NAMES])
    ax.set_yticklabels([f"A:{c}" for c in CH_NAMES])
    ax.set_title(f"Inter-brain Circular Corr — {band_name}")
    plt.colorbar(im, ax=ax, label="r (circ)")
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            val = cc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_circ_corr_comparison(ccs_by_band, out_path):
    fig, axes = plt.subplots(1, len(ccs_by_band), figsize=(4 * len(ccs_by_band), 4.5))
    if len(ccs_by_band) == 1:
        axes = [axes]
    for ax, (band, cc) in zip(axes, ccs_by_band.items()):
        im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CH_NAMES)))
        ax.set_yticks(range(len(CH_NAMES)))
        ax.set_xticklabels([f"B:{c}" for c in CH_NAMES], rotation=45, ha="right")
        ax.set_yticklabels([f"A:{c}" for c in CH_NAMES])
        ax.set_title(f"{band}  (mean={cc.mean():.2f})")
    fig.suptitle("Inter-brain Circular Correlation across frequency bands")
    plt.colorbar(im, ax=axes, shrink=0.8, label="r (circ)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_plv_comparison(plvs_by_band, out_path):
    fig, axes = plt.subplots(1, len(plvs_by_band), figsize=(4 * len(plvs_by_band), 4.5))
    if len(plvs_by_band) == 1:
        axes = [axes]
    for ax, (band, plv) in zip(axes, plvs_by_band.items()):
        im = ax.imshow(plv, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(CH_NAMES)))
        ax.set_yticks(range(len(CH_NAMES)))
        ax.set_xticklabels([f"B:{c}" for c in CH_NAMES], rotation=45, ha="right")
        ax.set_yticklabels([f"A:{c}" for c in CH_NAMES])
        ax.set_title(f"{band}  (mean={plv.mean():.2f})")
    fig.suptitle("Inter-brain PLV across frequency bands")
    plt.colorbar(im, ax=axes, shrink=0.8, label="PLV")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    p.add_argument("--epoch-len", type=float, default=2.0,
                   help="epoch length in seconds (default 2.0)")
    p.add_argument("--epoch-overlap", type=float, default=1.0,
                   help="epoch overlap in seconds (default 1.0)")
    p.add_argument("--bands", nargs="+", default=list(FREQ_BANDS.keys()),
                   help="which bands to compute (theta alpha beta)")
    p.add_argument("--surrogate", type=int, default=0,
                   help="number of surrogate permutations for significance (0=off)")
    p.add_argument("--amplitude-threshold", type=float, default=150.0,
                   help="peak-to-peak amplitude threshold in µV for epoch rejection "
                        "(default 150). Lower = stricter. 0 = disabled.")
    p.add_argument("--ica", action="store_true",
                   help="remove the single most blink-correlated ICA component per "
                        "subject before referencing (experimental — only 4 channels "
                        "available, so separation is weak)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    print("="*60)
    print("LOADING")
    print("="*60)
    onset_a = load_stimulus_onset(args.csv_a)
    onset_b = load_stimulus_onset(args.csv_b)
    if onset_a is not None and onset_b is not None:
        print(f"  IBS mode: stimulus markers found "
              f"(A={onset_a:.3f}s, B={onset_b:.3f}s)")
    elif onset_a is not None or onset_b is not None:
        print("  WARNING: only one recording has a stimulus marker — "
              "alignment skipped")
        onset_a = onset_b = None
    else:
        print("  No stimulus markers found — using full recordings")
    raw_a, fs_a = load_csv_to_raw(args.csv_a, "A", onset_s=onset_a)
    raw_b, fs_b = load_csv_to_raw(args.csv_b, "B", onset_s=onset_b)
    if fs_a != fs_b:
        print(f"  WARNING: sampling rates differ ({fs_a} vs {fs_b}). "
              "record_both.py should produce matched 64 Hz.")

    print()
    print("="*60)
    print("PREPROCESSING")
    print("="*60)
    # cap high-pass cutoff below Nyquist; for 64 Hz that's <32 Hz
    nyq = fs_a / 2
    h_freq = min(40.0, nyq * 0.95)
    ica_str = " + ICA blink removal" if args.ica else ""
    print(f"  bandpass 1-{h_freq:.0f} Hz{ica_str}, average reference")
    raw_a_pp = preprocess(raw_a, h_freq=h_freq, use_ica=args.ica, subject_label="A")
    raw_b_pp = preprocess(raw_b, h_freq=h_freq, use_ica=args.ica, subject_label="B")

    plot_raw_with_gaps(raw_a_pp, raw_b_pp, os.path.join(out_dir, "raw_with_gaps.png"))
    plot_psd(raw_a_pp, raw_b_pp, os.path.join(out_dir, "psd.png"))

    print()
    print("="*60)
    print("EPOCHING")
    print("="*60)
    amp_thresh = args.amplitude_threshold if args.amplitude_threshold > 0 else None
    thresh_str = f"{amp_thresh} µV" if amp_thresh else "disabled"
    print(f"  epoch_len={args.epoch_len}s  overlap={args.epoch_overlap}s  "
          f"amplitude_threshold={thresh_str}")
    epochs_a = epoch_with_gap_rejection(raw_a_pp, args.epoch_len, args.epoch_overlap,
                                        amplitude_uv=amp_thresh or 1e9)
    epochs_b = epoch_with_gap_rejection(raw_b_pp, args.epoch_len, args.epoch_overlap,
                                        amplitude_uv=amp_thresh or 1e9)
    print(f"  Subject A: {len(epochs_a)} epochs survived (out of "
          f"{len(epochs_a.drop_log)} attempted)")
    print(f"  Subject B: {len(epochs_b)} epochs survived (out of "
          f"{len(epochs_b.drop_log)} attempted)")

    if len(epochs_a) == 0 or len(epochs_b) == 0:
        print()
        print("  No surviving epochs. The recording is too gappy for this "
              "epoch length.")
        print("  Try:  --epoch-len 1.0 --epoch-overlap 0.5")
        print("  Or: get cleaner data (single-BT-adapter problem).")
        sys.exit(0)

    # Match epochs by original time-slot index, not by position in each
    # subject's own surviving list. A and B reject different epochs (whichever
    # overlap THEIR OWN gaps/artifacts), so truncating by position pairs
    # epochs from different real-world moments -- e.g. A's epoch #5 (its 5th
    # survivor) might be B's epoch #9 in time, growing worse the more the two
    # subjects' rejections diverge. Keep only slots that survived in both.
    common = np.intersect1d(epochs_a.selection, epochs_b.selection)
    epochs_a = epochs_a[np.isin(epochs_a.selection, common)]
    epochs_b = epochs_b[np.isin(epochs_b.selection, common)]
    n_ep = len(common)
    print(f"  Using {n_ep} time-aligned matched epochs for connectivity "
          f"(kept only time slots that survived rejection in both subjects).")

    print()
    print("="*60)
    print("PLV + CIRCULAR CORRELATION PER BAND")
    print("="*60)
    plvs = {}
    ccs = {}
    p_values = {}
    summary_lines = [f"Hyperscanning summary  {datetime.now().isoformat()}"]
    summary_lines.append(f"  A: {args.csv_a}")
    summary_lines.append(f"  B: {args.csv_b}")
    summary_lines.append(f"  fs={fs_a:.0f} Hz, epochs={n_ep}, "
                         f"epoch_len={args.epoch_len}s")
    summary_lines.append("")

    for band_name in args.bands:
        if band_name not in FREQ_BANDS:
            print(f"  skipping unknown band: {band_name}")
            continue
        band = FREQ_BANDS[band_name]
        if band[1] >= nyq:
            print(f"  skipping {band_name} ({band[0]}-{band[1]} Hz): "
                  f"above Nyquist ({nyq:.1f} Hz)")
            continue

        plv = plv_manual(epochs_a, epochs_b, band)
        plvs[band_name] = plv
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean PLV = {plv.mean():.3f}  max = {plv.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"plv_{band_name}.npy"), plv)

        cc = circular_corr_manual(epochs_a, epochs_b, band)
        ccs[band_name] = cc
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean circ-r = {cc.mean():.3f}  min = {cc.min():.3f}  max = {cc.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"circ_corr_{band_name}.npy"), cc)

        if args.surrogate > 0:
            print(f"     running {args.surrogate} surrogates...")
            null = surrogate_plv_distribution(
                epochs_a, epochs_b, band, args.surrogate
            )
            p_val = (null >= plv[None, :, :]).mean(axis=0)
            p_values[band_name] = p_val
            n_sig = int((p_val < 0.05).sum())
            line = f"     significant pairs (p<0.05): {n_sig}/{plv.size}"
            print(line)
            summary_lines.append(line)
            np.save(os.path.join(out_dir, f"plv_p_{band_name}.npy"), p_val)

        plot_plv_matrix(
            plv, band_name,
            os.path.join(out_dir, f"plv_interbrain_{band_name}.png"),
            surrogate_p=p_values.get(band_name),
        )
        plot_circ_corr_matrix(
            cc, band_name,
            os.path.join(out_dir, f"circ_corr_{band_name}.png"),
        )

    if plvs:
        plot_plv_comparison(plvs, os.path.join(out_dir, "plv_comparison.png"))
    if ccs:
        plot_circ_corr_comparison(ccs, os.path.join(out_dir, "circ_corr_comparison.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print()
    print(f"Wrote outputs to: {out_dir}")
    print("  plv_<band>.npy              - PLV matrices")
    print("  plv_interbrain_<band>.png   - PLV heatmaps (0 to 1)")
    print("  plv_comparison.png          - PLV all bands side by side")
    print("  circ_corr_<band>.npy        - circular correlation matrices")
    print("  circ_corr_<band>.png        - circular corr heatmaps (-1 to 1)")
    print("  circ_corr_comparison.png    - circular corr all bands side by side")
    print("  raw_with_gaps.png           - signal + gap markers")
    print("  psd.png                     - power spectrum QC")
    print("  summary.txt                 - numerical summary")


if __name__ == "__main__":
    main()