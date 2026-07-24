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
    plv_<band>.npy              raw 4x4 inter-brain connectivity matrices per band
    plv_interbrain_<band>.png   4x4 inter-brain matrices (the main result)
    raw_with_gaps.png           sanity check: signal + gap markers
    psd.png                     power spectrum per subject (QC)
    summary.txt                 numerical summary

Usage:
    python pipeline.py <subject_a.csv> <subject_b.csv>
    python pipeline.py a.csv b.csv --epoch-len 2.0 --epoch-overlap 0.5
    python pipeline.py a.csv b.csv --bands alpha             # one band only
    python pipeline.py a.csv b.csv --surrogate 100           # within-dyad null (epoch shuffle)
    python pipeline.py a.csv b.csv --pool-dir recordings/other_dyads/   # cross-dyad null (pseudo-pairs)

-----------------------------------------------------------------------------
CHANGES IN THIS VERSION (see conversation notes / internship writeup)
-----------------------------------------------------------------------------
1. PSEUDO-PAIR (CROSS-DYAD) SURROGATE, in addition to within-dyad epoch shuffle.
   Within-dyad epoch shuffling only destroys coupling that varies from epoch to
   epoch. A continuous, unchanging stimulus (e.g. a 6 Hz flicker running the
   whole recording) produces a near-constant per-epoch phase relationship, so
   shuffling epoch order barely changes the metric -- the "surrogate" ends up
   just as high as the real value, and you get a non-significant result even
   though the raw PLV/circ-corr is high. That is NOT a null result about
   coupling; it means the within-dyad shuffle test is the wrong tool for a
   stimulus-locked, stationary signal like SSVEP.
   Pseudo-pairing (comparing subject A against OTHER, non-partner people who
   watched the same or a comparable stimulus) tests the thing you actually
   care about: is this dyad's coupling any different from what you'd get by
   chance-pairing two people who were never in the same session together?
   This is the standard hyperscanning validity check in the literature
   (e.g. Burgess 2013 "cautionary note"; pseudo-pair designs used throughout
   the hyperscanning field). Use --pool-dir to point at a folder of OTHER
   subjects' single-person recordings (any CSVs with the same column format)
   and the pipeline will build a cross-dyad null distribution from them.

2. FDR CORRECTION instead of raw per-pair p<0.05 counts.
   With 16 channel pairs tested per band at uncorrected p<0.05, you'd expect
   ~0.8 false positives per band by chance alone even under pure noise. The
   old "1/16" and "2/16" significant-pair counts were not meaningfully
   different from chance. This version applies Benjamini-Hochberg FDR
   correction across the 16 pairs (per band) before counting "significant"
   pairs, and reports both raw and corrected counts so you can see the
   difference.

3. OPTIONAL PRE-FILTERING before epoching (--prefilter).
   Previously, each 2-second epoch was independently narrowband-filtered
   inside HyPyP's compute_freq_bands (filter_signal=True). For a narrow
   band like stim_6hz (1 Hz wide) a 4th-order IIR filter's transient/edge
   response can occupy a large fraction of a 2-second window, which can
   inflate apparent phase consistency across epochs independent of any real
   coupling. --prefilter instead band-passes the CONTINUOUS raw signal
   BEFORE epoching (so filter transients only occur once, at the start/end
   of the whole recording, not at every epoch boundary), then epochs the
   already-filtered signal and passes filter_signal=False into HyPyP.
   This is now the default; use --no-prefilter to restore the old
   per-epoch-filtering behavior for comparison.
-----------------------------------------------------------------------------
"""
import argparse
import glob
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
from hypyp import analyses


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


def load_and_epoch_subject(csv_path, subject_label, epoch_len_s, overlap_s,
                            h_freq=40.0, amplitude_uv=150.0, use_ica=False,
                            align_onset=True, quiet=False):
    """
    Full load -> preprocess -> epoch chain for ONE subject's CSV.

    Factored out so the same steps can be reused for the two main subjects
    AND for any --pool-dir subjects used in pseudo-pair validation (they all
    need to go through identical preprocessing to be a fair comparison).

    Returns (epochs, fs) or (None, None) if the file couldn't be used.
    """
    try:
        onset = load_stimulus_onset(csv_path) if align_onset else None
        raw, fs = load_csv_to_raw(csv_path, subject_label, onset_s=onset)
    except Exception as e:
        if not quiet:
            print(f"  WARNING: could not load {csv_path}: {e}")
        return None, None

    nyq = fs / 2
    h_freq_eff = min(h_freq, nyq * 0.95)
    raw_pp = preprocess(raw, h_freq=h_freq_eff, use_ica=use_ica, subject_label=subject_label)
    epochs = epoch_with_gap_rejection(raw_pp, epoch_len_s, overlap_s,
                                       amplitude_uv=amplitude_uv)
    if len(epochs) == 0:
        if not quiet:
            print(f"  WARNING: {csv_path} produced 0 usable epochs — skipping")
        return None, fs
    return epochs, fs


# ============================================================
# 3. CONNECTIVITY (PLV / circular correlation)
# ============================================================

def bandpass_epochs_array(data, sfreq, band, order=4):
    """
    Band-pass an (n_epochs, n_channels, n_times) array using the SAME filter
    design HyPyP would otherwise apply internally (IIR Butterworth), but
    applied to CONTINUOUS per-epoch arrays only when pre-filtering wasn't
    already done on the raw signal. Kept for completeness / fallback use;
    the preferred path (--prefilter) filters the CONTINUOUS raw signal
    instead, see `prefilter_raw_for_band`.
    """
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    low, high = band
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def prefilter_raw_for_band(raw, band, order=4):
    """
    Band-pass the CONTINUOUS raw signal into `band` BEFORE epoching.

    Filtering the full-length continuous signal means any filter edge/
    transient effects only happen once (at the very start/end of the whole
    recording), instead of once per 2-second epoch. This matters most for
    narrow bands (e.g. a 1 Hz-wide SSVEP band) where a short epoch may be
    only a few filter time-constants long, and per-epoch filtering can
    inflate phase consistency across all epochs independent of any real
    coupling.
    """
    raw_band = raw.copy()
    raw_band.filter(l_freq=band[0], h_freq=band[1], picks="eeg",
                     method="iir",
                     iir_params={"order": order, "ftype": "butter"},
                     verbose=False)
    return raw_band


def _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode, sfreq,
                               already_filtered=False):
    """
    Compute an inter-brain connectivity block with HyPyP.

    If already_filtered=True, the epochs' data is assumed to already be
    band-passed (via prefilter_raw_for_band on the continuous raw signal
    before epoching), and HyPyP is told not to filter again
    (filter_signal=False) — it will just compute the analytic (Hilbert)
    signal and the sync measure. If already_filtered=False (old default
    behaviour), HyPyP band-passes each short epoch independently.
    """
    n_ep = min(len(epochs_a), len(epochs_b))
    if n_ep == 0:
        return None

    data = np.array([
        epochs_a.get_data()[:n_ep],
        epochs_b.get_data()[:n_ep],
    ])
    freq_bands = {"band": list(band)}
    complex_signal = analyses.compute_freq_bands(
        data=data,
        sampling_rate=int(round(float(sfreq))),
        freq_bands=freq_bands,
        filter_signal=not already_filtered,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
    )
    con = analyses.compute_sync(
        complex_signal,
        mode=mode,
        epochs_average=True,
    )
    # HyPyP returns (n_freq, 2*n_channels, 2*n_channels) when epochs_average=True.
    con = np.asarray(con)[0]
    n_chan = len(CH_NAMES)
    return con[:n_chan, n_chan:2 * n_chan]


def plv_hypyp(epochs_a, epochs_b, band, sfreq, already_filtered=False):
    """
    Compute PLV between every channel pair (one from A, one from B) using HyPyP.

    Returns: (n_chan_a, n_chan_b) matrix of PLVs averaged across epochs.
    """
    return _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode="plv",
                                      sfreq=sfreq, already_filtered=already_filtered)


def circular_corr_hypyp(epochs_a, epochs_b, band, sfreq, already_filtered=False):
    """
    Adjusted circular correlation (ACCorr) between every channel pair, via HyPyP.

    Unlike PLV (which collapses phase difference to a magnitude), ACCorr
    preserves sign: positive = in-phase-leaning, negative = anti-phase-leaning,
    0 = no consistent relationship. Per-pair phase centering (rather than one
    global circular mean) gives a more accurate estimate than plain 'ccorr',
    which HyPyP computes by averaging abs(r) per epoch and therefore can never
    return a negative value.

    Zimmermann et al. (2024), Imaging Neuroscience, 2.

    Returns: (n_chan_a, n_chan_b) matrix, signed, or None if no epochs.
    """
    return _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode="accorr",
                                      sfreq=sfreq, already_filtered=already_filtered)


def surrogate_distribution(epochs_a, epochs_b, band, n_surrogates, metric_fn, sfreq,
                            already_filtered=False, seed=0):
    """
    WITHIN-DYAD null: shuffle epoch order between subjects A and B.

    If the real value falls outside this distribution, the coupling is unlikely
    to be due to chance ALONE -- but note the important caveat: this only tests
    whether coupling varies meaningfully from epoch to epoch. For a continuous,
    non-varying stimulus (e.g. a flicker running the whole recording), shuffling
    epoch order barely changes anything, and this test will under-detect real
    stimulus-locked signal. Use pseudo_pair_distribution() (cross-dyad) for a
    more appropriate test of stimulus-locked / stationary coupling.
    """
    rng = np.random.default_rng(seed)
    n_ep = min(len(epochs_a), len(epochs_b))
    nulls = []
    for k in range(n_surrogates):
        perm = rng.permutation(n_ep)
        # shuffle B's epochs
        epochs_b_shuffled = epochs_b.copy()
        epochs_b_shuffled._data = epochs_b_shuffled._data[perm]
        nulls.append(metric_fn(epochs_a, epochs_b_shuffled, band, sfreq,
                                already_filtered=already_filtered))
    return np.array(nulls)  # shape (n_surrogates, n_chan_a, n_chan_b)


def pseudo_pair_distribution(target_epochs, pool_epochs_list, band, metric_fn, sfreq,
                              already_filtered=False, shuffles_per_pool_member=1,
                              seed=0):
    """
    CROSS-DYAD null: compare `target_epochs` (one real subject) against
    epochs from OTHER people who were never in the same session with them
    (the --pool-dir recordings).

    This is the standard hyperscanning "pseudo-pair" validity check: if a
    real dyad's coupling is no higher than what you get pairing that person
    with random strangers who happened to be exposed to a similar stimulus,
    the "coupling" is not evidence of a genuine dyad-specific effect -- it's
    just shared, stimulus-driven, but independent, brain activity.

    Especially important for stationary/continuous stimuli (e.g. SSVEP),
    where within-dyad epoch shuffling (see surrogate_distribution) can't
    tell real coupling apart from shared independent entrainment, because
    shuffling doesn't change the stimulus each epoch is locked to.

    Returns: array shape (n_pool * shuffles_per_pool_member, n_chan, n_chan)
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for pool_epochs in pool_epochs_list:
        n_ep = min(len(target_epochs), len(pool_epochs))
        if n_ep == 0:
            continue
        for _ in range(shuffles_per_pool_member):
            perm = rng.permutation(len(pool_epochs))[:n_ep]
            pool_shuffled = pool_epochs.copy()
            pool_shuffled._data = pool_shuffled._data[perm]
            val = metric_fn(target_epochs, pool_shuffled, band, sfreq,
                             already_filtered=already_filtered)
            if val is not None:
                nulls.append(val)
    if not nulls:
        return None
    return np.array(nulls)


# ============================================================
# 3b. MULTIPLE-COMPARISONS CORRECTION
# ============================================================

def fdr_bh(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Takes a flat array of p-values, returns (reject_mask, corrected_pvals)
    both the same shape as the input. Used instead of raw per-pair p<0.05
    counting: with 16 channel-pair tests per band at uncorrected p<0.05,
    ~0.8 false positives are expected per band by chance alone, so raw
    counts of "1/16" or "2/16" significant pairs are not meaningfully
    different from noise. FDR correction controls the expected proportion
    of false discoveries among the pairs called significant.
    """
    pvals = np.asarray(pvals, dtype=float)
    shape = pvals.shape
    flat = pvals.ravel()
    n = len(flat)
    order = np.argsort(flat)
    ranked = flat[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    if below.any():
        max_i = np.max(np.where(below)[0])
        cutoff = ranked[max_i]
    else:
        cutoff = -1.0  # nothing survives
    reject_flat = flat <= cutoff
    # corrected p-values (BH step-up), monotone
    corrected = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        corrected[i] = val
        prev = val
    corrected_full = np.empty(n)
    corrected_full[order] = corrected
    reject_full = np.empty(n, dtype=bool)
    reject_full[order] = reject_flat
    return reject_full.reshape(shape), corrected_full.reshape(shape)


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
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_plv_matrix(plv, band_name, out_path, surrogate_p=None, sig_mask=None):
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
            if sig_mask is not None and sig_mask[i, j]:
                star = "*"
            elif surrogate_p is not None and surrogate_p[i, j] < 0.05:
                star = "(*)"  # uncorrected only
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_circ_corr_matrix(cc, band_name, out_path, surrogate_p=None, sig_mask=None):
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
            star = ""
            if sig_mask is not None and sig_mask[i, j]:
                star = "*"
            elif surrogate_p is not None and surrogate_p[i, j] < 0.05:
                star = "(*)"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
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
    p.add_argument("--stim-hz", type=float, default=None,
                   help="SSVEP reversal frequency (Hz), e.g. 6.0 for a 6 Hz "
                        "checkerboard/flicker. Adds a narrow band centered "
                        "here (see --stim-bandwidth) on top of --bands. NOTE: "
                        "for a continuous, non-varying flicker, prefer "
                        "--pool-dir (pseudo-pair) over --surrogate (within-"
                        "dyad shuffle) to validate this band -- see module "
                        "docstring.")
    p.add_argument("--stim-bandwidth", type=float, default=1.0,
                   help="full width in Hz of the narrow band around --stim-hz "
                        "(default 1.0, i.e. +/-0.5 Hz)")
    p.add_argument("--surrogate", type=int, default=0,
                   help="number of within-dyad epoch-shuffle surrogate "
                        "permutations (0=off). Weak for continuous/stationary "
                        "stimuli -- see --pool-dir.")
    p.add_argument("--pool-dir", default=None,
                   help="directory of OTHER subjects' single-person CSVs "
                        "(same column format) to build a cross-dyad "
                        "pseudo-pair null distribution from. Preferred over "
                        "--surrogate for stimulus-locked bands like --stim-hz.")
    p.add_argument("--pool-shuffles", type=int, default=3,
                   help="random epoch-order draws per pool member when "
                        "building the pseudo-pair null (default 3)")
    p.add_argument("--pool-amplitude-threshold", type=float, default=None,
                   help="peak-to-peak amplitude threshold (uV) applied when "
                        "epoching --pool-dir recordings, separate from "
                        "--amplitude-threshold. Defaults to whatever "
                        "--amplitude-threshold is. Pool recordings often come "
                        "from unrelated sessions with different noise levels "
                        "(e.g. a hardware test recording) -- loosen this (or "
                        "pass 0 to disable) if they get rejected wholesale "
                        "under the main dyad's threshold.")
    p.add_argument("--correction", choices=["fdr", "none"], default="fdr",
                   help="multiple-comparisons correction across the 16 "
                        "channel pairs per band (default fdr). 'none' "
                        "restores the old raw p<0.05 counting.")
    p.add_argument("--amplitude-threshold", type=float, default=150.0,
                   help="peak-to-peak amplitude threshold in µV for epoch rejection "
                        "(default 150). Lower = stricter. 0 = disabled.")
    p.add_argument("--ica", action="store_true",
                   help="remove the single most blink-correlated ICA component per "
                        "subject before referencing (experimental — only 4 channels "
                        "available, so separation is weak)")
    p.add_argument("--prefilter", dest="prefilter", action="store_true", default=True,
                   help="band-pass the CONTINUOUS raw signal into each band "
                        "before epoching, instead of filtering each short "
                        "epoch independently inside HyPyP (default: on). "
                        "Reduces filter edge/transient bias for narrow bands.")
    p.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                   help="disable --prefilter and restore old per-epoch "
                        "narrowband filtering inside HyPyP (for comparison).")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    freq_bands = dict(FREQ_BANDS)
    if args.stim_hz is not None:
        half = args.stim_bandwidth / 2
        stim_band_name = f"stim_{args.stim_hz:g}hz"
        freq_bands[stim_band_name] = (args.stim_hz - half, args.stim_hz + half)
        if stim_band_name not in args.bands:
            args.bands = list(args.bands) + [stim_band_name]
        print(f"  positive-control band added: {stim_band_name} "
              f"({freq_bands[stim_band_name][0]:.2f}-{freq_bands[stim_band_name][1]:.2f} Hz)")
        if args.pool_dir is None and args.surrogate > 0:
            print("  NOTE: validating a stimulus-locked band with within-dyad "
                  "--surrogate only. For a continuous/stationary flicker this "
                  "test is weak (see module docstring) -- consider also "
                  "passing --pool-dir with other subjects' recordings.")

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

    # ------------------------------------------------------------------
    # Optionally load a pool of OTHER subjects' recordings for pseudo-pair
    # (cross-dyad) validation. Each pool file goes through the SAME load ->
    # preprocess -> epoch chain as the two main subjects, so the comparison
    # is fair.
    # ------------------------------------------------------------------
    pool_epochs = []
    if args.pool_dir:
        print()
        print("="*60)
        print("LOADING POOL (for pseudo-pair / cross-dyad validation)")
        print("="*60)
        pool_files = sorted(
            glob.glob(os.path.join(args.pool_dir, "*.csv"))
        )
        # never pool the two real subjects' own files against themselves
        pool_files = [f for f in pool_files
                      if os.path.abspath(f) not in
                      (os.path.abspath(args.csv_a), os.path.abspath(args.csv_b))]
        if not pool_files:
            print(f"  WARNING: no CSVs found in {args.pool_dir} (or all "
                  "excluded as the real subjects) — pseudo-pair validation "
                  "will be skipped.")
        pool_amp_thresh_arg = (args.pool_amplitude_threshold
                               if args.pool_amplitude_threshold is not None
                               else args.amplitude_threshold)
        pool_amp_thresh = pool_amp_thresh_arg if pool_amp_thresh_arg > 0 else None
        print(f"  pool amplitude_threshold={pool_amp_thresh or 'disabled'} µV "
              f"{'(same as main dyad)' if args.pool_amplitude_threshold is None else '(override)'}")
        for f in pool_files:
            ep, fs_pool = load_and_epoch_subject(
                f, subject_label=f"POOL_{os.path.basename(f)}",
                epoch_len_s=args.epoch_len, overlap_s=args.epoch_overlap,
                h_freq=h_freq, amplitude_uv=pool_amp_thresh or 1e9,
                use_ica=args.ica, align_onset=True, quiet=False,
            )
            if ep is not None and fs_pool == fs_a:
                pool_epochs.append(ep)
            elif ep is not None:
                print(f"  WARNING: {f} has fs={fs_pool} != {fs_a}, skipping "
                      "(sampling rate must match for pooling)")
        print(f"  Pool ready: {len(pool_epochs)} usable recordings "
              f"(from {len(pool_files)} files found)")

    print()
    print("="*60)
    print("PLV + CIRCULAR CORRELATION PER BAND")
    print("="*60)
    if args.prefilter:
        print("  --prefilter is ON: continuous raw is band-passed BEFORE "
              "epoching for each band (reduces per-epoch filter edge bias).")
    else:
        print("  --prefilter is OFF: each short epoch is narrowband-filtered "
              "independently inside HyPyP (legacy behaviour).")

    plvs = {}
    ccs = {}
    p_values = {}
    cc_p_values = {}
    sig_masks = {}
    cc_sig_masks = {}
    summary_lines = [f"Hyperscanning summary  {datetime.now().isoformat()}"]
    summary_lines.append(f"  A: {args.csv_a}")
    summary_lines.append(f"  B: {args.csv_b}")
    summary_lines.append(f"  fs={fs_a:.0f} Hz, epochs={n_ep}, "
                         f"epoch_len={args.epoch_len}s")
    summary_lines.append(f"  prefilter={args.prefilter}  "
                         f"correction={args.correction}  "
                         f"pool_dir={args.pool_dir or '(none)'}")
    summary_lines.append("")

    for band_name in args.bands:
        if band_name not in freq_bands:
            print(f"  skipping unknown band: {band_name}")
            continue
        band = freq_bands[band_name]
        if band[1] >= nyq:
            print(f"  skipping {band_name} ({band[0]}-{band[1]} Hz): "
                  f"above Nyquist ({nyq:.1f} Hz)")
            continue

        # -- build (possibly pre-filtered) epochs for THIS band --------
        if args.prefilter:
            raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
            raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
            epochs_a_band = epoch_with_gap_rejection(
                raw_a_band, args.epoch_len, args.epoch_overlap,
                amplitude_uv=amp_thresh or 1e9)
            epochs_b_band = epoch_with_gap_rejection(
                raw_b_band, args.epoch_len, args.epoch_overlap,
                amplitude_uv=amp_thresh or 1e9)
            common_band = np.intersect1d(epochs_a_band.selection, epochs_b_band.selection)
            epochs_a_band = epochs_a_band[np.isin(epochs_a_band.selection, common_band)]
            epochs_b_band = epochs_b_band[np.isin(epochs_b_band.selection, common_band)]
            already_filtered = True
        else:
            epochs_a_band, epochs_b_band = epochs_a, epochs_b
            already_filtered = False

        if len(epochs_a_band) == 0 or len(epochs_b_band) == 0:
            print(f"  {band_name}: 0 epochs survive after band-specific "
                  "filtering/rejection — skipping")
            continue

        plv = plv_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                         already_filtered=already_filtered)
        plvs[band_name] = plv
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean PLV = {plv.mean():.3f}  max = {plv.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"plv_{band_name}.npy"), plv)

        cc = circular_corr_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                                  already_filtered=already_filtered)
        ccs[band_name] = cc
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean circ-r = {cc.mean():.3f}  min = {cc.min():.3f}  max = {cc.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"circ_corr_{band_name}.npy"), cc)

        # ---------------- WITHIN-DYAD surrogate (optional) -------------
        if args.surrogate > 0:
            print(f"     running {args.surrogate} within-dyad surrogates (PLV)...")
            null_plv = surrogate_distribution(
                epochs_a_band, epochs_b_band, band, args.surrogate,
                metric_fn=plv_hypyp, sfreq=fs_a, already_filtered=already_filtered,
            )
            p_val = (null_plv >= plv[None, :, :]).mean(axis=0)
            p_values[band_name] = p_val

            print(f"     running {args.surrogate} within-dyad surrogates (circ-corr)...")
            null_cc = surrogate_distribution(
                epochs_a_band, epochs_b_band, band, args.surrogate,
                metric_fn=circular_corr_hypyp, sfreq=fs_a, already_filtered=already_filtered,
            )
            cc_p_val = (np.abs(null_cc) >= np.abs(cc[None, :, :])).mean(axis=0)
            cc_p_values[band_name] = cc_p_val

            if args.correction == "fdr":
                sig_mask, p_corrected = fdr_bh(p_val)
                cc_sig_mask, cc_p_corrected = fdr_bh(cc_p_val)
                n_sig = int(sig_mask.sum())
                n_sig_cc = int(cc_sig_mask.sum())
                n_sig_raw = int((p_val < 0.05).sum())
                n_sig_cc_raw = int((cc_p_val < 0.05).sum())
                line = (f"     PLV significant pairs: {n_sig}/{plv.size} "
                        f"(FDR-corrected)   [{n_sig_raw}/{plv.size} raw p<0.05, uncorrected]")
                print(line)
                summary_lines.append(line)
                line = (f"     circ-corr significant pairs: {n_sig_cc}/{cc.size} "
                        f"(FDR-corrected)   [{n_sig_cc_raw}/{cc.size} raw p<0.05, uncorrected]")
                print(line)
                summary_lines.append(line)
                sig_masks[band_name] = sig_mask
                cc_sig_masks[band_name] = cc_sig_mask
            else:
                n_sig = int((p_val < 0.05).sum())
                n_sig_cc = int((cc_p_val < 0.05).sum())
                line = f"     PLV significant pairs (p<0.05, UNCORRECTED): {n_sig}/{plv.size}"
                print(line)
                summary_lines.append(line)
                line = f"     circ-corr significant pairs (p<0.05, UNCORRECTED): {n_sig_cc}/{cc.size}"
                print(line)
                summary_lines.append(line)

            np.save(os.path.join(out_dir, f"plv_p_within_{band_name}.npy"), p_val)
            np.save(os.path.join(out_dir, f"circ_corr_p_within_{band_name}.npy"), cc_p_val)

        # ---------------- CROSS-DYAD pseudo-pair null (preferred) ------
        if pool_epochs:
            print(f"     running pseudo-pair null against {len(pool_epochs)} "
                  f"pool recordings x{args.pool_shuffles} draws (PLV)...")
            # combine "A vs pool" and "B vs pool" for a fuller null
            null_plv_pool = []
            for target in (epochs_a_band, epochs_b_band):
                res = pseudo_pair_distribution(
                    target, pool_epochs, band, plv_hypyp, sfreq=fs_a,
                    already_filtered=already_filtered,
                    shuffles_per_pool_member=args.pool_shuffles,
                )
                if res is not None:
                    null_plv_pool.append(res)
            if null_plv_pool:
                null_plv_pool = np.concatenate(null_plv_pool, axis=0)
                p_val_pool = (null_plv_pool >= plv[None, :, :]).mean(axis=0)
                if args.correction == "fdr":
                    sig_mask_pool, _ = fdr_bh(p_val_pool)
                    n_sig_pool = int(sig_mask_pool.sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair, "
                            f"FDR-corrected): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                else:
                    n_sig_pool = int((p_val_pool < 0.05).sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair, "
                            f"UNCORRECTED): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                print(line)
                summary_lines.append(line)
                np.save(os.path.join(out_dir, f"plv_p_pool_{band_name}.npy"), p_val_pool)
                line = (f"     Interpretation: real PLV={plv.mean():.3f} vs "
                        f"pool (independent, same-stimulus) PLV={null_plv_pool.mean():.3f} "
                        f"-> {'ABOVE pool baseline' if plv.mean() > null_plv_pool.mean() else 'NOT above pool baseline'}")
                print(line)
                summary_lines.append(line)

        plot_plv_matrix(
            plv, band_name,
            os.path.join(out_dir, f"plv_interbrain_{band_name}.png"),
            surrogate_p=p_values.get(band_name),
            sig_mask=sig_masks.get(band_name),
        )
        plot_circ_corr_matrix(
            cc, band_name,
            os.path.join(out_dir, f"circ_corr_{band_name}.png"),
            surrogate_p=cc_p_values.get(band_name),
            sig_mask=cc_sig_masks.get(band_name),
        )

    if plvs:
        plot_plv_comparison(plvs, os.path.join(out_dir, "plv_comparison.png"))
    if ccs:
        plot_circ_corr_comparison(ccs, os.path.join(out_dir, "circ_corr_comparison.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print()
    print(f"Wrote outputs to: {out_dir}")
    print("  plv_<band>.npy                - PLV matrices")
    print("  plv_interbrain_<band>.png     - PLV heatmaps (0 to 1); * = FDR-sig, (*) = uncorrected only")
    print("  plv_comparison.png            - PLV all bands side by side")
    print("  circ_corr_<band>.npy          - circular correlation matrices")
    print("  circ_corr_<band>.png          - circular corr heatmaps (-1 to 1)")
    print("  circ_corr_comparison.png      - circular corr all bands side by side")
    print("  plv_p_within_<band>.npy       - within-dyad surrogate p-values (if --surrogate)")
    print("  plv_p_pool_<band>.npy         - cross-dyad pseudo-pair p-values (if --pool-dir)")
    print("  raw_with_gaps.png             - signal + gap markers")
    print("  psd.png                       - power spectrum QC")
    print("  summary.txt                   - numerical summary")


if __name__ == "__main__":
    main()