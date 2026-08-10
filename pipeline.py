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
              continuous sliding-window artifact detection (default, see
              point 6 below) -- OR fixed-length epochs (--legacy-epochs,
              see point 5)
                   |
                   v
              PLV + circular correlation per frequency band (theta, alpha, beta)
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
    python pipeline.py a.csv b.csv                           # default: continuous sliding-window artifact rejection (see point 6 below)
    python pipeline.py a.csv b.csv --legacy-epochs           # old fixed-epoch pipeline (see point 5), for comparison
    python pipeline.py a.csv b.csv --bands alpha             # one band only
    python pipeline.py a.csv b.csv --surrogate 100           # within-dyad null (epoch shuffle)
    python pipeline.py a.csv b.csv --pool-dir recordings/other_dyads/   # cross-dyad null (pseudo-pairs)
    python pipeline.py a.csv b.csv --stim-hz 6.0 --pool-dir recordings/other_dyads/
        # positive control: both subjects watched the SAME 6 Hz flicker.
        # Raw magnitude here is expected to be high but AMBIGUOUS -- see
        # point 4 below. Needs --pool-dir (not just --surrogate) to mean
        # anything, since a constant stimulus barely responds to epoch shuffling.
    python pipeline.py a.csv b.csv --tag-hz-a 6.0 --tag-hz-b 8.0 --surrogate 100 --pool-dir recordings/other_dyads/
        # frequency-tagging: A watched 6 Hz, B watched 8 Hz (two monitors,
        # see make_checkerboard.py --pos). Unambiguous cross-brain test --
        # see point 4 below.

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

4. FREQUENCY-TAGGING support (--tag-hz-a / --tag-hz-b).
   Even a validated --stim-hz positive control (both subjects watching the
   SAME flicker) is fundamentally ambiguous: elevated inter-brain PLV/circ-
   corr there is exactly what you'd see BOTH if the two people are really
   coupled to each other AND if they're simply two independent visual
   systems separately locking onto one shared external clock. A shuffle
   test can't tell these apart for a constant, non-varying stimulus (see
   point 1), and even a --pool-dir pseudo-pair test only bounds the size of
   the shared-clock effect -- it doesn't eliminate the ambiguity in
   principle.
   Frequency-tagging removes the ambiguity by construction: give subject A
   and subject B DIFFERENT reversal rates (needs two monitors -- see --pos
   in make_checkerboard.py). Subject B is never driven at A's frequency, so
   any inter-brain coupling at A's tag frequency cannot be explained by "both
   locked to the same clock" -- there was no clock at that frequency for B.
   This is the strongest single test available and should supersede --stim-hz
   once you have a two-monitor setup; --stim-hz remains useful only as a
   cheap single-monitor sanity check that the measurement pipeline is
   sensitive at all.

5. DEFAULT EPOCH LENGTH increased from 2s to 30s (--epoch-len), overlap
   from 1s to 0s (--epoch-overlap).
   Short epochs inflate PLV/circ-corr: on one real dyad recording, mean
   theta PLV dropped from 0.286 (2s epochs) to 0.131 (10s) to 0.075 (30s)
   to 0.022 (one ~180s window covering the whole recording) -- a ~10x
   difference in the SAME data depending only on epoch length, exactly the
   bias reported by Zimmermann et al. (2024) and Cassioli et al. (2025),
   who estimate ~55s (+20% for artifact loss) of clean data is needed for
   adequately powered detection of a moderate effect at 256 Hz. 30s was
   chosen as the new default over 55s because at 150uV amplitude
   rejection, whole-epoch rejection makes very long epochs fragile (a
   single blink anywhere in the window drops the whole window); 30s
   still yields multiple epochs per session at typical recording lengths.
   Pass --epoch-len 2.0 --epoch-overlap 1.0 to restore the old behavior.

6. FIXED EPOCHS REPLACED BY CONTINUOUS SLIDING-WINDOW ARTIFACT REJECTION,
   as the new default (--legacy-epochs restores the point-5 fixed-epoch
   pipeline).
   Suggested by Evan (project supervisor, 2026-07-29): even the 30s fixed
   epochs from point 5 have the same flaw at a different scale -- a single
   1s blink inside a 30s window throws out the OTHER 29s of perfectly good
   data too, since MNE's epoch rejection is all-or-nothing per epoch.
   Instead, continuous_bad_mask() slides a short window (500ms, 100ms step
   by default -- see --artifact-window/--artifact-step/--artifact-threshold)
   across each subject's continuous signal and marks only the actual bad
   stretch, in BOTH subjects, at the matching timepoints (PLV/circ-corr need
   paired samples from both brains at the same instant). The Hilbert
   transform (instantaneous phase) is computed ONCE on the full continuous
   band-passed recording -- so there's no splicing discontinuity anywhere --
   and only afterward are the jointly-clean samples selected for the PLV /
   circular-correlation sum. See plv_masked()/circ_corr_masked() and the
   "3a2. CONTINUOUS (MASKED) CONNECTIVITY" section below; this bypasses
   HyPyP's epoch-based API, which has no way to mask out individual
   timepoints within an epoch.

7b. NULL-DISTRIBUTION SAMPLE-SIZE LOGGING + LENGTH MATCHING (--match-null-length).
   Diagnosed 2026-08-07: a real dyad's joint-clean window can be MUCH longer
   than what individual pseudo-pair (--pool-dir) draws end up with. E.g. one
   run had a real dyad joint good_mask of 372.5s (95,372 samples @ 256 Hz),
   but pseudo_pair_continuous() intersects the real subject's bad-mask with
   EACH POOL MEMBER'S OWN bad-mask after a random circular shift -- if a pool
   member's clean stretches don't line up well with the target's after
   shifting, that intersection can leave far less usable overlap per draw.
   Per Zimmermann et al. (2024), SHORTER windows systematically INFLATE PLV
   for weakly/uncoupled signals -- so a null built from short pool draws can
   end up biased upward relative to a long, well-converged real value, making
   real coupling look "below baseline" for a reason that has nothing to do
   with actual inter-brain coupling: an N mismatch between the real value and
   its null.
   circular_shift_surrogates_continuous() and pseudo_pair_continuous() now
   both return the per-draw sample count (Ns) alongside the null values, and
   this is logged to console + summary.txt regardless of anything else, so
   an N mismatch like the one above is visible without needing to add ad hoc
   print statements.
   --match-null-length (default: on) additionally computes a LENGTH-MATCHED
   comparison: a target sample count is chosen as the smaller of (a) the
   real dyad's own good_mask size and (b) a robust (10th percentile) size
   across the null draws, floored at --min-null-seconds. Both the null draws
   AND a matched "observed" value (averaged over several random subsamples
   of the real dyad's good_mask down to that same target size) are then
   recomputed at that common N, so the p-value comparison is apples-to-
   apples. The ORIGINAL full-length real PLV/circ-r (computed on ALL
   available clean data, still the best point estimate) is still reported
   unchanged as the headline number -- only the null-comparison/p-value uses
   the length-matched version. Use --no-match-null-length to restore the old
   (potentially N-mismatched) behaviour.

7. ADJUSTED CIRCULAR CORRELATION now implemented in the continuous path too.
   Point 6 originally shipped with a caveat: the continuous path fell back
   to the classic Fisher & Lee (1983) circular correlation because the
   bias-adjusted CCorradj formula from Zimmermann et al. (2024) "wasn't
   confidently reproducible from the secondary source available." That
   formula is now confirmed against a known, tested, open-source
   implementation: pingouin.circ_corrcc(correction_uniform=True) (Vallat,
   2018), whose source directly implements Jammalamadaka & Sengupta (2001,
   p.177):

       r_minus = |sum_t exp(i*(phase_a_t - phase_b_t))|
       r_plus  = |sum_t exp(i*(phase_a_t + phase_b_t))|
       denom   = 2 * sqrt(sum(sin(phase_a - mean_a)^2) * sum(sin(phase_b - mean_b)^2))
       r_adj   = (r_minus - r_plus) / denom

   circ_corr_adjusted_masked() below is a vectorized, sample-masked version
   of exactly this formula (verified against pingouin's scalar
   implementation to float precision). This is the SAME correction
   HyPyP's 'accorr' mode already applies in the --legacy-epochs path, so
   both paths now report a consistent, literature-matched measure instead
   of two different circular correlation definitions. The classic
   (non-adjusted) version is kept as circ_corr_masked() and is still
   available via --circ-corr-method classic for comparison; adjusted is
   now the default (--circ-corr-method adjusted), consistent with
   Zimmermann et al.'s recommendation that adjusted circular correlation
   should be preferred for continuous EEG data, which does not have a
   well-defined circular mean.
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
    separation is weak compared to a real multi-channel montage -- at most
    the single most blink-correlated component is removed, to avoid
    discarding real brain signal along with the artifact.
    """
    proxy_ch = next((ch for ch in raw.ch_names if ch.endswith("AF7") or ch.endswith("AF8")), None)
    if proxy_ch is None:
        print(f"     {subject_label}: ICA skipped -- no frontal channel found for blink proxy")
        return raw

    ica = ICA(max_iter="auto", random_state=random_state)
    ica.fit(raw, verbose=False)

    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=proxy_ch, verbose=False)
    except Exception as e:
        print(f"     {subject_label}: ICA blink-detection via {proxy_ch} failed ({e}) -- no components removed")
        return raw

    if not eog_indices:
        print(f"     {subject_label}: ICA found no clear blink component -- no components removed")
        return raw

    ica.exclude = eog_indices[:1]
    print(f"     {subject_label}: ICA removed component {ica.exclude} (blink-correlated via {proxy_ch})")
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


def detect_bad_channels(raw, railed_uv=990.0, railed_frac_thresh=0.02, flat_std_uv=5.0):
    """
    Flag channels that are chronically railed (stuck near the Muse's ADC
    limit -- poor scalp contact or motion) or abnormally flat (near-zero
    variance -- a loose/disconnected electrode). Checked on the raw
    (pre-filter) signal, since railing is clearest there: band-pass
    filtering smooths sharp rail transitions and can hide them.

    railed_uv=990 sits just inside the Muse's +/-1000 uV rail. flat_std_uv=5
    is well below typical scalp EEG (tens of uV), so it only catches
    channels that are essentially dead, not just quiet.

    Returns a list of channel names to mark bad.
    """
    data_uv = raw.get_data(picks="eeg") * 1e6
    bad = []
    for ch_name, ch_data in zip(raw.ch_names, data_uv):
        railed_frac = float((np.abs(ch_data) >= railed_uv).mean())
        std_uv = float(ch_data.std())
        if railed_frac > railed_frac_thresh:
            print(f"     {ch_name}: railed {railed_frac:.1%} of samples "
                  f"(>= {railed_uv:.0f} uV) -- marking bad")
            bad.append(ch_name)
        elif std_uv < flat_std_uv:
            print(f"     {ch_name}: abnormally flat (std={std_uv:.1f} uV, "
                  f"< {flat_std_uv:.0f} uV) -- marking bad")
            bad.append(ch_name)
    return bad


def preprocess(raw, l_freq=1.0, h_freq=40.0, use_ica=False, subject_label=""):
    """
    Bandpass + optional ICA blink removal + average reference.

    Bad channels (see detect_bad_channels) are excluded from the reference
    computation -- a chronically railed or dead channel would otherwise
    drag the shared average down (or up) with it and contaminate the other
    3, otherwise-good, channels for that subject.
    """
    raw = raw.copy()
    bad_chs = detect_bad_channels(raw)
    raw.info["bads"] = bad_chs
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    if use_ica:
        raw = remove_blink_component(raw, subject_label)
    good_chs = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    if not good_chs:
        print(f"     {subject_label}: WARNING all channels flagged bad -- "
              "falling back to plain average reference")
        raw.set_eeg_reference("average", projection=False, verbose=False)
    else:
        if bad_chs:
            print(f"     {subject_label}: referencing to good channels only "
                  f"{good_chs} (excluding {bad_chs})")
        # average reference computed from good channels only, applied to
        # that subject's channels (MNE leaves bad channels' own data
        # unreferenced, which is fine -- their values are already flagged
        # unreliable downstream)
        raw.set_eeg_reference(ref_channels=good_chs, projection=False, verbose=False)
    return raw


def epoch_with_gap_rejection(raw, epoch_len_s, overlap_s, amplitude_uv=150.0):
    """
    Fixed-length epochs, rejecting any epoch that:
      - overlaps a BAD_gap annotation (lost BLE packets), OR
      - has peak-to-peak amplitude > amplitude_uv on any channel (muscle artifact)

    150 uV is a standard threshold for consumer EEG. Lower it (e.g. 100) for
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

    Legacy path (--legacy-epochs) only -- see load_and_preprocess_continuous
    for the default, continuous artifact-rejection path.

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
            print(f"  WARNING: {csv_path} produced 0 usable epochs -- skipping")
        return None, fs
    return epochs, fs


# ============================================================
# 2b. CONTINUOUS ARTIFACT DETECTION (default path, no fixed epochs)
# ============================================================
#
# Suggested by Evan (project supervisor, 2026-07-29): fixed-length epochs
# (even the 30s ones from the previous fix) get thrown out ENTIRELY if even
# a small fraction is bad -- e.g. 1s of blink inside a 30s window discards
# 29s of perfectly good data. A short sliding window instead marks and
# removes only the actual bad stretch, in BOTH subjects at the matching
# timepoints (since PLV/circ-corr need paired samples from both brains at
# the same instant), and the connectivity stats run on whatever continuous
# good data is left -- no arbitrary epoch boundaries at all.

def continuous_bad_mask(raw, window_s=0.5, step_s=0.1, threshold_uv=500.0, pad_s=0.3):
    """
    Slide a window across a subject's continuous signal; mark every sample
    covered by a window as bad if any channel's peak-to-peak amplitude in
    that window exceeds threshold_uv. Also folds in existing BAD_gap
    annotations (lost BLE packets), so both artifact types end up in one
    boolean mask over samples.

    Each resulting bad run is then padded by pad_s on both sides. This
    matters because the band-pass filter applied to the continuous signal
    (see prefilter_raw_for_band) can ring for a short distance around a
    sharp artifact edge (e.g. a railed/saturated segment) -- without
    padding, that ringing can leak into samples just outside the detected
    bad run and get silently counted as clean data.
    """
    data_uv = raw.get_data(picks="eeg") * 1e6
    sfreq = raw.info["sfreq"]
    n_samples = data_uv.shape[1]
    window_n = max(1, int(round(window_s * sfreq)))
    step_n = max(1, int(round(step_s * sfreq)))

    bad = np.zeros(n_samples, dtype=bool)
    starts = list(range(0, max(1, n_samples - window_n + 1), step_n))
    last_start = max(0, n_samples - window_n)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = min(start + window_n, n_samples)
        seg = data_uv[:, start:end]
        ptp = seg.max(axis=1) - seg.min(axis=1)
        if np.any(ptp > threshold_uv):
            bad[start:end] = True

    for ann in raw.annotations:
        if "BAD" in ann["description"]:
            onset_samp = int(round(ann["onset"] * sfreq))
            dur_samp = int(round(ann["duration"] * sfreq))
            bad[max(0, onset_samp):max(0, onset_samp) + dur_samp] = True

    pad_n = int(round(pad_s * sfreq))
    if pad_n > 0 and bad.any():
        kernel = np.ones(2 * pad_n + 1, dtype=np.uint8)
        bad = np.convolve(bad.astype(np.uint8), kernel, mode="same") > 0

    return bad


def load_and_preprocess_continuous(csv_path, subject_label, h_freq=40.0,
                                    use_ica=False, align_onset=True, quiet=False,
                                    artifact_window=0.5, artifact_step=0.1,
                                    artifact_threshold=500.0, artifact_pad=0.3):
    """
    Full load -> preprocess chain for ONE subject's CSV, for the default
    continuous (non-epoched) connectivity path.

    Returns (raw_pp, bad_mask, fs) or (None, None, None) if the file
    couldn't be used.
    """
    try:
        onset = load_stimulus_onset(csv_path) if align_onset else None
        raw, fs = load_csv_to_raw(csv_path, subject_label, onset_s=onset)
    except Exception as e:
        if not quiet:
            print(f"  WARNING: could not load {csv_path}: {e}")
        return None, None, None

    nyq = fs / 2
    h_freq_eff = min(h_freq, nyq * 0.95)
    raw_pp = preprocess(raw, h_freq=h_freq_eff, use_ica=use_ica, subject_label=subject_label)
    bad_mask = continuous_bad_mask(raw_pp, window_s=artifact_window,
                                    step_s=artifact_step,
                                    threshold_uv=artifact_threshold,
                                    pad_s=artifact_pad)
    return raw_pp, bad_mask, fs


# ============================================================
# 3. CONNECTIVITY (PLV / circular correlation)
# ============================================================

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


# ============================================================
# 2c. CROSS-DEVICE TIMING SYNC-CORRECTION (optional, --sync-signal-hz)
# ============================================================
#
# Two independent Muses stream over BLE, and LSL dejitters each stream onto
# a nominal 256 Hz grid -- which hides any small difference between the two
# devices' crystal-clock rates. That difference shows up as a slow, HIDDEN
# drift in the sample-by-sample alignment between the two recordings (tens
# of ms over a several-minute session), which erodes inter-brain phase
# coupling, worst in high-frequency bands (beta). It can only pull real
# coupling DOWN toward the null -- it never manufactures coupling -- so it
# is a sensitivity ceiling, not a false-positive risk.
#
# If BOTH subjects were exposed to the SAME external periodic driver (e.g. a
# shared flicker at --sync-signal-hz, picked up as an entrained SSVEP, or a
# photodiode/trigger channel), that shared signal is a timing ruler: both
# brains are driven by one clock, so their phase difference AT THE DRIVER
# FREQUENCY should be constant if the devices are aligned. A linear ramp in
# that phase difference over time IS the clock drift, and its slope gives
# the exact rate correction.
#
# IMPORTANT LIMITATION: at a SINGLE frequency, a constant time offset is
# indistinguishable from a constant NEURAL phase lag between the two
# subjects' responses (and wraps every 1/f seconds). So the DRIFT (slope)
# is unambiguous and safe to correct; the constant OFFSET (intercept) is
# confounded and is only corrected when explicitly requested (not
# --sync-drift-only), and even then is best sanity-checked against a
# positive control. To disambiguate the constant offset properly you need
# two driver frequencies (frequency-tagging) or a broadband shared trigger.

def _sync_reference_phase(raw, band):
    """Instantaneous phase of the strongest shared-driver component: pick
    the channel with the most power in `band`, band-pass, Hilbert."""
    from scipy.signal import hilbert
    raw_band = prefilter_raw_for_band(raw, band)
    data = raw_band.get_data(picks="eeg")
    idx = int(np.argmax(data.std(axis=1)))
    return np.angle(hilbert(data[idx]))


def estimate_sync_offset_drift(raw_a, raw_b, sync_hz, bandwidth=1.0, drift_only=False):
    """
    Estimate the cross-device timing offset (s) and drift rate (dimensionless
    s/s) between two recordings, from a shared periodic driver at sync_hz.

    Fits phase_a - phase_b = -2*pi*sync_hz*(offset + drift*t) over time:
      intercept -> constant offset,  slope -> drift rate.
    If drift_only, the constant offset is forced to 0 (only drift, the
    unambiguous part, is returned -- see the section note above).

    Returns (offset_s, drift_rate).
    """
    band = (sync_hz - bandwidth / 2.0, sync_hz + bandwidth / 2.0)
    fs = raw_a.info["sfreq"]
    pa = _sync_reference_phase(raw_a, band)
    pb = _sync_reference_phase(raw_b, band)
    n = min(len(pa), len(pb))
    t = np.arange(n) / fs
    dphi = np.unwrap(pa[:n] - pb[:n])
    w = 2.0 * np.pi * sync_hz
    if drift_only:
        slope = np.polyfit(t, dphi, 1)[0]
        return 0.0, -slope / w
    slope, intercept = np.polyfit(t, dphi, 1)
    return -intercept / w, -slope / w


def apply_sync_correction(raw_b, offset_s, drift_rate):
    """
    Resample raw_b onto raw_a's timeline given an estimated constant offset
    and linear drift (see estimate_sync_offset_drift). Sample i of the
    corrected signal is raw_b interpolated at grid time
    (t_i - offset)/(1 + drift). Returns a corrected copy; length unchanged.
    """
    fs = raw_b.info["sfreq"]
    data = raw_b.get_data()
    n = data.shape[1]
    grid = np.arange(n)
    src_idx = ((grid / fs - offset_s) / (1.0 + drift_rate)) * fs
    corrected = np.vstack([np.interp(src_idx, grid, data[ch]) for ch in range(data.shape[0])])
    raw_out = raw_b.copy()
    raw_out._data[:] = corrected
    return raw_out


def _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode, sfreq,
                               already_filtered=False):
    """
    Compute an inter-brain connectivity block with HyPyP.

    If already_filtered=True, the epochs' data is assumed to already be
    band-passed (via prefilter_raw_for_band on the continuous raw signal
    before epoching), and HyPyP is told not to filter again
    (filter_signal=False) -- it will just compute the analytic (Hilbert)
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
# 3a2. CONTINUOUS (MASKED) CONNECTIVITY -- default path, no fixed epochs
# ============================================================
#
# Companion to section 2b: instead of chopping the recording into fixed
# epochs and rejecting whole ones, the Hilbert transform is computed ONCE
# on the FULL continuous band-passed recording (so there's no splicing
# discontinuity anywhere), and only afterward do we select just the
# samples that are clean in BOTH subjects for the actual PLV / circular
# correlation sum. This bypasses HyPyP's epoch-based API (which has no way
# to mask out individual timepoints within an epoch), so PLV and the
# circular correlation coefficient are computed directly here instead.
#
# TWO circular correlation variants are provided:
#   circ_corr_masked()           classic Fisher & Lee (1983); assumes a
#                                 well-defined circular mean per channel.
#   circ_corr_adjusted_masked()  bias-adjusted version for arbitrary/not-
#                                 well-defined mean directions (Jammalamadaka
#                                 & Sengupta, 2001, p.177), the same
#                                 correction Zimmermann et al. (2024) use and
#                                 recommend for continuous EEG, and the same
#                                 one HyPyP's 'accorr' mode applies in the
#                                 --legacy-epochs path. This is now the
#                                 default (see --circ-corr-method).

def analytic_signal(raw_band, n_samples=None):
    """Hilbert transform of a band-passed CONTINUOUS Raw -> complex analytic
    signal per channel, shape (n_channels, n_times)."""
    from scipy.signal import hilbert
    data = raw_band.get_data(picks="eeg")
    if n_samples is not None:
        data = data[:, :n_samples]
    return hilbert(data, axis=-1)


def plv_masked(analytic_a, analytic_b, good_mask):
    """PLV between every channel pair, summed over good_mask samples only.
    Returns (n_chan_a, n_chan_b)."""
    phase_a = np.angle(analytic_a[:, good_mask])
    phase_b = np.angle(analytic_b[:, good_mask])
    diff = phase_a[:, None, :] - phase_b[None, :, :]
    return np.abs(np.mean(np.exp(1j * diff), axis=-1))


def circ_corr_masked(analytic_a, analytic_b, good_mask):
    """Classic Fisher & Lee (1983) circular correlation coefficient between
    every channel pair, summed over good_mask samples only. Signed:
    positive = in-phase-leaning, negative = anti-phase-leaning. Assumes each
    channel has a well-defined circular mean -- for continuous EEG this is
    often NOT the case (see circ_corr_adjusted_masked below and Zimmermann
    et al., 2024). Returns (n_chan_a, n_chan_b)."""
    phase_a = np.angle(analytic_a[:, good_mask])
    phase_b = np.angle(analytic_b[:, good_mask])
    mean_a = np.angle(np.mean(np.exp(1j * phase_a), axis=-1))
    mean_b = np.angle(np.mean(np.exp(1j * phase_b), axis=-1))
    sin_a = np.sin(phase_a - mean_a[:, None])
    sin_b = np.sin(phase_b - mean_b[:, None])
    num = np.einsum("it,jt->ij", sin_a, sin_b)
    den = np.sqrt(np.sum(sin_a ** 2, axis=-1)[:, None] * np.sum(sin_b ** 2, axis=-1)[None, :])
    return num / den


def circ_corr_adjusted_masked(analytic_a, analytic_b, good_mask):
    """
    Bias-adjusted circular correlation coefficient between every channel
    pair, summed over good_mask samples only. Signed, same sign convention
    as circ_corr_masked.

    Implements Jammalamadaka & Sengupta (2001, p.177), the formula
    Zimmermann et al. (2024) recommend for continuous EEG data (whose
    circular mean, unlike directional data such as wind bearings, is not
    well-defined over an arbitrary analysis window). This is the exact
    formula used by pingouin.circ_corrcc(correction_uniform=True) and by
    HyPyP's 'accorr' mode (used in the --legacy-epochs path), so this
    continuous-path implementation is now consistent with both:

        r_minus = |sum_t exp(i*(phase_a_t - phase_b_t))|
        r_plus  = |sum_t exp(i*(phase_a_t + phase_b_t))|
        denom   = 2 * sqrt(sum(sin(phase_a - mean_a)^2)
                            * sum(sin(phase_b - mean_b)^2))
        r_adj   = (r_minus - r_plus) / denom

    Verified against pingouin's scalar implementation to float precision
    before being vectorized here for the (n_chan_a, n_chan_b) pairwise case.

    Returns (n_chan_a, n_chan_b).
    """
    phase_a = np.angle(analytic_a[:, good_mask])  # (n_chan_a, n_t)
    phase_b = np.angle(analytic_b[:, good_mask])  # (n_chan_b, n_t)

    mean_a = np.angle(np.sum(np.exp(1j * phase_a), axis=-1))
    mean_b = np.angle(np.sum(np.exp(1j * phase_b), axis=-1))
    sin_a = np.sin(phase_a - mean_a[:, None])
    sin_b = np.sin(phase_b - mean_b[:, None])

    ea = np.exp(1j * phase_a)  # (n_chan_a, n_t)
    eb = np.exp(1j * phase_b)  # (n_chan_b, n_t)
    r_minus = np.abs(np.einsum("it,jt->ij", ea, np.conj(eb)))
    r_plus = np.abs(np.einsum("it,jt->ij", ea, eb))

    denom = 2 * np.sqrt(np.sum(sin_a ** 2, axis=-1)[:, None]
                         * np.sum(sin_b ** 2, axis=-1)[None, :])
    return (r_minus - r_plus) / denom


def subsample_good_mask(good, target_n, rng):
    """
    Randomly select exactly target_n True positions from a boolean mask,
    returning a new boolean mask of the same length with only those
    positions set. If good.sum() <= target_n, returns good unchanged
    (nothing to trim).

    Used to length-match the real dyad's (often much larger) good_mask, or
    an individual null draw's good_mask, down to a common sample count so
    PLV/circular-correlation comparisons aren't confounded by different
    N's -- short windows are known to inflate these metrics for weak/no
    coupling (Zimmermann et al., 2024), so comparing a long real estimate
    against a null built from shorter draws (or vice versa) is not a fair
    like-for-like test. Order doesn't matter for PLV/circular correlation
    (both are order-independent sums over samples), so random subsampling
    is valid here -- no need to preserve contiguous runs.
    """
    n_good = int(good.sum())
    if n_good <= target_n:
        return good
    idx = np.where(good)[0]
    chosen = rng.choice(idx, size=target_n, replace=False)
    out = np.zeros_like(good)
    out[chosen] = True
    return out


def circular_shift_surrogates_continuous(analytic_a, bad_a, analytic_b, bad_b,
                                          n_surrogates, metric_fn, seed=0,
                                          target_n=None, rng=None):
    """
    WITHIN-DYAD null for the continuous path: circularly time-shift subject
    B's ENTIRE analytic signal (and its own bad-mask) by a random amount,
    wrapping at the recording boundary, then recompute the joint good_mask
    against A's (unshifted) bad-mask. This is the standard continuous-
    signal surrogate method -- it preserves each subject's own temporal/
    spectral structure while destroying true moment-to-moment alignment
    between the two brains, playing the same role the old epoch-shuffle
    surrogate did before fixed epochs were removed.

    If target_n is given, only draws with at least target_n jointly-clean
    samples are kept, and each surviving draw's good_mask is randomly
    subsampled down to exactly target_n samples (see subsample_good_mask)
    before computing the metric. That makes the null comparison exact: every
    included draw is evaluated at the same N.

    Returns (nulls, ns): nulls is an array of per-draw metric matrices (or
    None if every draw had zero usable samples), ns is an array of the
    actual per-draw sample count used (before any target_n floor is hit,
    this equals the natural overlap size; useful for diagnosing N
    mismatches even when target_n/matching is off).
    """
    rng = rng or np.random.default_rng(seed)
    n = analytic_b.shape[-1]
    nulls = []
    ns = []
    for _ in range(n_surrogates):
        shift = int(rng.integers(1, n))
        b_shifted = np.roll(analytic_b, shift, axis=-1)
        bad_b_shifted = np.roll(bad_b, shift)
        good = ~(bad_a | bad_b_shifted)
        ns.append(int(good.sum()))
        if target_n is not None:
            if good.sum() < target_n:
                continue
            good = subsample_good_mask(good, target_n, rng)
        if good.sum() == 0:
            continue
        nulls.append(metric_fn(analytic_a, b_shifted, good))
    return (np.array(nulls) if nulls else None), np.array(ns)


def pseudo_pair_continuous(target_analytic, target_bad, pool_analytic_list, pool_bad_list,
                            metric_fn, shuffles_per_pool_member=3, seed=0,
                            target_n=None, rng=None):
    """
    CROSS-DYAD null for the continuous path: compare the target subject's
    continuous analytic signal against OTHER subjects' (--pool-dir)
    continuous analytic signal, at randomly shifted alignments, keeping
    only jointly-clean samples. See pseudo_pair_distribution() (legacy
    epoch-based version) for the rationale.

    Each pool member has their OWN bad-mask from their OWN recording
    session, so the size of the jointly-clean overlap after intersecting
    with the target's bad-mask varies draw to draw, and can be much smaller
    than the target's own (often much longer) good_mask. If target_n is
    given, only draws with at least target_n jointly-clean samples are kept,
    and each surviving draw is randomly subsampled down to exactly target_n
    samples before computing the metric (see subsample_good_mask), so the
    resulting null is length-matched and not biased by short-window PLV/
    circ-corr inflation relative to a longer real-dyad estimate.

    Returns (nulls, ns): nulls has shape (n_pool * shuffles_per_pool_member,
    n_chan, n_chan) or None if no pool member overlapped with clean data;
    ns is the array of actual per-draw sample counts (pre-target_n, so it
    reflects the natural overlap size for diagnostics even with matching on).
    """
    rng = rng or np.random.default_rng(seed)
    nulls = []
    ns = []
    for pool_analytic, pool_bad in zip(pool_analytic_list, pool_bad_list):
        n = min(target_analytic.shape[-1], pool_analytic.shape[-1])
        if n == 0:
            continue
        t_analytic = target_analytic[:, :n]
        t_bad = target_bad[:n]
        for _ in range(shuffles_per_pool_member):
            shift = int(rng.integers(0, pool_analytic.shape[-1]))
            p_analytic = np.roll(pool_analytic, shift, axis=-1)[:, :n]
            p_bad = np.roll(pool_bad, shift)[:n]
            good = ~(t_bad | p_bad)
            ns.append(int(good.sum()))
            if target_n is not None:
                if good.sum() < target_n:
                    continue
                good = subsample_good_mask(good, target_n, rng)
            if good.sum() == 0:
                continue
            nulls.append(metric_fn(t_analytic, p_analytic, good))
    if not nulls:
        return None, np.array(ns)
    return np.array(nulls), np.array(ns)


def matched_observed_value(analytic_a, analytic_b, good_mask, target_n, metric_fn,
                            n_draws=5, seed=0):
    """
    Recompute the observed PLV/circ-corr value on the real dyad, averaged
    over n_draws random subsamples of good_mask down to exactly target_n
    samples each (see subsample_good_mask).

    This is used ONLY for the null-comparison/p-value, so that the real
    value being compared against the null is at the SAME sample count as
    the null draws -- not for the headline PLV/circ-r number, which should
    still use ALL available clean data (lower variance, better point
    estimate). Averaging over several subsamples reduces the extra
    sampling noise introduced by only using target_n < the full available N.

    Returns the mean matrix over the n_draws subsamples.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        sub_mask = subsample_good_mask(good_mask, target_n, rng)
        if sub_mask.sum() == 0:
            continue
        draws.append(metric_fn(analytic_a, analytic_b, sub_mask))
    if not draws:
        return None
    return np.mean(draws, axis=0)


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
    ax.set_title(f"Inter-brain PLV -- {band_name}")
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
    ax.set_title(f"Inter-brain Circular Corr -- {band_name}")
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
    p.add_argument("--legacy-epochs", action="store_true",
                   help="use the OLD fixed-length-epoch pipeline (see "
                        "--epoch-len/--epoch-overlap/--amplitude-threshold "
                        "below) instead of the default continuous sliding-"
                        "window artifact rejection (see --artifact-window/"
                        "--artifact-step/--artifact-threshold). Fixed epochs "
                        "throw out the WHOLE epoch if even a small fraction "
                        "is bad (e.g. 1s of blink inside a 30s window wastes "
                        "29s of good data) -- kept only for reproducing/"
                        "comparing against older runs.")
    p.add_argument("--artifact-window", type=float, default=0.5,
                   help="(continuous/default path only) sliding window "
                        "length in seconds for artifact detection (default "
                        "0.5). A window is marked bad if any channel's "
                        "peak-to-peak amplitude inside it exceeds "
                        "--artifact-threshold.")
    p.add_argument("--artifact-step", type=float, default=0.1,
                   help="(continuous/default path only) step size in "
                        "seconds between sliding artifact-detection windows "
                        "(default 0.1).")
    p.add_argument("--artifact-threshold", type=float, default=500.0,
                   help="(continuous/default path only) peak-to-peak "
                        "amplitude threshold in uV for the sliding-window "
                        "artifact detector (default 500).")
    p.add_argument("--artifact-pad", type=float, default=0.3,
                   help="(continuous/default path only) seconds of margin "
                        "added on both sides of every detected bad run "
                        "before it's excluded (default 0.3), to absorb "
                        "band-pass filter ringing around artifact edges "
                        "(e.g. a railed/saturated segment) that would "
                        "otherwise leak into neighboring samples still "
                        "counted as clean. 0 = disabled (old behaviour).")
    p.add_argument("--sync-signal-hz", type=float, default=None,
                   help="(continuous/default path only) frequency (Hz) of a "
                        "shared external driver both subjects were exposed to "
                        "(e.g. a 6 Hz flicker seen as SSVEP, or a photodiode/"
                        "trigger channel). Used as a cross-device timing "
                        "ruler: the two Muses' clocks drift relative to each "
                        "other (hidden by LSL's nominal-rate dejittering), and "
                        "that drift erodes inter-brain phase coupling. The "
                        "shared driver's phase difference over time measures "
                        "and corrects it. Off by default. See --sync-drift-only.")
    p.add_argument("--sync-bandwidth", type=float, default=1.0,
                   help="full width in Hz of the band around --sync-signal-hz "
                        "used to extract the shared driver (default 1.0).")
    p.add_argument("--sync-drift-only", action="store_true",
                   help="correct only the cross-device DRIFT (slope), not the "
                        "constant offset. At a single driver frequency a "
                        "constant time offset is confounded with a genuine "
                        "neural phase lag (and wraps every 1/f s), so the "
                        "drift is the unambiguous, always-safe part to "
                        "correct; the constant offset is applied only when "
                        "this flag is absent.")
    p.add_argument("--epoch-len", type=float, default=30.0,
                   help="(--legacy-epochs only) epoch length in seconds "
                        "(default 30.0). Short epochs (e.g. 2s) inflate "
                        "PLV/circ-corr -- Zimmermann et al. (2024) and "
                        "Cassioli et al. (2025) report values stabilize "
                        "only past ~5-6s, with ~55s (+20%% for artifact "
                        "loss) needed for adequately powered detection of a "
                        "moderate effect at 256 Hz. Use --epoch-len 2.0 to "
                        "reproduce the old (inflated) behaviour for "
                        "comparison.")
    p.add_argument("--epoch-overlap", type=float, default=0.0,
                   help="(--legacy-epochs only) epoch overlap in seconds "
                        "(default 0.0, i.e. non-overlapping). Overlapping "
                        "long epochs share most of their data and are not "
                        "independent samples for epoch-averaging or "
                        "surrogate shuffling.")
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
                        "/ --tag-hz-a / --tag-hz-b (default 1.0, i.e. +/-0.5 Hz)")
    p.add_argument("--tag-hz-a", type=float, default=None,
                   help="FREQUENCY-TAGGING design: subject A's own SSVEP "
                        "reversal rate, when A and B watched DIFFERENT "
                        "flicker rates (needs two monitors -- see --pos in "
                        "make_checkerboard.py). Adds a narrow inter-brain band "
                        "at this frequency. Because B was never driven at "
                        "A's rate, any inter-brain PLV/circ-corr here can't "
                        "be explained by both brains locking onto one shared "
                        "external clock -- unlike --stim-hz, where both "
                        "subjects see the same flicker and elevated coupling "
                        "is ambiguous between 'real interpersonal effect' and "
                        "'two brains independently entrained to the same "
                        "signal'. Pass --tag-hz-b too for the full design.")
    p.add_argument("--tag-hz-b", type=float, default=None,
                   help="FREQUENCY-TAGGING design: subject B's own SSVEP "
                        "reversal rate. See --tag-hz-a.")
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
    p.add_argument("--match-null-length", dest="match_null_length",
                   action="store_true", default=True,
                   help="(continuous/default path only) length-match the "
                        "null distribution (--surrogate and/or --pool-dir) "
                        "against the real dyad's observed value before "
                        "computing p-values (default: on). Pool draws in "
                        "particular can end up with much less usable "
                        "overlap than the real dyad's own (often much "
                        "longer) joint-clean window, and short windows "
                        "inflate PLV/circ-corr for weak/no coupling -- "
                        "comparing a long real estimate to a null built "
                        "from short draws is not apples-to-apples. When on, "
                        "a common target sample count is chosen (see "
                        "--min-null-seconds) and both the null draws and a "
                        "matched copy of the observed value (averaged over "
                        "several random subsamples) are computed at that "
                        "N for the p-value comparison. The full-length "
                        "observed PLV/circ-r reported as the headline "
                        "number is unaffected -- only the null comparison "
                        "changes. Per-draw sample counts (Ns) are always "
                        "logged regardless of this flag.")
    p.add_argument("--no-match-null-length", dest="match_null_length",
                   action="store_false",
                   help="disable --match-null-length (restore old, "
                        "potentially N-mismatched null comparison).")
    p.add_argument("--min-null-seconds", type=float, default=10.0,
                   help="(continuous/default path only, --match-null-length) "
                        "floor, in seconds, on the length-matching target "
                        "sample count (default 10.0). Prevents matching down "
                        "to a degenerately short window if some pool draws "
                        "have very little usable overlap; draws shorter than "
                        "this are excluded from the target-size calculation "
                        "(they still appear in the logged N distribution).")
    p.add_argument("--pool-amplitude-threshold", type=float, default=None,
                   help="(--legacy-epochs only) peak-to-peak amplitude "
                        "threshold (uV) applied when epoching --pool-dir "
                        "recordings, separate from --amplitude-threshold. "
                        "Defaults to whatever --amplitude-threshold is. Pool "
                        "recordings often come from unrelated sessions with "
                        "different noise levels (e.g. a hardware test "
                        "recording) -- loosen this (or pass 0 to disable) "
                        "if they get rejected wholesale under the main "
                        "dyad's threshold.")
    p.add_argument("--correction", choices=["fdr", "none"], default="fdr",
                   help="multiple-comparisons correction across the 16 "
                        "channel pairs per band (default fdr). 'none' "
                        "restores the old raw p<0.05 counting.")
    p.add_argument("--amplitude-threshold", type=float, default=150.0,
                   help="(--legacy-epochs only) peak-to-peak amplitude "
                        "threshold in uV for epoch rejection (default 150). "
                        "Lower = stricter. 0 = disabled.")
    p.add_argument("--ica", action="store_true",
                   help="remove the single most blink-correlated ICA component per "
                        "subject before referencing (experimental -- only 4 channels "
                        "available, so separation is weak)")
    p.add_argument("--prefilter", dest="prefilter", action="store_true", default=True,
                   help="(--legacy-epochs only; the continuous/default path "
                        "always band-passes the continuous signal before the "
                        "Hilbert transform, since that step is required, not "
                        "optional, there) band-pass the CONTINUOUS raw "
                        "signal into each band before epoching, instead of "
                        "filtering each short epoch independently inside "
                        "HyPyP (default: on). Reduces filter edge/transient "
                        "bias for narrow bands.")
    p.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                   help="(--legacy-epochs only) disable --prefilter and restore old per-epoch "
                        "narrowband filtering inside HyPyP (for comparison).")
    p.add_argument("--circ-corr-method", choices=["adjusted", "classic"], default="adjusted",
                   help="(continuous/default path only) which circular "
                        "correlation formula to use. 'adjusted' (default) "
                        "is the bias-adjusted Jammalamadaka & Sengupta (2001) "
                        "formula for arbitrary/not-well-defined circular "
                        "means, matching Zimmermann et al. (2024)'s "
                        "recommendation for continuous EEG and consistent "
                        "with HyPyP's 'accorr' mode used in --legacy-epochs. "
                        "'classic' restores the plain Fisher & Lee (1983) "
                        "formula (circ_corr_masked), which can be biased/"
                        "unstable for continuous EEG whose per-window "
                        "circular mean is not well defined -- kept for "
                        "comparison only. The --legacy-epochs path is "
                        "unaffected by this flag; it always uses HyPyP's "
                        "'accorr' (equivalent to 'adjusted').")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    circ_corr_fn_continuous = (circ_corr_adjusted_masked if args.circ_corr_method == "adjusted"
                                else circ_corr_masked)

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

    tag_band_names = []
    half = args.stim_bandwidth / 2
    if args.tag_hz_a is not None:
        tag_a_name = f"tagA_{args.tag_hz_a:g}hz"
        freq_bands[tag_a_name] = (args.tag_hz_a - half, args.tag_hz_a + half)
        if tag_a_name not in args.bands:
            args.bands = list(args.bands) + [tag_a_name]
        tag_band_names.append(tag_a_name)
        print(f"  frequency-tagging band added: {tag_a_name} "
              f"({freq_bands[tag_a_name][0]:.2f}-{freq_bands[tag_a_name][1]:.2f} Hz, "
              f"subject A's own reversal rate)")
    if args.tag_hz_b is not None:
        tag_b_name = f"tagB_{args.tag_hz_b:g}hz"
        freq_bands[tag_b_name] = (args.tag_hz_b - half, args.tag_hz_b + half)
        if tag_b_name not in args.bands:
            args.bands = list(args.bands) + [tag_b_name]
        tag_band_names.append(tag_b_name)
        print(f"  frequency-tagging band added: {tag_b_name} "
              f"({freq_bands[tag_b_name][0]:.2f}-{freq_bands[tag_b_name][1]:.2f} Hz, "
              f"subject B's own reversal rate)")
    if tag_band_names:
        print("  NOTE: only ONE subject was actually driven at each tag "
              "frequency (the other's activity there is incidental), so "
              "elevated inter-brain coupling in a tag_* band -- if it also "
              "clears --surrogate/--pool-dir -- is not explainable by both "
              "brains sharing one external clock, unlike --stim-hz.")
    if args.tag_hz_a is not None and args.tag_hz_b is not None and \
       abs(args.tag_hz_a - args.tag_hz_b) < args.stim_bandwidth:
        print(f"  WARNING: --tag-hz-a ({args.tag_hz_a}) and --tag-hz-b "
              f"({args.tag_hz_b}) are closer than --stim-bandwidth "
              f"({args.stim_bandwidth}) -- their bands overlap, defeating "
              "the point of tagging each subject at a different rate.")

    print("="*60)
    print("LOADING")
    print("="*60)
    onset_a = load_stimulus_onset(args.csv_a)
    onset_b = load_stimulus_onset(args.csv_b)
    if onset_a is not None and onset_b is not None:
        print(f"  IBS mode: stimulus markers found "
              f"(A={onset_a:.3f}s, B={onset_b:.3f}s)")
    elif onset_a is not None or onset_b is not None:
        print("  WARNING: only one recording has a stimulus marker -- "
              "alignment skipped")
        onset_a = onset_b = None
    else:
        print("  No stimulus markers found -- using full recordings")
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

    if args.sync_signal_hz is not None and not args.legacy_epochs:
        offset_s, drift_rate = estimate_sync_offset_drift(
            raw_a_pp, raw_b_pp, args.sync_signal_hz,
            bandwidth=args.sync_bandwidth, drift_only=args.sync_drift_only)
        dur = raw_b_pp.n_times / fs_a
        print(f"  cross-device sync ({args.sync_signal_hz:g} Hz driver): "
              f"offset={offset_s * 1000:.1f}ms  drift={drift_rate * 1e6:.1f}ppm "
              f"(~{drift_rate * dur * 1000:.1f}ms over the recording)"
              f"{'  [drift only]' if args.sync_drift_only else ''}")
        raw_b_pp = apply_sync_correction(raw_b_pp, offset_s, drift_rate)

    plot_raw_with_gaps(raw_a_pp, raw_b_pp, os.path.join(out_dir, "raw_with_gaps.png"))
    plot_psd(raw_a_pp, raw_b_pp, os.path.join(out_dir, "psd.png"))

    epochs_a = epochs_b = None
    n_ep = None
    bad_a = bad_b = good_mask = None
    n_common = None

    if args.legacy_epochs:
        print()
        print("="*60)
        print("EPOCHING (legacy fixed-length epochs, --legacy-epochs)")
        print("="*60)
        amp_thresh = args.amplitude_threshold if args.amplitude_threshold > 0 else None
        thresh_str = f"{amp_thresh} uV" if amp_thresh else "disabled"
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
    else:
        print()
        print("="*60)
        print("CONTINUOUS ARTIFACT DETECTION (default; see --legacy-epochs "
              "for the old fixed-epoch path)")
        print("="*60)
        print(f"  sliding window={args.artifact_window}s  step={args.artifact_step}s  "
              f"threshold={args.artifact_threshold} uV  pad={args.artifact_pad}s")
        print(f"  circular correlation method: {args.circ_corr_method}")
        bad_a = continuous_bad_mask(raw_a_pp, window_s=args.artifact_window,
                                     step_s=args.artifact_step,
                                     threshold_uv=args.artifact_threshold,
                                     pad_s=args.artifact_pad)
        bad_b = continuous_bad_mask(raw_b_pp, window_s=args.artifact_window,
                                     step_s=args.artifact_step,
                                     threshold_uv=args.artifact_threshold,
                                     pad_s=args.artifact_pad)
        n_common = min(len(bad_a), len(bad_b))
        bad_a = bad_a[:n_common]
        bad_b = bad_b[:n_common]
        good_mask = ~(bad_a | bad_b)
        total_s = n_common / fs_a
        good_s = good_mask.sum() / fs_a
        print(f"  Subject A: {100 * (~bad_a).mean():.1f}% clean")
        print(f"  Subject B: {100 * (~bad_b).mean():.1f}% clean")
        print(f"  Jointly clean (usable for connectivity): {good_s:.1f}s / "
              f"{total_s:.1f}s ({100 * good_mask.mean():.1f}%)")

        if good_mask.sum() == 0:
            print()
            print("  No jointly-clean samples survive. Try loosening "
                  "--artifact-threshold, or check hardware/fit.")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Optionally load a pool of OTHER subjects' recordings for pseudo-pair
    # (cross-dyad) validation. Each pool file goes through the SAME load ->
    # preprocess chain as the two main subjects, so the comparison is fair.
    # ------------------------------------------------------------------
    pool_epochs = []
    pool_data = []  # continuous mode: list of (raw_pp, bad_mask)
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
                  "excluded as the real subjects) -- pseudo-pair validation "
                  "will be skipped.")
        pool_amp_thresh_arg = (args.pool_amplitude_threshold
                               if args.pool_amplitude_threshold is not None
                               else args.amplitude_threshold)
        pool_amp_thresh = pool_amp_thresh_arg if pool_amp_thresh_arg > 0 else None

        if args.legacy_epochs:
            print(f"  pool amplitude_threshold={pool_amp_thresh or 'disabled'} uV "
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
        else:
            for f in pool_files:
                raw_pool, bad_pool, fs_pool = load_and_preprocess_continuous(
                    f, subject_label=f"POOL_{os.path.basename(f)}",
                    h_freq=h_freq, use_ica=args.ica, align_onset=True, quiet=False,
                    artifact_window=args.artifact_window,
                    artifact_step=args.artifact_step,
                    artifact_threshold=args.artifact_threshold,
                    artifact_pad=args.artifact_pad,
                )
                if raw_pool is not None and fs_pool == fs_a:
                    pool_data.append((raw_pool, bad_pool))
                elif raw_pool is not None:
                    print(f"  WARNING: {f} has fs={fs_pool} != {fs_a}, skipping "
                          "(sampling rate must match for pooling)")
            print(f"  Pool ready: {len(pool_data)} usable recordings "
                  f"(from {len(pool_files)} files found)")

    print()
    print("="*60)
    print("PLV + CIRCULAR CORRELATION PER BAND")
    print("="*60)
    if args.legacy_epochs:
        if args.prefilter:
            print("  --prefilter is ON: continuous raw is band-passed BEFORE "
                  "epoching for each band (reduces per-epoch filter edge bias).")
        else:
            print("  --prefilter is OFF: each short epoch is narrowband-filtered "
                  "independently inside HyPyP (legacy behaviour).")
    else:
        print("  continuous path: band-passing + Hilbert transform run on the "
              "full recording per band, then only jointly-clean samples "
              "(good_mask) are used for the PLV/circ-corr sum "
              f"(circ-corr method: {args.circ_corr_method}).")

    plvs = {}
    ccs = {}
    p_values = {}
    cc_p_values = {}
    sig_masks = {}
    cc_sig_masks = {}
    summary_lines = [f"Hyperscanning summary  {datetime.now().isoformat()}"]
    summary_lines.append(f"  A: {args.csv_a}")
    summary_lines.append(f"  B: {args.csv_b}")
    if args.legacy_epochs:
        summary_lines.append(f"  fs={fs_a:.0f} Hz, epochs={n_ep}, "
                             f"epoch_len={args.epoch_len}s")
    else:
        summary_lines.append(
            f"  fs={fs_a:.0f} Hz, continuous artifact rejection "
            f"(window={args.artifact_window}s step={args.artifact_step}s "
            f"threshold={args.artifact_threshold}uV), "
            f"{good_mask.sum() / fs_a:.1f}s/{n_common / fs_a:.1f}s usable "
            f"({100 * good_mask.mean():.1f}%), circ-corr method={args.circ_corr_method}"
        )
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

        if args.legacy_epochs:
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
                      "filtering/rejection -- skipping")
                continue

            plv = plv_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                             already_filtered=already_filtered)
            cc = circular_corr_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                                      already_filtered=already_filtered)

            def compute_within_null(n_surr, _epochs_a_band=epochs_a_band,
                                     _epochs_b_band=epochs_b_band, _band=band,
                                     _already_filtered=already_filtered):
                null_plv = surrogate_distribution(
                    _epochs_a_band, _epochs_b_band, _band, n_surr,
                    metric_fn=plv_hypyp, sfreq=fs_a, already_filtered=_already_filtered)
                null_cc = surrogate_distribution(
                    _epochs_a_band, _epochs_b_band, _band, n_surr,
                    metric_fn=circular_corr_hypyp, sfreq=fs_a, already_filtered=_already_filtered)
                return null_plv, null_cc

            def compute_pool_null(_epochs_a_band=epochs_a_band, _epochs_b_band=epochs_b_band,
                                   _band=band, _already_filtered=already_filtered):
                null_plv_pool = []
                for target in (_epochs_a_band, _epochs_b_band):
                    res = pseudo_pair_distribution(
                        target, pool_epochs, _band, plv_hypyp, sfreq=fs_a,
                        already_filtered=_already_filtered,
                        shuffles_per_pool_member=args.pool_shuffles)
                    if res is not None:
                        null_plv_pool.append(res)
                return np.concatenate(null_plv_pool, axis=0) if null_plv_pool else None

            have_pool = bool(pool_epochs)
        else:
            raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
            raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
            analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
            analytic_b = analytic_signal(raw_b_band, n_samples=n_common)

            plv = plv_masked(analytic_a, analytic_b, good_mask)
            cc = circ_corr_fn_continuous(analytic_a, analytic_b, good_mask)

            def compute_within_null(n_surr, target_n=None, seed=0,
                                     _analytic_a=analytic_a, _analytic_b=analytic_b,
                                     _bad_a=bad_a, _bad_b=bad_b):
                null_plv, ns_plv = circular_shift_surrogates_continuous(
                    _analytic_a, _bad_a, _analytic_b, _bad_b, n_surr, plv_masked,
                    seed=seed, target_n=target_n)
                null_cc, ns_cc = circular_shift_surrogates_continuous(
                    _analytic_a, _bad_a, _analytic_b, _bad_b, n_surr, circ_corr_fn_continuous,
                    seed=seed, target_n=target_n)
                return null_plv, null_cc, ns_plv, ns_cc

            def compute_pool_null(target_n=None, seed=0,
                                   _analytic_a=analytic_a, _analytic_b=analytic_b,
                                   _bad_a=bad_a, _bad_b=bad_b, _band=band):
                if not pool_data:
                    return None, np.array([])
                pool_analytic_list = []
                pool_bad_list = []
                for raw_pool, bad_pool in pool_data:
                    raw_pool_band = prefilter_raw_for_band(raw_pool, _band)
                    pool_analytic_list.append(analytic_signal(raw_pool_band))
                    pool_bad_list.append(bad_pool)
                null_plv_pool = []
                ns_pool_all = []
                for target_analytic, target_bad in ((_analytic_a, _bad_a), (_analytic_b, _bad_b)):
                    res, ns = pseudo_pair_continuous(
                        target_analytic, target_bad, pool_analytic_list, pool_bad_list,
                        plv_masked, shuffles_per_pool_member=args.pool_shuffles,
                        seed=seed, target_n=target_n)
                    ns_pool_all.append(ns)
                    if res is not None:
                        null_plv_pool.append(res)
                ns_pool_all = np.concatenate(ns_pool_all) if ns_pool_all else np.array([])
                pooled = np.concatenate(null_plv_pool, axis=0) if null_plv_pool else None
                return pooled, ns_pool_all

            have_pool = bool(pool_data)

        plvs[band_name] = plv
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean PLV = {plv.mean():.3f}  max = {plv.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"plv_{band_name}.npy"), plv)

        ccs[band_name] = cc
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean circ-r = {cc.mean():.3f}  min = {cc.min():.3f}  max = {cc.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"circ_corr_{band_name}.npy"), cc)

        # ---------------- WITHIN-DYAD surrogate (optional) -------------
        if args.surrogate > 0:
            print(f"     running {args.surrogate} within-dyad surrogates "
                  f"(PLV + circ-corr)...")
            if args.legacy_epochs:
                null_plv, null_cc = compute_within_null(args.surrogate)
            else:
                null_plv, null_cc, ns_within_plv, ns_within_cc = compute_within_null(args.surrogate)
                real_n = int(good_mask.sum())
                line = (f"     within-dyad null sample sizes: "
                        f"min={ns_within_plv.min()/fs_a:.1f}s  "
                        f"median={np.median(ns_within_plv)/fs_a:.1f}s  "
                        f"max={ns_within_plv.max()/fs_a:.1f}s  "
                        f"(real dyad N={real_n/fs_a:.1f}s)")
                print(line)
                summary_lines.append(line)
            p_val = (null_plv >= plv[None, :, :]).mean(axis=0)
            p_values[band_name] = p_val

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
        if have_pool:
            n_pool = len(pool_epochs) if args.legacy_epochs else len(pool_data)
            print(f"     running pseudo-pair null against {n_pool} "
                  f"pool recordings x{args.pool_shuffles} draws (PLV)...")

            plv_for_pval = plv  # may be replaced by a length-matched value below

            if args.legacy_epochs:
                null_plv_pool = compute_pool_null()
            else:
                null_plv_pool, ns_pool = compute_pool_null()
                real_n = int(good_mask.sum())
                if ns_pool.size > 0:
                    line = (f"     pool null sample sizes: "
                            f"min={ns_pool.min()/fs_a:.1f}s  "
                            f"median={np.median(ns_pool)/fs_a:.1f}s  "
                            f"max={ns_pool.max()/fs_a:.1f}s  "
                            f"(real dyad N={real_n/fs_a:.1f}s)")
                    print(line)
                    summary_lines.append(line)

                    if args.match_null_length:
                        nonzero = ns_pool[ns_pool > 0]
                        floor_n = int(args.min_null_seconds * fs_a)
                        robust_pool_n = int(np.percentile(nonzero, 10)) if nonzero.size else 0
                        target_n = max(floor_n, robust_pool_n)
                        target_n = min(target_n, real_n) if real_n > 0 else target_n
                        target_n = max(target_n, 1)

                        if 0 < target_n < real_n:
                            line = (f"     length-matching pool-null comparison to "
                                    f"N={target_n/fs_a:.1f}s (10th pct of pool draw "
                                    f"sizes, floored at {args.min_null_seconds:.0f}s)")
                            print(line)
                            summary_lines.append(line)

                            null_plv_pool_matched, ns_pool_matched = compute_pool_null(
                                target_n=target_n, seed=1)
                            plv_matched_obs = matched_observed_value(
                                analytic_a, analytic_b, good_mask, target_n,
                                plv_masked, n_draws=5, seed=2)

                            if null_plv_pool_matched is not None and plv_matched_obs is not None:
                                null_plv_pool = null_plv_pool_matched
                                plv_for_pval = plv_matched_obs
                                line = (f"     length-matched observed PLV "
                                        f"(N={target_n/fs_a:.1f}s, avg of 5 subsamples) "
                                        f"= {plv_matched_obs.mean():.3f}  "
                                        f"[full-length headline PLV = {plv.mean():.3f}]")
                                print(line)
                                summary_lines.append(line)
                            else:
                                line = ("     length-matched recompute failed "
                                        "(insufficient samples) -- falling back to "
                                        "unmatched null comparison")
                                print(line)
                                summary_lines.append(line)

            if null_plv_pool is not None:
                p_val_pool = (null_plv_pool >= plv_for_pval[None, :, :]).mean(axis=0)
                matched_tag = " (length-matched)" if plv_for_pval is not plv else ""
                if args.correction == "fdr":
                    sig_mask_pool, _ = fdr_bh(p_val_pool)
                    n_sig_pool = int(sig_mask_pool.sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair{matched_tag}, "
                            f"FDR-corrected): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                else:
                    n_sig_pool = int((p_val_pool < 0.05).sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair{matched_tag}, "
                            f"UNCORRECTED): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                print(line)
                summary_lines.append(line)
                np.save(os.path.join(out_dir, f"plv_p_pool_{band_name}.npy"), p_val_pool)
                line = (f"     Interpretation: real PLV={plv_for_pval.mean():.3f}{matched_tag} vs "
                        f"pool (independent, same-stimulus) PLV={null_plv_pool.mean():.3f} "
                        f"-> {'ABOVE pool baseline' if plv_for_pval.mean() > null_plv_pool.mean() else 'NOT above pool baseline'}")
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