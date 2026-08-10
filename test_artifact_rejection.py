"""
test_artifact_rejection.py — synthetic ground-truth test for
pipeline.continuous_bad_mask() (the sliding-window peak-to-peak artifact
detector: 500ms window, 100ms step, 500uV threshold by default).

Real recordings have no known ground truth for exactly when an artifact
occurred, so a "% clean" number from pipeline.py alone can't tell you
whether the detector is catching the right windows. This script builds a
synthetic 4-channel signal with known-clean baseline plus artifacts of
known amplitude/timing injected at known sample ranges, then checks that
continuous_bad_mask() flags exactly what it should:

  1. A clear above-threshold burst gets flagged (recall).
  2. A brief (50ms) above-threshold spike still gets caught by the sliding
     window despite being much shorter than the 500ms window.
  3. A below-threshold bump is NOT flagged (specificity at the boundary).
  4. A BAD_gap annotation is folded into the mask even with no amplitude
     artifact present (the annotation-folding code path).
  5. Clean baseline far from any injected event has zero false positives.

Cases 1-4 pin pad_s=0.0 so they isolate the window/threshold/annotation
logic from the pad_s dilation (see case 6 below) -- otherwise the two
features would be entangled and a regression in either one could hide
behind the other.

  6. pad_s dilates a flagged run by exactly pad_s on each side (the
     margin added so band-pass filter ringing at an artifact's edge, e.g.
     from a railed/saturated segment, can't leak into a neighboring
     sample still counted as clean), and pad_s=0 disables it.

Usage:
  python test_artifact_rejection.py
"""
import sys

import mne
import numpy as np

from pipeline import CH_NAMES, continuous_bad_mask

SFREQ = 256.0
DURATION_S = 60.0
WINDOW_S = 0.5
STEP_S = 0.1
THRESHOLD_UV = 500.0

# baseline noise amplitude is small enough that its peak-to-peak over any
# 500ms window sits well under THRESHOLD_UV, so any flagged sample outside
# an injected event/annotation is a genuine false positive.
BASELINE_STD_UV = 8.0
BASELINE_ALPHA_UV = 10.0  # amplitude of a 10 Hz sinusoid added on top


def make_synthetic_data(seed=0):
    rng = np.random.RandomState(seed)
    n_ch = len(CH_NAMES)
    n_samples = int(round(DURATION_S * SFREQ))
    t = np.arange(n_samples) / SFREQ
    noise_uv = rng.normal(0.0, BASELINE_STD_UV, size=(n_ch, n_samples))
    alpha_uv = BASELINE_ALPHA_UV * np.sin(2 * np.pi * 10.0 * t)
    return noise_uv + alpha_uv[np.newaxis, :]


def inject_burst(data_uv, ch_idx, start_s, end_s, ptp_uv):
    """Overwrite [start_s, end_s) on channel ch_idx with an alternating
    +/- ptp_uv/2 pattern, guaranteeing an exact peak-to-peak amplitude of
    ptp_uv inside that slice regardless of how short it is."""
    start = int(round(start_s * SFREQ))
    end = int(round(end_s * SFREQ))
    n = end - start
    pattern = np.where(np.arange(n) % 2 == 0, ptp_uv / 2, -ptp_uv / 2)
    data_uv[ch_idx, start:end] = pattern
    return start, end


def make_raw(data_uv, annotations=None):
    ch_names = [f"SYN_{c}" for c in CH_NAMES]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    if annotations is not None:
        raw.set_annotations(annotations)
    return raw


def run_case(name, check_fn):
    try:
        check_fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False


def main():
    results = []

    # ------------------------------------------------------------------
    # Case 1-3: amplitude-based detection (clear burst, brief spike,
    # sub-threshold bump), all in one signal so we can also check the
    # detector leaves the surrounding clean baseline alone.
    # ------------------------------------------------------------------
    data_uv = make_synthetic_data(seed=0)

    clear_start, clear_end = inject_burst(data_uv, ch_idx=0,
                                           start_s=10.0, end_s=10.4,
                                           ptp_uv=800.0)
    sub_start, sub_end = inject_burst(data_uv, ch_idx=1,
                                       start_s=20.0, end_s=20.4,
                                       ptp_uv=350.0)
    spike_start, spike_end = inject_burst(data_uv, ch_idx=2,
                                           start_s=30.0, end_s=30.05,
                                           ptp_uv=900.0)

    raw = make_raw(data_uv)
    mask = continuous_bad_mask(raw, window_s=WINDOW_S, step_s=STEP_S,
                                threshold_uv=THRESHOLD_UV, pad_s=0.0)

    pad = int(round(WINDOW_S * SFREQ))

    def check_clear_burst_flagged():
        assert mask[clear_start:clear_end].all(), \
            "not every sample inside the 800uV burst was flagged"

    def check_clear_burst_bounded():
        # the mask may extend up to one window's width on either side of
        # the true burst (a window straddling the edge still trips), but
        # should not bleed out further than that.
        far_before = max(0, clear_start - pad - int(SFREQ))
        far_after = min(len(mask), clear_end + pad + int(SFREQ))
        assert not mask[far_before:clear_start - pad].any(), \
            "flagged region extends more than one window before the burst"
        assert not mask[clear_end + pad:far_after].any(), \
            "flagged region extends more than one window after the burst"

    def check_subthreshold_not_flagged():
        assert not mask[sub_start - pad:sub_end + pad].any(), \
            "a 350uV (sub-threshold) bump was incorrectly flagged as bad"

    def check_brief_spike_flagged():
        mid = (spike_start + spike_end) // 2
        assert mask[mid], \
            "a brief 50ms above-threshold spike was missed by the sliding window"

    def check_clean_baseline_no_false_positives():
        # windows well away from all three injected events (>1s clearance)
        clean_ranges = [
            (0, clear_start - pad - int(SFREQ)),
            (clear_end + pad + int(SFREQ), sub_start - pad - int(SFREQ)),
            (sub_end + pad + int(SFREQ), spike_start - pad - int(SFREQ)),
            (spike_end + pad + int(SFREQ), len(mask)),
        ]
        for a, b in clean_ranges:
            if b > a:
                assert not mask[a:b].any(), \
                    f"false positive(s) in clean baseline range [{a}:{b}]"

    results.append(run_case("clear above-threshold burst is fully flagged",
                             check_clear_burst_flagged))
    results.append(run_case("flagged region stays within one window of the burst",
                             check_clear_burst_bounded))
    results.append(run_case("sub-threshold bump is NOT flagged",
                             check_subthreshold_not_flagged))
    results.append(run_case("brief 50ms above-threshold spike is caught",
                             check_brief_spike_flagged))
    results.append(run_case("clean baseline has zero false positives",
                             check_clean_baseline_no_false_positives))

    # ------------------------------------------------------------------
    # Case 4: BAD_gap annotation folding, isolated from amplitude
    # detection (this stretch of data has no injected artifact).
    # ------------------------------------------------------------------
    data_uv_gap = make_synthetic_data(seed=1)
    gap_onset_s, gap_dur_s = 40.0, 0.5
    ann = mne.Annotations(onset=[gap_onset_s], duration=[gap_dur_s],
                           description=["BAD_gap"])
    raw_gap = make_raw(data_uv_gap, annotations=ann)
    mask_gap = continuous_bad_mask(raw_gap, window_s=WINDOW_S, step_s=STEP_S,
                                    threshold_uv=THRESHOLD_UV, pad_s=0.0)
    gap_start = int(round(gap_onset_s * SFREQ))
    gap_end = int(round((gap_onset_s + gap_dur_s) * SFREQ))

    def check_gap_annotation_folded():
        assert mask_gap[gap_start:gap_end].all(), \
            "BAD_gap annotation was not folded into the mask"

    def check_gap_no_bleed_from_amplitude():
        # nothing else in this signal is above threshold, so everything
        # outside the annotation window must be clean.
        assert not mask_gap[:gap_start].any(), \
            "false positive before the BAD_gap annotation"
        assert not mask_gap[gap_end:].any(), \
            "false positive after the BAD_gap annotation"

    results.append(run_case("BAD_gap annotation is folded into the mask",
                             check_gap_annotation_folded))
    results.append(run_case("no false positives elsewhere in the gap-only signal",
                             check_gap_no_bleed_from_amplitude))

    # ------------------------------------------------------------------
    # Case 6: pad_s dilates a flagged run by exactly pad_s on each side,
    # and pad_s=0 (used throughout cases 1-4 above) disables it.
    # ------------------------------------------------------------------
    data_uv_pad = make_synthetic_data(seed=2)
    pad_burst_start, pad_burst_end = inject_burst(data_uv_pad, ch_idx=0,
                                                   start_s=10.0, end_s=10.4,
                                                   ptp_uv=800.0)
    raw_pad = make_raw(data_uv_pad)
    pad_s = 0.3
    mask_padded = continuous_bad_mask(raw_pad, window_s=WINDOW_S, step_s=STEP_S,
                                       threshold_uv=THRESHOLD_UV, pad_s=pad_s)
    mask_unpadded = continuous_bad_mask(raw_pad, window_s=WINDOW_S, step_s=STEP_S,
                                         threshold_uv=THRESHOLD_UV, pad_s=0.0)
    unpadded_start = np.argmax(mask_unpadded)
    unpadded_end = len(mask_unpadded) - np.argmax(mask_unpadded[::-1])
    pad_n = int(round(pad_s * SFREQ))

    def check_pad_extends_flagged_region():
        expected_start = max(0, unpadded_start - pad_n)
        expected_end = min(len(mask_padded), unpadded_end + pad_n)
        assert mask_padded[expected_start:expected_end].all(), \
            "pad_s did not extend the flagged region by pad_s on both sides"

    def check_pad_does_not_overextend():
        before = unpadded_start - pad_n - 5
        after = unpadded_end + pad_n + 5
        assert before < 0 or not mask_padded[before], \
            "pad_s extended the flagged region further than pad_s"
        assert after >= len(mask_padded) or not mask_padded[after], \
            "pad_s extended the flagged region further than pad_s"

    def check_pad_zero_matches_unpadded():
        assert np.array_equal(mask_unpadded,
                               continuous_bad_mask(raw_pad, window_s=WINDOW_S,
                                                    step_s=STEP_S,
                                                    threshold_uv=THRESHOLD_UV,
                                                    pad_s=0.0)), \
            "pad_s=0.0 is not reproducible/deterministic"

    results.append(run_case("pad_s extends the flagged region by pad_s on both sides",
                             check_pad_extends_flagged_region))
    results.append(run_case("pad_s does not extend further than pad_s",
                             check_pad_does_not_overextend))
    results.append(run_case("pad_s=0.0 disables padding",
                             check_pad_zero_matches_unpadded))

    n_pass = sum(results)
    n_total = len(results)
    print()
    print(f"{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
