"""
test_synthetic_connectivity.py — ground-truth validation for the pipeline's
PLV / circular-correlation estimate on the default continuous path
(preprocess -> continuous_bad_mask -> prefilter_raw_for_band ->
analytic_signal -> plv_masked / circ_corr_masked).

Real recordings have no known ground truth for how much genuine inter-brain
coupling is present, so a low PLV on real data is ambiguous: it could mean
"no real coupling" or "the pipeline is failing to detect coupling that's
there". This script builds pairs of synthetic 4-channel "subjects" with
KNOWN coupling built in, runs them through the actual production functions
(not a reimplementation), and checks the reported PLV/circ-corr against
what's expected:

  1. Perfect coupling (both subjects share the exact same alpha-band
     waveform, zero lag) -> PLV should be high.
  2. Zero coupling (independent alpha-band waveforms) -> PLV should sit
     near the finite-sample chance level, not near 1.
  3. Dose-response: mixing a shared component into both subjects' alpha
     band at increasing strength should increase PLV monotonically.
  4. Artifact-contamination robustness: a large synchronized artifact
     burst injected into BOTH subjects at the same timepoints (e.g. a
     shared movement/lighting event -- a classic hyperscanning confound)
     inflates a naive whole-recording PLV estimate, but the pipeline's
     continuous_bad_mask()-derived good_mask should mask it back down
     toward the true (zero-coupling) baseline from case 2.

Usage:
  python test_synthetic_connectivity.py
"""
import sys

import numpy as np
from scipy.signal import butter, filtfilt

from pipeline import (
    FREQ_BANDS,
    preprocess,
    continuous_bad_mask,
    prefilter_raw_for_band,
    analytic_signal,
    plv_masked,
    circ_corr_masked,
    matched_observed_value,
    pseudo_pair_continuous,
    summarize_positive_control,
)
import mne

SFREQ = 256.0
DURATION_S = 180.0
N_SAMPLES = int(round(DURATION_S * SFREQ))
CH_NAMES_SYN = ["TP9", "AF7", "AF8", "TP10"]
ALPHA_BAND = FREQ_BANDS["alpha"]

ALPHA_AMP_UV = 40.0
NOISE_STD_UV = 10.0

# Per-channel topography weights (mean=1.0 each) for broadcasting a shared
# source into a subject's 4 channels. This MUST vary across channels: an
# identical broadcast is exactly what average referencing (see
# preprocess()) removes completely -- it's common-mode by construction, so
# it gets zeroed out before any connectivity code even runs. Real scalp
# topographies are never perfectly flat either, so this is also more
# realistic than a uniform broadcast.
TOPO_A = np.array([1.6, 0.4, 1.2, 0.8])
TOPO_B = np.array([0.6, 1.4, 0.8, 1.2])
ARTIFACT_TOPO = np.array([1.5, 0.6, 1.3, 0.9])


def band_limited_noise(n_samples, sfreq, band, rng, amp_uv):
    """Gaussian white noise band-passed into `band`, scaled to amp_uv std."""
    white = rng.standard_normal(n_samples)
    nyq = sfreq / 2
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
    filtered = filtfilt(b, a, white)
    return filtered / filtered.std() * amp_uv


def make_subject_raw(data_uv, subject_label):
    ch_names = [f"{subject_label}_{c}" for c in CH_NAMES_SYN]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def run_pipeline_plv(data_a_uv, data_b_uv, good_mask_override=None):
    """
    Run the actual production continuous-path functions end to end:
    preprocess -> continuous_bad_mask -> prefilter_raw_for_band ->
    analytic_signal -> plv_masked / circ_corr_masked.

    If good_mask_override is given, it's used in place of the pipeline's
    own artifact mask (case 4 uses this to show what a naive, unmasked
    estimate looks like).
    """
    raw_a = preprocess(make_subject_raw(data_a_uv, "A"), subject_label="A")
    raw_b = preprocess(make_subject_raw(data_b_uv, "B"), subject_label="B")

    if good_mask_override is None:
        bad_a = continuous_bad_mask(raw_a)
        bad_b = continuous_bad_mask(raw_b)
        good_mask = ~(bad_a | bad_b)
    else:
        good_mask = good_mask_override

    raw_a_band = prefilter_raw_for_band(raw_a, ALPHA_BAND)
    raw_b_band = prefilter_raw_for_band(raw_b, ALPHA_BAND)
    n_common = min(raw_a_band.n_times, raw_b_band.n_times, len(good_mask))
    analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
    analytic_b = analytic_signal(raw_b_band, n_samples=n_common)
    good_mask = good_mask[:n_common]

    plv = plv_masked(analytic_a, analytic_b, good_mask)
    cc = circ_corr_masked(analytic_a, analytic_b, good_mask)
    return plv, cc


def run_pipeline_plv_length_matched(data_a_uv, data_b_uv, target_n, n_draws=5):
    """
    Run the production continuous-path pipeline and recompute the observed
    PLV on random subsamples matched to target_n.

    This exercises the same length-matching helper used by the pseudo-pair
    null comparison in pipeline.py.
    """
    raw_a = preprocess(make_subject_raw(data_a_uv, "A"), subject_label="A")
    raw_b = preprocess(make_subject_raw(data_b_uv, "B"), subject_label="B")
    bad_a = continuous_bad_mask(raw_a)
    bad_b = continuous_bad_mask(raw_b)
    good_mask = ~(bad_a | bad_b)

    raw_a_band = prefilter_raw_for_band(raw_a, ALPHA_BAND)
    raw_b_band = prefilter_raw_for_band(raw_b, ALPHA_BAND)
    n_common = min(raw_a_band.n_times, raw_b_band.n_times, len(good_mask))
    analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
    analytic_b = analytic_signal(raw_b_band, n_samples=n_common)
    good_mask = good_mask[:n_common]

    plv_full = plv_masked(analytic_a, analytic_b, good_mask)
    plv_matched = matched_observed_value(
        analytic_a,
        analytic_b,
        good_mask,
        target_n=target_n,
        metric_fn=plv_masked,
        n_draws=n_draws,
        seed=11,
    )
    return plv_full, plv_matched


def run_short_pool_length_match_check(target_n):
    """Return the null output for a deliberately too-short pool draw.

    With exact length matching enabled in pipeline.py, a pool draw that does
    not have at least target_n jointly-clean samples should be skipped rather
    than treated as matched at a shorter N.
    """
    target_analytic = np.exp(1j * np.zeros((1, 100)))
    target_bad = np.zeros(100, dtype=bool)
    pool_analytic = np.exp(1j * np.zeros((1, 40)))
    pool_bad = np.zeros(40, dtype=bool)
    nulls, ns = pseudo_pair_continuous(
        target_analytic,
        target_bad,
        [pool_analytic],
        [pool_bad],
        plv_masked,
        shuffles_per_pool_member=1,
        seed=0,
        target_n=target_n,
    )
    return nulls, ns


def make_broadband(rng, n_channels=4):
    return np.stack([rng.standard_normal(N_SAMPLES) * NOISE_STD_UV
                      for _ in range(n_channels)])


def run_case(name, check_fn):
    try:
        check_fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False


def test_positive_control_verdict():
    verdict = summarize_positive_control(
        np.array([0.15, 0.18]),
        np.array([0.05, 0.06]),
        np.array([0.01, 0.02]),
        sig_mask=np.array([True, False]),
    )
    assert verdict["status"] == "PASS"
    assert verdict["passed"] is True
    assert verdict["n_sig_pairs"] == 1

    weak_verdict = summarize_positive_control(
        np.array([0.15, 0.18]),
        np.array([0.05, 0.06]),
        np.array([0.2, 0.25]),
        sig_mask=np.array([False, False]),
    )
    assert weak_verdict["status"] == "WEAK"
    assert weak_verdict["passed"] is False


def main():
    results = []

    # ------------------------------------------------------------------
    # Case 1: perfect coupling -- both subjects' 4 channels all carry the
    # exact same alpha-band waveform (zero lag) plus independent noise.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    shared = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    data_a = TOPO_A[:, np.newaxis] * shared[np.newaxis, :] + make_broadband(rng)
    data_b = TOPO_B[:, np.newaxis] * shared[np.newaxis, :] + make_broadband(rng)
    plv_perfect, cc_perfect = run_pipeline_plv(data_a, data_b)
    print(f"  [case 1] perfect coupling: mean PLV={plv_perfect.mean():.3f}  "
          f"mean |circ-r|={np.abs(cc_perfect).mean():.3f}")

    def check_perfect_coupling_high_plv():
        assert plv_perfect.mean() > 0.7, \
            f"expected PLV > 0.7 for identical shared signal, got {plv_perfect.mean():.3f}"

    def check_perfect_coupling_high_circ_r():
        # Average referencing means each channel's residual coefficient
        # (topo weight - 1) sums to zero across the 4 channels, so some
        # pairs get a same-sign (same-direction) relationship and some
        # get flipped (opposite-direction, negative circ-r) even under
        # true zero-lag coupling. abs() checks for a strong DETERMINISTIC
        # phase relationship without assuming a particular sign pattern.
        assert np.abs(cc_perfect).mean() > 0.5, \
            f"expected |circ-r| > 0.5 for identical (zero-lag) shared signal, got {np.abs(cc_perfect).mean():.3f}"

    results.append(run_case("perfect coupling -> high PLV", check_perfect_coupling_high_plv))
    results.append(run_case("perfect coupling -> high positive circ-r", check_perfect_coupling_high_circ_r))

    # ------------------------------------------------------------------
    # Case 2: zero coupling -- independent alpha-band waveforms per subject.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1)
    indep_a = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    indep_b = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    data_a_indep = TOPO_A[:, np.newaxis] * indep_a[np.newaxis, :] + make_broadband(rng)
    data_b_indep = TOPO_B[:, np.newaxis] * indep_b[np.newaxis, :] + make_broadband(rng)
    plv_indep, cc_indep = run_pipeline_plv(data_a_indep, data_b_indep)
    print(f"  [case 2] zero coupling: mean PLV={plv_indep.mean():.3f}  "
          f"mean circ-r={cc_indep.mean():.3f}")

    def check_independent_low_plv():
        assert plv_indep.mean() < 0.25, \
            f"expected PLV < 0.25 for independent signals, got {plv_indep.mean():.3f}"

    def check_independent_low_circ_r():
        assert abs(cc_indep.mean()) < 0.15, \
            f"expected |circ-r| < 0.15 for independent signals, got {cc_indep.mean():.3f}"

    def check_perfect_beats_independent():
        assert plv_perfect.mean() > plv_indep.mean() + 0.5, \
            ("expected a large PLV gap between perfectly-coupled and "
             f"independent signals, got {plv_perfect.mean():.3f} vs {plv_indep.mean():.3f}")

    results.append(run_case("independent signals -> low PLV", check_independent_low_plv))
    results.append(run_case("independent signals -> near-zero circ-r", check_independent_low_circ_r))
    results.append(run_case("perfect coupling clearly exceeds independent baseline",
                             check_perfect_beats_independent))

    # ------------------------------------------------------------------
    # Case 3: dose-response -- mix the same shared/independent components
    # at increasing coupling strength and check PLV rises monotonically.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(2)
    shared_dr = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    indep_a_dr = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    indep_b_dr = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    noise_a_dr = make_broadband(rng)
    noise_b_dr = make_broadband(rng)

    coupling_levels = [0.0, 0.3, 0.6, 1.0]
    dose_response_plv = []
    for c in coupling_levels:
        mix_a = np.sqrt(c) * shared_dr + np.sqrt(1 - c) * indep_a_dr
        mix_b = np.sqrt(c) * shared_dr + np.sqrt(1 - c) * indep_b_dr
        data_a_c = TOPO_A[:, np.newaxis] * mix_a[np.newaxis, :] + noise_a_dr
        data_b_c = TOPO_B[:, np.newaxis] * mix_b[np.newaxis, :] + noise_b_dr
        plv_c, _ = run_pipeline_plv(data_a_c, data_b_c)
        dose_response_plv.append(plv_c.mean())
    print(f"  [case 3] dose-response PLV at c={coupling_levels}: "
          f"{[round(v, 3) for v in dose_response_plv]}")

    def check_dose_response_monotonic_trend():
        corr = np.corrcoef(coupling_levels, dose_response_plv)[0, 1]
        assert corr > 0.9, \
            f"expected strong positive correlation between coupling strength and PLV, got r={corr:.3f}"

    def check_dose_response_endpoints():
        assert dose_response_plv[-1] > dose_response_plv[0] + 0.3, \
            (f"expected a clear PLV increase from c=0 to c=1, got "
             f"{dose_response_plv[0]:.3f} -> {dose_response_plv[-1]:.3f}")

    results.append(run_case("dose-response: PLV correlates with coupling strength",
                             check_dose_response_monotonic_trend))
    results.append(run_case("dose-response: PLV clearly higher at c=1 than c=0",
                             check_dose_response_endpoints))

    # ------------------------------------------------------------------
    # Case 4: artifact-contamination robustness. Start from the zero-
    # coupling (case 2) signals, inject a large SIMULTANEOUS artifact
    # burst into both subjects at the same timepoints (simulating a
    # shared external confound, e.g. both subjects flinching at the same
    # on-screen event), and check that the pipeline's own artifact mask
    # protects the PLV estimate from being inflated by it.
    # ------------------------------------------------------------------
    rng_art = np.random.default_rng(3)
    data_a_art = data_a_indep.copy()
    data_b_art = data_b_indep.copy()
    burst_starts_s = np.arange(10, DURATION_S - 10, 10.0)  # every 10s
    for start_s in burst_starts_s:
        start = int(round(start_s * SFREQ))
        end = int(round((start_s + 1.0) * SFREQ))
        n = end - start
        # broadband noise, not an alternating square wave: a sample-rate
        # square wave sits almost entirely outside the alpha band and gets
        # filtered away by prefilter_raw_for_band before it can contaminate
        # anything -- real movement/EMG artifacts have broadband spectral
        # content that leaks into every band, including alpha.
        burst = rng_art.standard_normal(n) * 500.0
        # per-channel amplitude variation, same reason as TOPO_A/TOPO_B: an
        # identical broadcast across a subject's 4 channels is common-mode
        # and gets removed entirely by average referencing before it ever
        # reaches continuous_bad_mask.
        data_a_art[:, start:end] = ARTIFACT_TOPO[:, np.newaxis] * burst[np.newaxis, :]
        data_b_art[:, start:end] = ARTIFACT_TOPO[:, np.newaxis] * burst[np.newaxis, :]  # same artifact, same instant, both subjects

    plv_naive, _ = run_pipeline_plv(
        data_a_art, data_b_art,
        good_mask_override=np.ones(N_SAMPLES, dtype=bool),  # no masking at all
    )
    plv_protected, _ = run_pipeline_plv(data_a_art, data_b_art)  # pipeline's real masking
    print(f"  [case 4] shared-artifact PLV: naive(unmasked)={plv_naive.mean():.3f}  "
          f"protected(masked)={plv_protected.mean():.3f}  "
          f"zero-coupling baseline (case 2)={plv_indep.mean():.3f}")

    def check_naive_estimate_inflated():
        assert plv_naive.mean() > plv_indep.mean() + 0.05, \
            ("expected the unmasked estimate to be visibly inflated by the "
             f"shared artifact, got naive={plv_naive.mean():.3f} vs "
             f"true baseline={plv_indep.mean():.3f}")

    def check_masking_recovers_baseline():
        assert abs(plv_protected.mean() - plv_indep.mean()) < 0.08, \
            (f"expected the pipeline's own artifact mask to bring PLV back "
             f"near the true zero-coupling baseline ({plv_indep.mean():.3f}), "
             f"got {plv_protected.mean():.3f}")

    def check_masking_beats_naive():
        assert plv_protected.mean() < plv_naive.mean() - 0.05, \
            ("expected masking to measurably reduce the artifact-inflated "
             f"estimate, got protected={plv_protected.mean():.3f} vs "
             f"naive={plv_naive.mean():.3f}")

    results.append(run_case("unmasked estimate is inflated by a shared artifact",
                             check_naive_estimate_inflated))
    results.append(run_case("pipeline's artifact mask recovers the true baseline",
                             check_masking_recovers_baseline))
    results.append(run_case("masked estimate is clearly lower than the naive one",
                             check_masking_beats_naive))

    # ------------------------------------------------------------------
    # Case 5: length-matched observed PLV -- subsample a long, strongly
    # coupled recording down to a much shorter target_n and confirm the
    # estimate stays high, remains clearly above the independent baseline,
    # and stays close to the full-length point estimate.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(4)
    shared_lm = band_limited_noise(N_SAMPLES, SFREQ, ALPHA_BAND, rng, ALPHA_AMP_UV)
    data_a_lm = TOPO_A[:, np.newaxis] * shared_lm[np.newaxis, :] + make_broadband(rng)
    data_b_lm = TOPO_B[:, np.newaxis] * shared_lm[np.newaxis, :] + make_broadband(rng)
    target_n = int(round(20.0 * SFREQ))
    plv_full_lm, plv_matched_lm = run_pipeline_plv_length_matched(
        data_a_lm, data_b_lm, target_n=target_n, n_draws=8
    )
    print(f"  [case 5] length-matched strong coupling: full={plv_full_lm.mean():.3f}  "
          f"matched({target_n / SFREQ:.1f}s)={plv_matched_lm.mean():.3f}  "
          f"independent baseline(case 2)={plv_indep.mean():.3f}")

    def check_length_matched_remains_high():
        assert plv_matched_lm.mean() > 0.65, \
            (f"expected a strong coupled signal to stay clearly above baseline "
             f"after length matching, got {plv_matched_lm.mean():.3f}")

    def check_length_matched_stays_close_to_full():
        assert abs(plv_matched_lm.mean() - plv_full_lm.mean()) < 0.15, \
            (f"expected the length-matched observed PLV to stay near the full-"
             f"length estimate, got {plv_full_lm.mean():.3f} vs "
             f"{plv_matched_lm.mean():.3f}")

    def check_length_matched_still_beats_independent():
        assert plv_matched_lm.mean() > plv_indep.mean() + 0.45, \
            (f"expected the matched coupled signal to remain clearly above the "
             f"independent baseline, got {plv_matched_lm.mean():.3f} vs "
             f"{plv_indep.mean():.3f}")

    results.append(run_case("length-matched observed PLV stays high",
                             check_length_matched_remains_high))
    results.append(run_case("length-matched observed PLV stays close to full-length",
                             check_length_matched_stays_close_to_full))
    results.append(run_case("length-matched observed PLV still exceeds the independent baseline",
                             check_length_matched_still_beats_independent))

    # ------------------------------------------------------------------
    # Case 6: exact length-matching -- if a pool draw is shorter than the
    # requested target_n, it should be skipped rather than reused at a
    # shorter, mismatched sample size.
    # ------------------------------------------------------------------
    short_nulls, short_ns = run_short_pool_length_match_check(target_n=50)

    def check_short_pool_draw_is_skipped():
        assert short_nulls is None, \
            "expected a too-short pool draw to be skipped under exact length matching"
        assert short_ns.size == 1 and short_ns[0] == 40, \
            f"expected the diagnostic N log to record the short draw size, got {short_ns}"

    results.append(run_case("short pool draws are skipped under exact length matching",
                             check_short_pool_draw_is_skipped))

    n_pass = sum(results)
    n_total = len(results)
    print()
    print(f"{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
