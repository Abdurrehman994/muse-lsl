"""
test_sync_correction.py -- ground-truth test for the cross-device timing
sync-correction (pipeline.estimate_sync_offset_drift / apply_sync_correction,
enabled on real runs via --sync-signal-hz).

Two independent Muses' clocks drift relative to each other; LSL's nominal-
rate dejittering hides it, and the drift erodes inter-brain phase coupling.
If both subjects saw the same periodic driver (a shared flicker/SSVEP), its
phase difference over time is a timing ruler. This test injects a KNOWN
offset + drift into subject B and checks that:

  1. estimate_sync_offset_drift() recovers the injected offset and drift.
  2. apply_sync_correction() restores inter-brain coupling that the injected
     misalignment had destroyed (PLV on a genuinely-coupled band recovers).
  3. --sync-drift-only recovers the drift while leaving the offset at 0.

The synthetic driver has identical phase across channels, so offset and
drift are both cleanly recoverable here -- on real single-frequency data the
constant offset is confounded with a neural phase lag (see the note in
pipeline.py section 2c); this test validates the correction MATH, which is
what the pipeline relies on once a clean shared driver is present.

Usage:
  python test_sync_correction.py
"""
import sys

import numpy as np
from scipy.signal import butter, filtfilt

from pipeline import (
    FREQ_BANDS, estimate_sync_offset_drift, apply_sync_correction,
    preprocess, prefilter_raw_for_band, analytic_signal, plv_masked,
)
import mne

SFREQ = 256.0
DURATION_S = 200.0
N = int(round(DURATION_S * SFREQ))
CH = ["TP9", "AF7", "AF8", "TP10"]
SYNC_HZ = 6.0            # shared flicker/driver frequency
ALPHA_BAND = FREQ_BANDS["alpha"]

# Per-channel topography so nothing is common-mode-removed by avg reference.
# Both are chosen so the strongest post-reference channel is ch0 with a
# POSITIVE driver residual for A and B alike -- otherwise average
# referencing would flip one subject's sync channel by pi, and that
# spatial phase flip would be absorbed into the offset estimate as a
# spurious ~half-period timing offset (the single-frequency offset/neural-
# phase confound documented in pipeline.py section 2c). Different weights
# on the other channels keep the two topographies distinct/realistic.
TOPO_A = np.array([1.7, 0.5, 1.1, 0.7])
TOPO_B = np.array([1.6, 0.6, 0.9, 0.9])


def band_noise(band, rng, amp):
    white = rng.standard_normal(N)
    b, a = butter(4, [band[0] / (SFREQ / 2), band[1] / (SFREQ / 2)], btype="band")
    f = filtfilt(b, a, white)
    return f / f.std() * amp


def make_raw(data_uv, label):
    info = mne.create_info([f"{label}_{c}" for c in CH], SFREQ, "eeg")
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def continuous_underlying(rng):
    """A continuous, resamplable underlying signal shared by both subjects:
    a 6 Hz driver + a coupled alpha component. Returned as a function of
    time (via interpolation) so subject B can be sampled on a misaligned
    clock."""
    t = np.arange(N) / SFREQ
    driver = np.sin(2 * np.pi * SYNC_HZ * t)          # shared flicker/SSVEP
    coupled_alpha = band_noise(ALPHA_BAND, rng, 1.0)  # shared alpha coupling
    return t, driver, coupled_alpha


def sample_at(t_query, t_grid, series):
    return np.interp(t_query, t_grid, series)


def run_case(name, check_fn):
    try:
        check_fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False


def build_pair(offset_s, drift_rate, seed=0):
    """A and B share the same underlying 6 Hz driver + coupled alpha, but B
    is sampled on a clock misaligned by (offset_s, drift_rate). Returns
    preprocessed raws."""
    rng = np.random.default_rng(seed)
    t, driver, coupled = continuous_underlying(rng)
    drv_amp, cpl_amp, noise_amp = 30.0, 25.0, 10.0

    # A on the true grid
    shared_a = drv_amp * driver + cpl_amp * coupled
    noise_a = np.stack([rng.standard_normal(N) * noise_amp for _ in range(4)])
    data_a = TOPO_A[:, None] * shared_a[None, :] + noise_a

    # B sampled at misaligned times t_b = t*(1+drift) + offset
    t_b = t * (1.0 + drift_rate) + offset_s
    driver_b = sample_at(t_b, t, driver)
    coupled_b = sample_at(t_b, t, coupled)
    shared_b = drv_amp * driver_b + cpl_amp * coupled_b
    noise_b = np.stack([rng.standard_normal(N) * noise_amp for _ in range(4)])
    data_b = TOPO_B[:, None] * shared_b[None, :] + noise_b

    raw_a = preprocess(make_raw(data_a, "A"), subject_label="A")
    raw_b = preprocess(make_raw(data_b, "B"), subject_label="B")
    return raw_a, raw_b


def alpha_plv(raw_a, raw_b):
    n = min(raw_a.n_times, raw_b.n_times)
    aa = analytic_signal(prefilter_raw_for_band(raw_a, ALPHA_BAND), n_samples=n)
    ab = analytic_signal(prefilter_raw_for_band(raw_b, ALPHA_BAND), n_samples=n)
    good = np.ones(n, dtype=bool)
    return plv_masked(aa, ab, good).mean()


def main():
    results = []

    TRUE_OFFSET = 0.030      # 30 ms
    TRUE_DRIFT = 5e-4        # 500 ppm -> ~100 ms over 200 s

    raw_a, raw_b = build_pair(TRUE_OFFSET, TRUE_DRIFT, seed=0)

    est_offset, est_drift = estimate_sync_offset_drift(
        raw_a, raw_b, SYNC_HZ, bandwidth=1.0)
    print(f"  injected: offset={TRUE_OFFSET*1000:.1f}ms  drift={TRUE_DRIFT*1e6:.0f}ppm")
    print(f"  estimated: offset={est_offset*1000:.2f}ms  drift={est_drift*1e6:.1f}ppm")

    def check_offset_recovered():
        assert abs(est_offset - TRUE_OFFSET) < 0.003, \
            f"offset off by {(est_offset-TRUE_OFFSET)*1000:.2f}ms (>3ms)"

    def check_drift_recovered():
        assert abs(est_drift - TRUE_DRIFT) < 5e-5, \
            f"drift off by {(est_drift-TRUE_DRIFT)*1e6:.1f}ppm (>50ppm)"

    results.append(run_case("estimate recovers injected offset", check_offset_recovered))
    results.append(run_case("estimate recovers injected drift", check_drift_recovered))

    # PLV before vs after correction
    plv_before = alpha_plv(raw_a, raw_b)
    raw_b_corr = apply_sync_correction(raw_b, est_offset, est_drift)
    plv_after = alpha_plv(raw_a, raw_b_corr)
    # reference ceiling: a perfectly-aligned pair (no injected misalignment)
    raw_a0, raw_b0 = build_pair(0.0, 0.0, seed=0)
    plv_aligned = alpha_plv(raw_a0, raw_b0)
    print(f"  alpha PLV: misaligned={plv_before:.3f}  corrected={plv_after:.3f}  "
          f"aligned-reference={plv_aligned:.3f}")

    def check_correction_recovers_coupling():
        assert plv_after > plv_before + 0.1, \
            (f"correction did not restore coupling "
             f"(before={plv_before:.3f}, after={plv_after:.3f})")

    def check_correction_reaches_reference():
        assert abs(plv_after - plv_aligned) < 0.1, \
            (f"corrected PLV ({plv_after:.3f}) not close to the aligned "
             f"reference ({plv_aligned:.3f})")

    results.append(run_case("correction restores coupling destroyed by misalignment",
                             check_correction_recovers_coupling))
    results.append(run_case("corrected PLV reaches the aligned-reference ceiling",
                             check_correction_reaches_reference))

    # drift-only mode: recovers drift, leaves offset at 0
    est_off_do, est_drift_do = estimate_sync_offset_drift(
        raw_a, raw_b, SYNC_HZ, bandwidth=1.0, drift_only=True)

    def check_drift_only_zeroes_offset():
        assert est_off_do == 0.0, f"drift-only returned nonzero offset {est_off_do}"

    def check_drift_only_keeps_drift():
        assert abs(est_drift_do - TRUE_DRIFT) < 5e-5, \
            f"drift-only drift off by {(est_drift_do-TRUE_DRIFT)*1e6:.1f}ppm"

    results.append(run_case("drift-only forces offset to zero", check_drift_only_zeroes_offset))
    results.append(run_case("drift-only still recovers the drift", check_drift_only_keeps_drift))

    n_pass = sum(results)
    n_total = len(results)
    print()
    print(f"{n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
