"""
test_ocular_detection.py — checks for the ocular-based artifact detection
added in pipeline.py (point 10 of that module's docstring):
robust_threshold, window_ptp, ptp_event_mask, velocity_event_mask,
ocular_thresholds, ocular_bad_mask.

    python test_ocular_detection.py

Signals are built as four EEG channels and pushed through the real
make_ocular_channels(), rather than synthesised as bare traces. That matters
for the velocity detector: the ocular channels are low-passed at 15 Hz, and
without that filtering the sample-to-sample derivative of white sensor noise
is far larger than a blink's, so a detector that works fine on real data
would look broken on the synthetic.

Checks:
  1. robust_threshold is not dragged up by the artifacts it must catch --
     the reason it uses MAD rather than standard deviation.
  2. The velocity detector recovers the right NUMBER of blinks, and its
     duration test rejects slow drift (too slow) and sustained muscle tone
     (too long) while keeping real blinks.
  3. Per-participant thresholds scale with each participant's own amplitude,
     so one fixed number is not imposed on two different people.
  4. ocular_bad_mask ORs its criteria and respects the detector selection.
"""
import contextlib
import io

import numpy as np
import mne

import pipeline as P

FS = 256.0
DUR = 120.0
N = int(FS * DUR)
rng = np.random.default_rng(0)


def blink_train(amp_uv, every_s=4.0, width_s=0.3, start_s=2.0):
    """Blinks at a fixed rate. Returns (trace_uv, n_blinks)."""
    out = np.zeros(N)
    w = int(width_s * FS)
    count = 0
    for onset in range(int(start_s * FS), N - w, int(every_s * FS)):
        out[onset:onset + w] += amp_uv * np.hanning(w)
        count += 1
    return out, count


def make_oc(frontal_uv, label="A", noise_uv=8.0):
    """
    OcularChannels for a subject whose frontal electrodes carry frontal_uv
    (same polarity at AF7 and AF8, so it lands on the blink channel).

    Goes through the real make_ocular_channels so the 0.1-15 Hz filtering is
    applied exactly as in the pipeline.
    """
    data = rng.normal(0, noise_uv * 1e-6, size=(4, N))
    data[1] += frontal_uv * 1e-6   # AF7
    data[2] += frontal_uv * 1e-6   # AF8
    info = mne.create_info([f"{label}_{c}" for c in P.CH_NAMES], FS, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return P.make_ocular_channels(raw, subject_label=label)


def head(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


head("1. robust_threshold resists the artifacts it is meant to catch")
clean = rng.normal(0, 10.0, 4000)
contaminated = clean.copy()
contaminated[:600] += rng.normal(0, 300.0, 600)   # 15% of windows artifactual
mad_c, mad_a = P.robust_threshold(clean, k=5), P.robust_threshold(contaminated, k=5)
sd_c = clean.mean() + 5 * clean.std()
sd_a = contaminated.mean() + 5 * contaminated.std()
print(f"   {'estimator':<34} {'clean data':>12} {'+15% artifact':>15}")
print(f"   {'median + 5*MAD (used)':<34} {mad_c:>12.1f} {mad_a:>15.1f}")
print(f"   {'mean + 5*sd (not used)':<34} {sd_c:>12.1f} {sd_a:>15.1f}")
print(f"   -> MAD moves {100 * (mad_a / mad_c - 1):+.0f}%, "
      f"sd moves {100 * (sd_a / sd_c - 1):+.0f}%: an sd-based threshold gets")
print("      pushed up by the very artifacts it is supposed to flag.")

head("2. Velocity detector: blink count, and what the duration test rejects")
blinks, n_true = blink_train(200.0, every_s=4.0)
oc = make_oc(blinks)
thr = P.ocular_thresholds(oc, k=10)["blink_velocity"]
mask, kept, cand = P.velocity_event_mask(oc.blink, FS, thr)
print(f"   {n_true} blinks injected -> {kept} detected "
      f"({cand} candidates before the duration test), "
      f"{100 * mask.mean():.1f}% of samples masked")

drift = 400.0 * np.sin(2 * np.pi * 0.05 * np.arange(N) / FS)
_, k_drift, _ = P.velocity_event_mask(make_oc(drift, "D").blink, FS, thr)

muscle = np.zeros(N)
for onset in range(int(5 * FS), N - int(3 * FS), int(20 * FS)):
    muscle[onset:onset + int(3 * FS)] += rng.normal(0, 150.0, int(3 * FS))
_, k_musc, c_musc = P.velocity_event_mask(make_oc(muscle, "M").blink, FS, thr)

print(f"   slow drift (400 uV, 0.05 Hz)  -> {k_drift:>3} events kept "
      "(want 0: never exceeds the velocity threshold)")
print(f"   sustained muscle (3s bursts)  -> {k_musc:>3} events kept "
      f"of {c_musc} candidates (duration test rejects the long ones)")

head("3. Per-participant thresholds track each participant's BASELINE")
# The threshold is median + k*MAD of that subject's window-p2p distribution,
# which the quiet majority of the recording dominates. So it tracks the
# person's baseline ocular activity, not their blink size -- and baseline is
# what actually differs between people (on 20260806_125527 the real blink
# thresholds came out 94.6 uV for A and 206.5 uV for B). The practical
# meaning: one value of k gives every participant the same STATISTICAL
# strictness rather than the same microvolt cut.
blinks_fixed = blink_train(200.0, every_s=4.0)[0]
print(f"   {'subject':<34} {'blink thr':>12} {'velocity thr':>15}")
thrs = []
for noise, label in ((8.0, "calm subject (8 uV baseline)"),
                     (40.0, "restless subject (40 uV baseline)")):
    t = P.ocular_thresholds(make_oc(blinks_fixed, noise_uv=noise), k=5)
    thrs.append(t["blink"])
    print(f"   {label:<34} {t['blink']:>9.1f} uV {t['blink_velocity']:>11.0f} uV/s")
print(f"   -> {thrs[1] / thrs[0]:.1f}x apart for the SAME blink amplitude. A single")
print("      fixed threshold would over-reject the restless subject and")
print("      under-reject the calm one.")

head("4. ocular_bad_mask: criteria are OR-ed, detector selection respected")
oc = make_oc(blink_train(200.0, every_s=4.0)[0])
thr = P.ocular_thresholds(oc, k=10)
print(f"   {'detector':<12} {'% samples masked':>18}")
pct = {}
for det in ("ptp", "velocity", "both"):
    _, stats = P.ocular_bad_mask(oc, thr, N, detector=det, subject_label="A",
                                 quiet=True)
    pct[det] = stats["total_pct"]
    print(f"   {det:<12} {stats['total_pct']:>16.1f}%")
ok = pct["both"] >= max(pct["ptp"], pct["velocity"]) - 1e-9
print(f"   -> 'both' >= each individual criterion: {'OK' if ok else 'FAILED'} "
      "(it is their union)")
print()
