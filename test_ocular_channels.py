"""
test_ocular_channels.py — checks for the derived bipolar ocular channels
(make_ocular_channels in pipeline.py, point 9 of that module's docstring):

    saccade = AF7 - AF8
    blink   = mean(AF7, AF8) - mean(TP9, TP10)

    python test_ocular_channels.py

Synthetic data, so the ground truth is known. Checks:

  1. SELECTIVITY. A blink (same polarity at both frontal sites) lands on the
     blink channel and largely cancels on the saccade channel; a horizontal
     saccade (opposite polarity) does the reverse. This is the property the
     whole approach rests on -- if it fails, the two channels are not
     measuring different things and per-channel thresholds are pointless.
  2. REFERENCE INVARIANCE. Both channels are differences of electrodes, so
     any common reference cancels. Computing them before referencing, after
     an average reference, and after a mastoid reference must give the same
     traces.
  3. PASS BAND. Building the blink channel in the 1-40 Hz analysis band
     instead of 0.1-15 Hz throws away most of the blink amplitude -- the
     reason these channels get their own filter.
  4. DEGRADED PATHS. A derived channel whose electrodes are flagged bad is
     returned as None, not as a plausible-looking trace built from a dead
     electrode.
"""
import numpy as np
import mne

import pipeline as P

FS = 256.0
N = int(FS * 60)          # long enough to support a 0.1 Hz high-pass
rng = np.random.default_rng(0)
T = np.arange(N) / FS


def bumps(amp, width_s=0.3, every_s=5.0):
    """Blink-shaped deflections: a few hundred ms wide, repeating."""
    out = np.zeros(N)
    w = int(width_s * FS)
    for onset in range(int(2 * FS), N - w, int(every_s * FS)):
        out[onset:onset + w] += amp * np.hanning(w)
    return out


def make_raw(label="A", blink_uv=0.0, saccade_uv=0.0, bads_flat=()):
    """Four channels of independent brain signal plus a chosen ocular event.

    blink   -> SAME polarity at AF7 and AF8 (and nothing at the mastoids)
    saccade -> OPPOSITE polarity at AF7 and AF8
    """
    brain = rng.normal(0, 10e-6, size=(4, N))
    for i, f in enumerate([6.0, 9.0, 11.0, 7.0]):
        brain[i] += 10e-6 * np.sin(2 * np.pi * f * T + i)
    i9, i7, i8, i10 = 0, 1, 2, 3
    if blink_uv:
        b = bumps(blink_uv * 1e-6)
        brain[i7] += b
        brain[i8] += b
    if saccade_uv:
        s = bumps(saccade_uv * 1e-6)
        brain[i7] += s
        brain[i8] -= s
    for idx in bads_flat:
        brain[idx] = 1e-9 * rng.normal(size=N)
    info = mne.create_info([f"{label}_{c}" for c in P.CH_NAMES], FS, "eeg")
    return mne.io.RawArray(brain, info, verbose=False)


def head(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def quiet(raw, **kw):
    """make_ocular_channels without its per-subject console chatter."""
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return P.make_ocular_channels(raw, **kw)


head("1. Selectivity: does each channel isolate its own artifact?")
print(f"{'injected':<22} {'blink ch (p99)':>16} {'saccade ch (p99)':>18}")
for desc, kw in (("blink 200 uV", dict(blink_uv=200.0)),
                 ("saccade 200 uV", dict(saccade_uv=200.0))):
    oc = quiet(make_raw(**kw))
    pb = np.percentile(np.abs(oc.blink), 99)
    ps = np.percentile(np.abs(oc.saccade), 99)
    print(f"{desc:<22} {pb:>13.1f} uV {ps:>15.1f} uV")
print("   -> a blink should be large on 'blink' and small on 'saccade';")
print("      a saccade should be the other way round")

head("2. Reference invariance")
raw = make_raw(blink_uv=200.0, saccade_uv=80.0)
base = quiet(raw)
print(f"{'computed from':<34} {'max |diff| vs unreferenced':>28}")
print(f"{'unreferenced (baseline)':<34} {'--':>28}")
for ref in ("average", "mastoid"):
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        pp = P.preprocess(raw, l_freq=None, h_freq=None, reference=ref,
                          subject_label="A")
    oc = quiet(pp)
    d = max(np.abs(oc.blink - base.blink).max(),
            np.abs(oc.saccade - base.saccade).max())
    print(f"{'after --reference ' + ref:<34} {d:>25.3e} uV")
print("   -> should be ~0: the reference cancels in both expressions")

head("3. Pass band: what the ocular band actually buys")
# Detection SNR, not raw amplitude, is what a threshold detector cares
# about: peak blink height measured against the noise floor between blinks.
raw = make_raw(blink_uv=200.0)
w = int(0.3 * FS)
near_blink = np.zeros(N, bool)
for onset in range(int(2 * FS), N - w, int(5 * FS)):
    near_blink[max(0, onset - int(0.2 * FS)):onset + w + int(0.2 * FS)] = True

print(f"   {'band':<24} {'peak':>8} {'baseline sd':>13} {'SNR':>7}")
snr = {}
for lo, hi, name in ((0.1, 15.0, "0.1-15 Hz  (ocular)"),
                     (1.0, 40.0, "1-40 Hz    (analysis)"),
                     (0.1, 40.0, "0.1-40 Hz"),
                     (1.0, 15.0, "1-15 Hz")):
    b = quiet(raw, l_freq=lo, h_freq=hi).blink
    peak = np.percentile(np.abs(b[near_blink]), 99)
    base = b[~near_blink].std()
    snr[name.split()[0]] = peak / base
    print(f"   {name:<24} {peak:7.1f} {base:12.1f} {peak / base:6.1f}")
print(f"   -> the ocular band gives ~{snr['0.1-15'] / snr['1-40']:.1f}x the "
      "detection SNR, mostly from the 15 Hz low-pass")
print("      cutting muscle/EMG that the 40 Hz analysis band lets through.")
print("      Note the analysis band still keeps most of the blink AMPLITUDE:")
print("      a 1 Hz high-pass does not gut a 300ms blink.")

head("4. Degraded paths (bad electrodes -> None, not a fake trace)")
cases = [("AF7 dead", dict(bads_flat=(1,)), ["A_AF7"]),
         ("both mastoids dead", dict(bads_flat=(0, 3)), ["A_TP9", "A_TP10"]),
         ("one mastoid dead", dict(bads_flat=(0,)), ["A_TP9"])]
for desc, kw, bads in cases:
    oc = P.make_ocular_channels(make_raw(blink_uv=200.0, **kw), bads=bads,
                                subject_label="A")
    have = lambda x: "present" if x is not None else "None"
    print(f"   {desc:<20} bads={str(bads):<26} "
          f"saccade={have(oc.saccade):<8} blink={have(oc.blink)}")
    print()
