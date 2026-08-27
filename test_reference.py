"""
test_reference.py — checks for the --reference average|mastoid re-referencing
added in pipeline.py (see point 8 of that module's docstring).

Synthetic data, so the ground truth is known: four channels of distinct
"brain" signal that all share ONE ocular common-mode component, which is what
the Muse actually records (every channel is X - FPZ, and FPZ sits over the
eyes).

    python test_reference.py

Checks:
  1. Both references cancel the FPZ-borne ocular signal. Measured as the
     correlation of each output channel with the KNOWN injected waveform --
     not as "mean across channels", which is zero by definition under an
     average reference and so cannot tell the two schemes apart.
  2. Linked mastoid makes TP9/TP10 exact mirror images (corr = -1), the
     reference artifact that leaves only AF7/AF8 independent; and the
     4-channel average reference measurably mixes AF7 and AF8 into each
     other, which mastoid does not.
  3. The degraded paths (one mastoid bad, both bad, everything bad) warn and
     degrade instead of crashing.
  4. Under a mastoid reference the TP9 and TP10 rows/columns of a 4x4
     cross-brain matrix are EXACT duplicates for both PLV and adjusted
     circular correlation -- the redundancy that motivates restricting the
     metrics to AF7/AF8 (--analysis-channels, point 11).
"""
import numpy as np
import mne

import pipeline as P

FS = 256.0
N = int(FS * 30)
rng = np.random.default_rng(0)


def ocular_waveform():
    """Blink-like bumps: the ground-truth artifact injected below."""
    w = np.zeros(N)
    width = int(0.3 * FS)
    for onset in range(int(2 * FS), N - int(FS), int(4 * FS)):
        w[onset:onset + width] += 300e-6 * np.hanning(width)
    return w


def make_raw(label="A", tp9_bad=False, tp10_bad=False, af_bad=False):
    """Four channels of distinct brain signal sharing one FPZ ocular term."""
    t = np.arange(N) / FS
    brain = rng.normal(0, 20e-6, size=(4, N))
    for i, f in enumerate([6.0, 9.0, 11.0, 7.0]):
        brain[i] += 15e-6 * np.sin(2 * np.pi * f * t + i)
    data = brain - ocular_waveform()  # every channel is (X - FPZ)
    dead = 1e-9 * rng.normal(size=N)
    if tp9_bad:
        data[0] = dead
    if tp10_bad:
        data[3] = dead
    if af_bad:
        data[1] = dead
        data[2] = dead
    info = mne.create_info([f"{label}_{c}" for c in P.CH_NAMES], FS, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def corr_row(raw, target):
    return "  ".join(
        f"{n.split('_')[-1]}={abs(np.corrcoef(d, target)[0, 1]):.3f}"
        for n, d in zip(raw.ch_names, raw.get_data()))


def head(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


# l_freq/h_freq are None throughout: these checks are about the reference
# arithmetic, and filtering would only blur what is being measured.
head("1. FPZ ocular common-mode cancellation")
ocular = ocular_waveform()
print("   correlation of each channel with the injected ocular waveform")
print(f"   before  : {corr_row(make_raw(), ocular)}")
for ref in ("average", "mastoid"):
    out = P.preprocess(make_raw(), l_freq=None, h_freq=None, reference=ref,
                       subject_label="A")
    print(f"   {ref:8s}: {corr_row(out, ocular)}")
print("   -> both should drop from ~0.90 to ~0.01: FPZ cancels either way")

head("2. What each reference leaves behind")
m = P.preprocess(make_raw(), l_freq=None, h_freq=None, reference="mastoid",
                 subject_label="A")
a = P.preprocess(make_raw(), l_freq=None, h_freq=None, reference="average",
                 subject_label="A")


def ch(raw, name):
    return raw.get_data()[raw.ch_names.index(f"A_{name}")]


print(f"   mastoid: corr(TP9, TP10) = "
      f"{np.corrcoef(ch(m, 'TP9'), ch(m, 'TP10'))[0, 1]:+.6f}   "
      "(expect -1: reference artifact, they are mirror images)")
print(f"   mastoid: corr(AF7, AF8)  = "
      f"{np.corrcoef(ch(m, 'AF7'), ch(m, 'AF8'))[0, 1]:+.6f}")
print(f"   average: corr(AF7, AF8)  = "
      f"{np.corrcoef(ch(a, 'AF7'), ch(a, 'AF8'))[0, 1]:+.6f}   "
      "(the 4-channel average mixes the frontals into each other)")

head("3. Degraded mastoid paths (must warn, not crash)")
for desc, kwargs in [("one mastoid dead (TP9)", dict(tp9_bad=True)),
                     ("both mastoids dead", dict(tp9_bad=True, tp10_bad=True)),
                     ("every channel dead", dict(tp9_bad=True, tp10_bad=True,
                                                 af_bad=True))]:
    print(f"-- {desc}:")
    out = P.preprocess(make_raw(**kwargs), l_freq=None, h_freq=None,
                       reference="mastoid", subject_label="A")
    print(f"   bads = {out.info['bads']}")
    print()

head("4. Mastoid reference makes TP9/TP10 metrics literally redundant")
# Point 11: under a linked-mastoid reference the two mastoid channels are
# mirror images, and both PLV and adjusted circular correlation are invariant
# to the pi phase shift that separates them -- so their rows/columns in a 4x4
# cross-brain matrix are exact copies, not merely similar.
def analytic_of(raw, band=(8.0, 13.0)):
    banded = P.prefilter_raw_for_band(raw, band)
    return P.analytic_signal(banded)


ra = P.preprocess(make_raw("A"), reference="mastoid", subject_label="A")
rb = P.preprocess(make_raw("B"), reference="mastoid", subject_label="B")
aa, ab = analytic_of(ra), analytic_of(rb)
good = np.ones(min(aa.shape[1], ab.shape[1]), bool)
aa, ab = aa[:, :len(good)], ab[:, :len(good)]
for name, fn in (("PLV", P.plv_masked),
                 ("circ-corr (adjusted)", P.circ_corr_adjusted_masked)):
    m = fn(aa, ab, good)
    i9, i10 = 0, 3
    print(f"   {name:<22} row TP9 vs row TP10: max|diff| = "
          f"{np.abs(m[i9] - m[i10]).max():.2e}")
    print(f"   {'':<22} col TP9 vs col TP10: max|diff| = "
          f"{np.abs(m[:, i9] - m[:, i10]).max():.2e}")
print("   -> ~1e-16 means the two mastoid channels contribute duplicate")
print("      tests, so a 4x4 matrix under this reference has 9 distinct")
print("      cells, not 16. Hence --analysis-channels auto -> AF7/AF8.")
