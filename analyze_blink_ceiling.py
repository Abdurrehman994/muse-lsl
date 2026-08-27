"""
analyze_blink_ceiling.py — how long can a clean window be if every blink is
masked?

    python analyze_blink_ceiling.py

Motivation. The whole point of better artifact handling was to keep LARGER
continuous chunks, because short windows systematically inflate PLV (point 5
of pipeline.py: 0.286 at 2s epochs vs 0.022 over the whole recording, in the
same data). So the question that decides between masking and regression is
not "how much data survives" but "how long are the surviving pieces".

This script answers it directly, and independently of any threshold tuning.
It runs ONLY the velocity blink detector (pipeline.velocity_event_mask) --
no peak-to-peak criterion, no EEG amplitude criterion, no padding -- so what
it measures is the ceiling imposed by the blink rate itself. A perfect
detector cannot beat these numbers; every extra rejection criterion can only
make them worse.

The key column is JOINT_max. In a hyperscanning design the usable mask is the
INTERSECTION of two people's clean stretches, so both participants' blink
schedules apply at once. An individual can have a 50s blink-free stretch
while the dyad has none.

Output columns:
    A_max_gap / B_max_gap   longest blink-free stretch for each subject
    JOINT_max               longest stretch clean in BOTH -- the real ceiling
    joint_clean%            fraction of samples clean in both
    rate/min                detected blinks per minute (sanity check: normal
                            spontaneous blink rate is roughly 10-20/min, so a
                            number far outside that means the detector is
                            mistuned, not that the participant is unusual)
"""
import contextlib
import glob
import io
import os

import numpy as np

import pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
K = 10.0  # threshold = median + K*MAD of the velocity distribution


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def blink_mask(path, label):
    """(mask, fs, n_blinks) from the velocity detector alone, or (None,)*3."""
    try:
        onset = P.load_stimulus_onset(path)
    except Exception:
        onset = None  # missing or malformed marker sidecar
    raw, _ = quiet(P.load_csv_to_raw, path, label, onset_s=onset)
    pp = quiet(P.preprocess, raw, h_freq=40.0, subject_label=label)
    oc = quiet(P.make_ocular_channels, raw, subject_label=label,
               bads=pp.info["bads"])
    if oc.blink is None:
        return None, None, None
    thr = quiet(P.ocular_thresholds, oc, k=K)
    mask, kept, _ = P.velocity_event_mask(oc.blink, oc.fs,
                                          thr["blink_velocity"])
    return mask, oc.fs, kept


def longest_true_s(mask, fs):
    runs = P.mask_runs(mask)
    return max([(e - s) / fs for s, e in runs], default=0.0)


def main():
    stamps = sorted({os.path.basename(f).rsplit("_", 2)[0]
                     for f in glob.glob(os.path.join(ROOT, "recordings",
                                                     "*_D1_A1.csv"))})
    hdr = ("%-16s %10s %10s %11s %13s %9s %9s"
           % ("session", "A_max_gap", "B_max_gap", "JOINT_max",
              "joint_clean%", "A_rate", "B_rate"))
    print(hdr)
    print("-" * len(hdr))

    joints = []
    for stamp in stamps:
        a = os.path.join(ROOT, "recordings", stamp + "_D1_A1.csv")
        b = next((os.path.join(ROOT, "recordings", stamp + s)
                  for s in ("_D1_6F.csv", "_CB_54.csv")
                  if os.path.exists(os.path.join(ROOT, "recordings", stamp + s))),
                 None)
        if b is None:
            continue
        try:
            ma, fs, ka = blink_mask(a, "A")
            mb, _, kb = blink_mask(b, "B")
        except Exception as e:
            print("%-16s   (failed: %s)" % (stamp, type(e).__name__))
            continue
        if ma is None or mb is None:
            print("%-16s   (no blink channel -- electrodes flagged bad)" % stamp)
            continue
        n = min(len(ma), len(mb))
        ma, mb = ma[:n], mb[:n]
        good = ~(ma | mb)
        dur = n / fs
        joint = longest_true_s(good, fs)
        joints.append(joint)
        print("%-16s %9.1fs %9.1fs %10.1fs %12.1f%% %8.1f %9.1f"
              % (stamp, longest_true_s(~ma, fs), longest_true_s(~mb, fs),
                 joint, 100 * good.mean(), 60 * ka / dur, 60 * kb / dur))

    if joints:
        print()
        print("JOINT ceiling across dyads: %.1f-%.1fs (median %.1fs)"
              % (min(joints), max(joints), np.median(joints)))
        print()
        print("Read this against pipeline.py point 5: PLV measured on ~10s")
        print("windows runs about 6x the whole-recording value in this data.")
        print("So masking every blink keeps most of the SAMPLES (>90%) while")
        print("capping window length in exactly the range where the metric is")
        print("least trustworthy -- which is the case for correcting blinks by")
        print("regression rather than removing them.")


if __name__ == "__main__":
    main()
