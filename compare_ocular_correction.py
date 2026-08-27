"""
compare_ocular_correction.py — masking vs regression, across every dyad.

    python compare_ocular_correction.py

Runs each session under three ocular-artifact strategies and tabulates what
each one costs and buys:

    none      the existing pipeline: EEG amplitude thresholding only
    mask      detect blinks on the derived ocular channels and reject them
    regress   subtract beta * blink(t) per channel and KEEP the samples
    emcp      the faithful Gratton procedure (template blink detector,
              separate blink/saccade propagation factors), vertical EOG only
    emcp2ch   the same with both derived EOG channels, as the original
              usually runs -- expected to over-strip on four electrodes

The column that decides between them is longest_s, the longest continuous
JOINTLY-clean run. Clean fraction alone is misleading: masking blinks keeps
>90% of samples while shattering them into ~10s pieces, and short windows
inflate PLV badly (pipeline.py point 5). See analyze_blink_ceiling.py for why
that ~10s is a hard ceiling rather than a tuning failure.

TWO READING HAZARDS, both visible in the current results.

1. longest_s only measures what MASKING costs. On a session where the
   amplitude detector was rejecting nothing, 'none' and 'regress' show the
   same window length -- but they are not equivalent: 'none' still has the
   blinks in the data, it simply is not throwing anything away. Window
   length cannot distinguish those two, so on such sessions the regression
   rows show pure variance cost and no visible benefit. That is a limit of
   this table, not evidence that the correction did nothing.

2. Sessions where every electrode is railed (currently 20260730_131127 and
   20260805_125417) exit early under 'none'/'regress'/'emcp' with 0.0%, but
   the 'mask' strategy uses --artifact-source ocular, which REPLACES the EEG
   amplitude criterion -- so with no usable ocular channel it applies no
   rejection at all and reports ~100% "clean" on junk. Those rows carry the
   largest alpha values in the table (0.396, 0.242) and are meaningless.
   pipeline.py prints a warning for this case; ignore those rows.

Read the regress rows against pipeline.py point 12: the correction genuinely
restores long windows, but on this montage the regressor is built from the
same electrodes it corrects, so it also removes a large amount of real signal
(29-63% of variance outside any blink). var_kept reports how much survives,
averaged over channels and subjects -- treat a low value as a warning, not a
success.
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def out_root_dir():
    return os.path.join(ROOT, "out", "ocular_compare")

STRATEGIES = [
    ("none",    ["--ocular-correction", "none", "--artifact-source", "eeg"]),
    ("mask",    ["--ocular-correction", "none", "--artifact-source", "ocular",
                 "--ocular-detector", "velocity", "--ocular-k", "10"]),
    ("regress", ["--ocular-correction", "regress", "--regress-channels", "blink",
                 "--artifact-source", "eeg"]),
    ("emcp",    ["--ocular-correction", "emcp", "--emcp-channels", "blink",
                 "--artifact-source", "eeg"]),
    ("emcp2ch", ["--ocular-correction", "emcp", "--emcp-channels", "both",
                 "--artifact-source", "eeg"]),
]


def grab(txt, pat, default=float("nan")):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else default


def main():
    stamps = sorted({os.path.basename(f).rsplit("_", 2)[0]
                     for f in glob.glob(os.path.join(ROOT, "recordings",
                                                     "*_D1_A1.csv"))})
    hdr = "%-16s %-9s %8s %10s %9s %8s" % (
        "session", "strategy", "joint%", "longest_s", "var_kept", "alpha")
    lines = [hdr, "-" * len(hdr)]
    print(hdr)
    print("-" * len(hdr))

    for stamp in stamps:
        a = os.path.join(ROOT, "recordings", stamp + "_D1_A1.csv")
        b = next((os.path.join(ROOT, "recordings", stamp + s)
                  for s in ("_D1_6F.csv", "_CB_54.csv")
                  if os.path.exists(os.path.join(ROOT, "recordings", stamp + s))),
                 None)
        if b is None:
            continue
        printed = False
        for name, flags in STRATEGIES:
            run = subprocess.run(
                [sys.executable, "pipeline.py", a, b, "--out-dir",
                 os.path.join(ROOT, "out", "ocular_compare", f"{stamp}_{name}")]
                + flags,
                cwd=ROOT, capture_output=True, text=True)
            txt = run.stdout + run.stderr
            if run.returncode != 0:
                print("%-16s %-9s FAILED" % (stamp if not printed else "", name))
                printed = True
                continue
            kept = [float(v) for v in re.findall(r"=(\d+)%", txt)]
            row = "%-16s %-9s %8.1f %10.1f %9s %8.3f" % (
                stamp if not printed else "", name,
                grab(txt, r"Jointly clean.*?\((\d+\.\d+)%\)"),
                grab(txt, r"Longest continuous jointly-clean run: ([\d.]+)s"),
                f"{sum(kept) / len(kept):.0f}%" if kept else "-",
                grab(txt, r"alpha.*?mean PLV = ([\d.]+)"))
            print(row)
            lines.append(row)
            printed = True
        print()
        lines.append("")

    dest = os.path.join(out_root_dir(), "ocular_correction_comparison.txt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
