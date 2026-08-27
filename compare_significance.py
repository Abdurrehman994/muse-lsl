"""
compare_significance.py — does the ocular strategy change the INFERENCE?

    python compare_significance.py [--surrogates 200] [--sessions N]

Why this exists. compare_ocular_correction.py compares raw PLV across
strategies, and masking makes raw PLV rise (short windows inflate it, per
point 5). That was used to argue masking is unusable. But the pipeline does
not test raw PLV against zero -- it tests the real dyad against a null built
from the SAME data with the SAME window structure, and --match-null-length
(point 7b, on by default) explicitly equalises the sample counts. If a
strategy inflates the real value and its null by the same factor, the
p-values do not move and the conclusion is unaffected.

So the question is not "does masking change PLV" (it does) but "does masking
change what we conclude". This script answers that directly by comparing the
surrogate p-values, not the effect sizes.

What to read:
    mean_plv    the raw value -- expected to differ between strategies
    p_med       median of the 16 pairwise surrogate p-values
    p_min       smallest p across the 16 pairs
    raw<.05     pairs significant before correction (expect ~0.8 of 16 by chance)
    fdr         pairs surviving Benjamini-Hochberg

If p_med sits near 0.5 and raw<.05 near chance for every strategy, then the
strategies agree on the inference even where they disagree on the number,
and the window-length objection to masking does not affect the conclusion.

Note on resolution: with S surrogates the smallest achievable p is 1/(S+1).
FDR over 16 tests needs p below ~0.003 at the strictest rank, so S must be
several hundred for the fdr column to be able to fire at all. That is why
p_med/p_min are the columns to read at modest S, not fdr.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BANDS = ("theta", "alpha", "beta")

STRATEGIES = [
    ("none",    ["--ocular-correction", "none", "--artifact-source", "eeg"]),
    ("mask",    ["--ocular-correction", "none", "--artifact-source", "ocular",
                 "--ocular-detector", "velocity", "--ocular-k", "10"]),
    ("regress", ["--ocular-correction", "regress", "--regress-channels", "blink",
                 "--artifact-source", "eeg"]),
]


def sessions():
    out = []
    for a in sorted(glob.glob(os.path.join(ROOT, "recordings", "*_D1_A1.csv"))):
        stamp = os.path.basename(a).rsplit("_", 2)[0]
        for suffix in ("_D1_6F.csv", "_CB_54.csv"):
            b = os.path.join(ROOT, "recordings", stamp + suffix)
            if os.path.exists(b):
                out.append((stamp, a, b))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogates", type=int, default=200)
    ap.add_argument("--sessions", type=int, default=0,
                    help="limit to the first N sessions (0 = all)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these session stamps. Useful because the "
                         "comparison is slow and only the LONG sessions "
                         "discriminate: masking has to fragment a long "
                         "recording before window length can matter.")
    ap.add_argument("--out-dir", default=os.path.join("out", "sig_compare"))
    args = ap.parse_args()

    out_root = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    sess = sessions()
    if args.only:
        sess = [x for x in sess if x[0] in set(args.only)]
    if args.sessions:
        sess = sess[:args.sessions]

    hdr = ("%-16s %-8s %-6s %9s %8s %8s %9s %5s"
           % ("session", "strategy", "band", "mean_plv", "p_med", "p_min",
              "raw<.05", "fdr"))
    lines = [hdr, "-" * len(hdr)]
    print(hdr)
    print(lines[1])

    for stamp, a, b in sess:
        printed_session = False
        for name, flags in STRATEGIES:
            d = os.path.join(out_root, f"{stamp}_{name}")
            run = subprocess.run(
                [sys.executable, "pipeline.py", a, b,
                 "--surrogate", str(args.surrogates),
                 "--analysis-channels", "all",   # keep the pair count comparable
                 "--out-dir", d] + flags,
                cwd=ROOT, capture_output=True, text=True)
            txt = run.stdout + run.stderr
            if run.returncode != 0:
                row = "%-16s %-8s  run failed" % (
                    stamp if not printed_session else "", name)
                print(row)
                lines.append(row)
                printed_session = True
                continue

            fdr_counts = re.findall(r"PLV significant pairs: (\d+)/(\d+)", txt)
            for i, band in enumerate(BANDS):
                pfile = os.path.join(d, f"plv_p_within_{band}.npy")
                mfile = os.path.join(d, f"plv_{band}.npy")
                if not (os.path.exists(pfile) and os.path.exists(mfile)):
                    continue
                p = np.load(pfile).ravel()
                m = np.load(mfile).ravel()
                fdr = fdr_counts[i][0] if i < len(fdr_counts) else "-"
                row = ("%-16s %-8s %-6s %9.4f %8.3f %8.3f %5d/%-3d %5s"
                       % (stamp if not printed_session else "",
                          name if band == BANDS[0] else "", band,
                          float(np.mean(m)), float(np.median(p)),
                          float(np.min(p)), int((p < 0.05).sum()), p.size, fdr))
                print(row)
                lines.append(row)
                printed_session = True
            lines.append("")
            print()

    dest = os.path.join(out_root, "significance_comparison.txt")
    with open(dest, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
