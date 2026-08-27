"""
compare_references.py — run every dyad recording through pipeline.py under
BOTH --reference settings and tabulate the result.

    python compare_references.py [--out-dir out/ref_compare]

Writes one pipeline output directory per (session, reference) plus a summary
table at <out-dir>/reference_comparison.txt.

The columns that matter are joint% and longest_s. Clean fraction says how
much data survived; longest continuous jointly-clean run says whether it
survived in usable-length pieces, and that is the one to watch: short windows
systematically inflate PLV (point 5 of pipeline.py's docstring), so a
preprocessing change that keeps the same fraction but shatters it into
shorter runs has made things worse, not better.

Both references are run with --analysis-channels all so the PLV columns
compare like with like. Note that under a mastoid reference the TP9/TP10
rows and columns are exact duplicates (pipeline.py point 11), so the 4x4
mean for those rows is diluted by redundant cells -- read the mastoid PLV
numbers here as a comparison aid, not as the value you would report.

Caveat when reading the table: the artifact detector currently thresholds
peak-to-peak amplitude on the EEG channels, and an average reference
suppresses whatever is common to all four channels by construction -- so it
partly hides shared artifacts from the detector. Retention differences
between the two references therefore reflect detector sensitivity as much as
data quality, and cannot settle which reference is better on their own.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PARTNER_SUFFIXES = ("_D1_6F.csv", "_CB_54.csv")


def find_sessions():
    """(stamp, subject_a_csv, subject_b_csv) for every complete dyad."""
    out = []
    for a in sorted(glob.glob(os.path.join(ROOT, "recordings", "*_D1_A1.csv"))):
        stamp = os.path.basename(a).rsplit("_", 2)[0]
        for suffix in PARTNER_SUFFIXES:
            b = os.path.join(ROOT, "recordings", stamp + suffix)
            if os.path.exists(b):
                out.append((stamp, a, b))
                break
        else:
            print(f"  skipping {stamp}: no partner file")
    return out


def grab(txt, pattern, default=float("nan")):
    m = re.search(pattern, txt)
    return float(m.group(1)) if m else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join("out", "ref_compare"))
    p.add_argument("--references", nargs="+", default=["average", "mastoid"])
    args = p.parse_args()

    out_root = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for stamp, csv_a, csv_b in find_sessions():
        for ref in args.references:
            run = subprocess.run(
                [sys.executable, "pipeline.py", csv_a, csv_b,
                 "--reference", ref,
                 # pin the channel set: --analysis-channels defaults to
                 # 'auto', which would give the mastoid runs a 2x2 matrix and
                 # the average runs a 4x4 one, making the mean-PLV columns
                 # mean different things per row. Held at 'all' so the two
                 # references stay directly comparable here.
                 "--analysis-channels", "all",
                 "--out-dir", os.path.join(out_root, f"{stamp}_{ref}")],
                cwd=ROOT, capture_output=True, text=True)
            txt = run.stdout + run.stderr
            if run.returncode != 0:
                print(f"  FAILED {stamp} {ref} (rc={run.returncode})")
                print(txt[-1500:])
                continue
            rows.append((
                stamp, ref,
                grab(txt, r"Jointly clean.*?\(([\d.]+)%\)"),
                grab(txt, r"Jointly clean.*?: ([\d.]+)s"),
                grab(txt, r"Longest continuous jointly-clean run: ([\d.]+)s"),
                grab(txt, r"theta.*?mean PLV = ([\d.]+)"),
                grab(txt, r"alpha.*?mean PLV = ([\d.]+)"),
                grab(txt, r"beta.*?mean PLV = ([\d.]+)"),
            ))
            print(f"  done {stamp} {ref:<8s} joint={rows[-1][2]:.1f}% "
                  f"longest={rows[-1][4]:.1f}s")

    fmt = "%-16s %-8s %8s %9s %10s %7s %7s %7s"
    lines = [fmt % ("session", "ref", "joint%", "joint_s", "longest_s",
                    "theta", "alpha", "beta")]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append("%-16s %-8s %8.1f %9.1f %10.1f %7.3f %7.3f %7.3f" % r)
    table = "\n".join(lines)

    print()
    print(table)
    dest = os.path.join(out_root, "reference_comparison.txt")
    with open(dest, "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
