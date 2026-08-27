"""
demo_ssvep_gate.py — one-command demonstration that the SSVEP pre-flight gate
(check_ssvep_control.py) correctly discriminates a valid positive control from
a dead recording. For showing the tool works (e.g. in a supervisor update)
without needing a fresh recording.

Runs the gate on two ground-truth sessions:
  - 20260805_125417  a real, strong 6 Hz SSVEP        -> expect GO
  - 20260811_113201  a recording with no flicker drive -> expect NO-GO

Usage:  python demo_ssvep_gate.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "check_ssvep_control.py")

CASES = [
    ("20260805_125417", "GO",    "real strong 6 Hz SSVEP (the one valid positive control)"),
    ("20260811_113201", "NO-GO", "no detectable flicker drive (dead recording)"),
]

for stamp, expected, desc in CASES:
    print("\n" + "#" * 70)
    print(f"# {stamp}  --  expect {expected}   ({desc})")
    print("#" * 70)
    rc = subprocess.run([sys.executable, GATE, "--session", stamp]).returncode
    got = "GO" if rc == 0 else "NO-GO/WEAK"
    print(f"\n>>> gate returned: {got}   (expected {expected})   "
          f"{'[OK]' if got.startswith(expected.split('/')[0]) or (expected=='GO' and got=='GO') else '[MISMATCH]'}")
