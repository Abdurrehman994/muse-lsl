"""
cooperative_task.py — block-design cue script for an ALONE vs COOPERATIVE
joint verbal puzzle (20-questions style) task.

Alternates ALONE / COOPERATIVE blocks, firing a distinct LSL marker at every
condition switch on the same "StimulusMarkers" stream that record_both.py
already watches for (same handshake as play_stimulus.py / stimulus_marker.py
/ the old eye_contact_task.py).

Rationale (see conversation notes / internship writeup): passive shared-video
viewing showed near-chance inter-brain PLV/circ-corr even after fixing the
measurement pipeline (epoch-length bias, then continuous artifact
rejection) -- expected, per the hyperscanning literature, since two people
independently watching one video is "neural entrainment" (a confound), not
genuine inter-brain synchrony. Cooperation (shared goal, mutual dependency,
verbal coordination) has the strongest and most replicated effect size in
the field (e.g. Czeszumski et al. 2022 meta-analysis, g=1.98 for cooperative
vs. non-cooperative). This script implements that as a joint 20-questions
game: the script picks a hidden target, shown ONLY to the experimenter, and
the two subjects cooperate verbally to guess it.

Task design
────────────────────────────────────────────────────────────────
  ALONE block (control): subjects sit quietly, no talking, no eye contact.
    Rules out task-unrelated shared arousal/entrainment as an explanation
    for any COOPERATIVE-block effect.
  COOPERATIVE block (experimental): the experimenter reveals a hidden target
    to THEMSELVES ONLY (keep your screen turned away from participants).
    Subjects take turns asking yes/no questions out loud; you answer
    "yes"/"no"/"sort of" until they guess it (or give up). Press ENTER when
    the round ends.

IBS workflow
────────────────────────────────────────────────────────────────
  Terminal 1:  python record_both.py 600
  Terminal 2:  python cooperative_task.py --reps 4

  1. When both scripts are running, press ENTER to begin.
  2. Blocks alternate automatically; a marker fires at the start of each.
  3. record_both.py captures every marker into its _markers.json sidecars.
  4. Analyze with: python compare_conditions.py <A.csv> <B.csv>

Usage:
  python cooperative_task.py                          # 60s alone blocks, 4 reps, default target list
  python cooperative_task.py --reps 6                  # more blocks -> better-powered contrast test
  python cooperative_task.py --alone-len 45            # shorter alone/baseline block
  python cooperative_task.py --targets my_targets.txt   # one target per line, instead of the built-in list
  python cooperative_task.py --start cooperative        # start with cooperative instead of alone
────────────────────────────────────────────────────────────────
"""

import argparse
import random
import time

from pylsl import StreamInfo, StreamOutlet, local_clock

try:
    import winsound
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False

MARKER_STREAM_NAME = "StimulusMarkers"

DEFAULT_TARGETS = [
    "elephant", "umbrella", "guitar", "volcano", "sandwich",
    "bicycle", "penguin", "telescope", "waterfall", "backpack",
    "cactus", "lighthouse", "kangaroo", "trampoline", "compass",
    "spaghetti", "octopus", "helicopter", "pineapple", "violin",
    "avalanche", "flamingo", "microscope", "hammock", "dandelion",
    "accordion", "porcupine", "snorkel", "windmill", "jellyfish",
]

CONDITIONS = {
    "alone": {
        "marker": "ALONE_start",
        "label": "ALONE — sit quietly, no talking, no eye contact",
        "beeps": 1,
    },
    "cooperative": {
        "marker": "COOPERATIVE_start",
        "label": "COOPERATIVE — joint 20-questions",
        "beeps": 2,
    },
}


def load_targets(path):
    if path is None:
        return list(DEFAULT_TARGETS)
    with open(path) as f:
        targets = [line.strip() for line in f if line.strip()]
    if not targets:
        raise ValueError(f"{path} contained no targets (one per line expected)")
    return targets


def build_outlet():
    info = StreamInfo(
        MARKER_STREAM_NAME, "Markers", 1, 0, "string", "stimulus_markers_001"
    )
    return StreamOutlet(info)


def beep(n, audio_enabled):
    if not (audio_enabled and _HAVE_WINSOUND):
        return
    for _ in range(n):
        winsound.Beep(880, 200)
        time.sleep(0.15)


def countdown_wait(seconds, label):
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        print(f"\r  {label}: {remaining:4.1f}s remaining ...", end="", flush=True)
        time.sleep(min(0.2, remaining))
    print(f"\r  {label}: done.                              ")


def run(alone_len, reps, start_condition, audio_enabled, targets, target_time_budget):
    order = ["alone", "cooperative"]
    if start_condition == "cooperative":
        order = ["cooperative", "alone"]

    rng = random.Random()
    remaining_targets = list(targets)
    rng.shuffle(remaining_targets)

    total_blocks = reps * len(order)

    print("=" * 60)
    print("Cooperative-task (ALONE vs COOPERATIVE) block script")
    print(f"  alone block length : {alone_len:.0f}s")
    print(f"  reps/condition     : {reps}  ({total_blocks} blocks total)")
    print(f"  starting with      : {order[0]}")
    print(f"  audio cues         : {'on' if audio_enabled and _HAVE_WINSOUND else 'off'}")
    print(f"  target pool        : {len(targets)} words")
    if reps < 3:
        print("  NOTE: --reps below 3 gives the later block-permutation "
              "contrast test very little resolution -- consider --reps 3+.")
    print("=" * 60)
    print("Make sure record_both.py is already running for the full session "
          "length.\n")

    outlet = build_outlet()
    if not outlet.have_consumers():
        print("Waiting for record_both.py to connect to the marker stream "
              "(up to 30s) ...")
        outlet.wait_for_consumers(30.0)
    if outlet.have_consumers():
        print("  Recorder connected.\n")
    else:
        print("  WARNING: no recorder connected after 30s — markers will "
              "NOT be captured. Proceeding anyway.\n")

    input("Press ENTER when both participants are ready to begin ...")

    block_num = 0
    for rep in range(reps):
        for cond_key in order:
            block_num += 1
            cond = CONDITIONS[cond_key]
            t_marker = local_clock()
            outlet.push_sample([cond["marker"]], t_marker)
            print(f"\n[{block_num}/{total_blocks}] >> {cond['label']}  "
                  f"(marker '{cond['marker']}' @ LSL t={t_marker:.3f})")
            beep(cond["beeps"], audio_enabled)

            if cond_key == "alone":
                countdown_wait(alone_len, "alone")
            else:
                if not remaining_targets:
                    remaining_targets = list(targets)
                    rng.shuffle(remaining_targets)
                target = remaining_targets.pop()
                print("\n" + "!" * 60)
                print("  EXPERIMENTER ONLY -- keep this screen turned away "
                      "from both participants.")
                print(f"  HIDDEN TARGET: {target.upper()}")
                print("!" * 60)
                print("  Tell participants: take turns asking yes/no "
                      "questions out loud to guess the hidden word together.")
                print(f"  Suggested time budget: ~{target_time_budget:.0f}s.")
                input("  Press ENTER when this round ends (guessed, or "
                      "given up) ...")

    print("\nAll blocks complete. Stop record_both.py now if it's still running.")
    print("Analyze with:  python compare_conditions.py <A.csv> <B.csv>")


def main():
    p = argparse.ArgumentParser(
        description="Block-design cue script for an ALONE vs COOPERATIVE "
                     "joint verbal puzzle task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alone-len", type=float, default=60.0,
                   help="seconds for each ALONE block (default 60)")
    p.add_argument("--reps", type=int, default=4,
                   help="repetitions of each condition (default 4). "
                        "compare_conditions.py's block-permutation contrast "
                        "test needs several reps per condition to have any "
                        "resolution -- 1-2 reps is not enough.")
    p.add_argument("--start", choices=["alone", "cooperative"], default="alone",
                   help="which condition to start with (default alone)")
    p.add_argument("--no-audio", action="store_true",
                   help="disable beep cues (console text only)")
    p.add_argument("--targets", default=None,
                   help="path to a text file of hidden targets, one per "
                        "line, instead of the built-in list")
    p.add_argument("--target-time-budget", type=float, default=90.0,
                   help="suggested (not enforced) seconds per COOPERATIVE "
                        "round, just printed as guidance (default 90)")
    args = p.parse_args()

    targets = load_targets(args.targets)
    run(args.alone_len, args.reps, args.start, not args.no_audio, targets,
        args.target_time_budget)


if __name__ == "__main__":
    main()
