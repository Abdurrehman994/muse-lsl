"""
eye_contact_task.py — block-design cue script for the mutual eye-contact vs.
gaze-aversion task.

Alternates EYE_CONTACT / GAZE_AVERSION blocks, firing a distinct LSL marker at
every condition switch on the same "StimulusMarkers" stream that
record_both.py already watches for (same handshake as play_stimulus.py /
stimulus_marker.py).

Because both subjects are looking at each other, not a screen, the cue is an
audio beep pattern the experimenter can also read off the console:
    1 beep  -> "look at your partner"   (EYE_CONTACT_start)
    2 beeps -> "look away"              (GAZE_AVERSION_start)

IBS workflow
────────────────────────────────────────────────────────────────
  Terminal 1:  python record_both.py 330
  Terminal 2:  python eye_contact_task.py --block-len 30 --reps 5

  1. When both scripts are running, press ENTER to begin.
  2. Blocks alternate automatically; a marker fires at each switch.
  3. record_both.py captures every marker into its _markers.json sidecars.
────────────────────────────────────────────────────────────────

Usage:
  python eye_contact_task.py                        # 30s blocks x 5 reps/condition
  python eye_contact_task.py --block-len 45 --reps 6
  python eye_contact_task.py --start gaze_aversion   # start with gaze aversion instead
  python eye_contact_task.py --no-audio              # console cues only, no beeps
"""

import argparse
import time

from pylsl import StreamInfo, StreamOutlet, local_clock

try:
    import winsound
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False

MARKER_STREAM_NAME = "StimulusMarkers"

CONDITIONS = {
    "eye_contact": {
        "marker": "EYE_CONTACT_start",
        "label": "EYE CONTACT — look at your partner",
        "beeps": 1,
    },
    "gaze_aversion": {
        "marker": "GAZE_AVERSION_start",
        "label": "GAZE AVERSION — look away",
        "beeps": 2,
    },
}


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


def run(block_len, reps, start_condition, audio_enabled):
    order = ["eye_contact", "gaze_aversion"]
    if start_condition == "gaze_aversion":
        order = ["gaze_aversion", "eye_contact"]

    total_blocks = reps * len(order)
    total_s = total_blocks * block_len

    print("=" * 60)
    print("Eye-contact / gaze-aversion block task")
    print(f"  block length  : {block_len:.0f}s")
    print(f"  reps/condition: {reps}  ({total_blocks} blocks total, ~{total_s:.0f}s)")
    print(f"  starting with : {order[0]}")
    print(f"  audio cues    : {'on' if audio_enabled and _HAVE_WINSOUND else 'off'}")
    print("=" * 60)
    print("Make sure record_both.py is already running for at least "
          f"{total_s:.0f}s.\n")

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
            countdown_wait(block_len, cond_key)

    print("\nAll blocks complete. Stop record_both.py now if it's still running.")


def main():
    p = argparse.ArgumentParser(
        description="Block-design cue script for a mutual eye-contact vs. "
                     "gaze-aversion task."
    )
    p.add_argument("--block-len", type=float, default=30.0,
                   help="seconds per block (default 30)")
    p.add_argument("--reps", type=int, default=5,
                   help="repetitions of each condition (default 5)")
    p.add_argument("--start", choices=["eye_contact", "gaze_aversion"],
                   default="eye_contact",
                   help="which condition to start with (default eye_contact)")
    p.add_argument("--no-audio", action="store_true",
                   help="disable beep cues (console text only)")
    args = p.parse_args()
    run(args.block_len, args.reps, args.start, audio_enabled=not args.no_audio)


if __name__ == "__main__":
    main()
