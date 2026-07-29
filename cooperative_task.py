"""
cooperative_task.py — block-design cue script for an ALONE vs COOPERATIVE
joint two-alternative forced-choice (2AFC) task ("which side has more
dots?"), with an on-screen participant display, using the same OpenCV
window pattern as play_stimulus.py.

Alternates ALONE / COOPERATIVE blocks, firing a distinct LSL marker at every
condition switch on the same "StimulusMarkers" stream that record_both.py
already watches for (same handshake as play_stimulus.py / stimulus_marker.py
/ the old eye_contact_task.py).

Task design (Bahrami et al. 2010, Science — "two heads better than one"
collective perceptual decision-making paradigm, the standard design behind
the joint-selection literature):
  Each round shows two dot-scatter panels, LEFT and RIGHT, one with
  slightly more dots than the other. Neither subject is told the answer.
    ALONE block: both look and silently decide, on their own, which side
      (L or R) has more -- no talking, no comparing. Individual-decision
      baseline.
    COOPERATIVE block: both look at a (new) pair of panels and choose
      LEFT or RIGHT at the same time, using a shared response window.
      The script records both participants' selections in a single trial,
      which is ideal for simultaneous-choice interbrain synchrony work.
  Binary agreement (pick a side) is much faster and easier for a dyad to
  converge on than an exact numeric consensus, and gives a clean
  correct/incorrect score per trial -- so besides the EEG synchrony
  contrast, you also get joint accuracy as a bonus behavioral measure
  (the original "collective benefit" finding is whether joint accuracy
  beats the better individual's accuracy). Both conditions do the SAME
  perceptual task, controlling for shared visual attention/engagement.

Rationale for cooperation generally (see conversation notes / internship
writeup): passive shared-video viewing showed near-chance inter-brain
PLV/circ-corr even after fixing the measurement pipeline (epoch-length
bias, then continuous artifact rejection) -- expected, per the
hyperscanning literature, since two people independently watching one
video is "neural entrainment" (a confound), not genuine inter-brain
synchrony. Cooperation (shared goal, mutual dependency, verbal
coordination) has the strongest and most replicated effect size in the
field (e.g. Czeszumski et al. 2022 meta-analysis, g=1.98 for cooperative
vs. non-cooperative).

IBS workflow
────────────────────────────────────────────────────────────────
  Terminal 1:  python record_both.py 600
  Terminal 2:  python cooperative_task.py --reps 4

  1. Position the OpenCV window on the participant-facing monitor
     (--window-x/--window-y, or --fullscreen).
  2. Press ENTER in the terminal to begin.
  3. Blocks alternate automatically; a marker fires at the start of each.
     ALONE rounds auto-advance (no input needed). COOPERATIVE rounds wait
     for you to type the pair's final agreed side (L/R).
  4. record_both.py captures every marker into its _markers.json sidecars.
  5. Analyze with: python compare_conditions.py <A.csv> <B.csv>

Usage:
  python cooperative_task.py                       # 4 reps, 4 rounds/block each condition
  python cooperative_task.py --reps 6               # more blocks -> better-powered contrast test
  python cooperative_task.py --alone-round-time 12   # faster/slower alone rounds
  python cooperative_task.py --delta-min 6 --delta-max 12  # make the choice easier (bigger gap)
  python cooperative_task.py --window-x 1920         # place the participant window on a second monitor
  python cooperative_task.py --fullscreen            # participant window fullscreen
  python cooperative_task.py --start cooperative     # start with cooperative instead of alone
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import cv2
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

try:
    import winsound
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False

MARKER_STREAM_NAME = "StimulusMarkers"
WIN_NAME = "Cooperative Task"

CONDITIONS = {
    "alone": {"marker": "ALONE_start", "beeps": 1},
    "cooperative": {"marker": "COOPERATIVE_start", "beeps": 2},
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


# ============================================================
# DOT-COUNT STIMULUS (procedurally generated, no external content needed)
# ============================================================

def make_dot_image(size, n_dots, rng):
    """Scattered white dots on black, non-overlapping. Returns (image, actual_count)
    -- actual_count may be slightly below n_dots if the canvas got too crowded."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    dot_radius = max(6, size // 45)
    min_dist = dot_radius * 2.4
    margin = dot_radius * 2
    placed = []
    attempts = 0
    max_attempts = n_dots * 300
    while len(placed) < n_dots and attempts < max_attempts:
        x = rng.uniform(margin, size - margin)
        y = rng.uniform(margin, size - margin)
        if all((x - px) ** 2 + (y - py) ** 2 > min_dist ** 2 for px, py in placed):
            placed.append((x, y))
        attempts += 1
    for x, y in placed:
        cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas, len(placed)


def make_choice_trial(rng, base_range, delta_range):
    """Pick (n_left, n_right, correct_side) for one 2AFC trial."""
    base = rng.randint(*base_range)
    delta = rng.randint(*delta_range)
    correct_side = rng.choice(["left", "right"])
    if correct_side == "left":
        n_left, n_right = base + delta, base
    else:
        n_left, n_right = base, base + delta
    return n_left, n_right, correct_side


def parse_side(answer):
    norm = answer.strip().lower()
    if norm in ("l", "left"):
        return "left"
    if norm in ("r", "right"):
        return "right"
    return None


def parse_simultaneous_choices(raw):
    parts = raw.split()
    if len(parts) != 2:
        return None
    participant_a = parse_side(parts[0])
    participant_b = parse_side(parts[1])
    if participant_a is None or participant_b is None:
        return None
    return {"participant_a": participant_a, "participant_b": participant_b}


# ============================================================
# PARTICIPANT-FACING WINDOW (OpenCV, same style as play_stimulus.py)
# ============================================================

def make_window(fullscreen, window_x, window_y, width=1100, height=700):
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN_NAME, width, height)
        if window_x is not None or window_y is not None:
            cv2.moveWindow(WIN_NAME, window_x or 0, window_y or 0)
    return (height, width)


def blank_frame(shape):
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def centered_text(frame, text, y, scale, color, thick, x_center=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    cx = w // 2 if x_center is None else x_center
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (cx - tw // 2, y), font, scale, color, thick, cv2.LINE_AA)


def render_choice_frame(shape, title, subtitle, left_img, right_img):
    frame = blank_frame(shape)
    h, w = shape
    centered_text(frame, title, 50, 1.1, (255, 255, 255), 2)
    centered_text(frame, subtitle, 88, 0.65, (150, 150, 150), 2)

    gap = 40
    panel_w = (w - 60 - gap) // 2
    avail_h = h - 190
    left_x0, right_x0 = 30, 30 + panel_w + gap

    for img, x0, label in ((left_img, left_x0, "LEFT"), (right_img, right_x0, "RIGHT")):
        centered_text(frame, label, 130, 0.9, (0, 200, 200), 2, x_center=x0 + panel_w // 2)
        dh, dw = img.shape[:2]
        scale = min(panel_w / dw, avail_h / dh, 1.0)
        nw, nh = max(1, int(dw * scale)), max(1, int(dh * scale))
        resized = cv2.resize(img, (nw, nh))
        y0 = 150
        xoff = x0 + (panel_w - nw) // 2
        frame[y0:y0 + nh, xoff:xoff + nw] = resized

    return frame


def render_feedback_frame(shape, n_left, n_right, correct_side, agreed_side=None,
                          participant_choices=None, joint_correct=None):
    frame = blank_frame(shape)
    h, w = shape
    centered_text(frame, f"LEFT={n_left}   RIGHT={n_right}", h // 2 - 120, 0.9,
                  (150, 150, 150), 2)
    centered_text(frame, f"Correct: {correct_side.upper()}", h // 2 - 60, 1.1,
                  (120, 220, 120), 2)
    if participant_choices is not None:
        a_choice = participant_choices["participant_a"].upper()
        b_choice = participant_choices["participant_b"].upper()
        centered_text(frame, f"A: {a_choice}   B: {b_choice}", h // 2 + 10, 0.9,
                      (220, 220, 220), 2)
        if joint_correct is not None:
            color = (120, 220, 120) if joint_correct else (100, 100, 220)
            mark = "jointly correct" if joint_correct else "jointly incorrect"
            centered_text(frame, mark, h // 2 + 60, 0.95, color, 2)
    elif agreed_side is not None:
        is_correct = agreed_side == correct_side
        color = (120, 220, 120) if is_correct else (100, 100, 220)
        mark = "correct!" if is_correct else "not quite"
        centered_text(frame, f"You said: {agreed_side.upper()} -- {mark}",
                      h // 2 + 20, 0.9, color, 2)
    else:
        centered_text(frame, "(no answer recorded)", h // 2 + 20, 0.8, (120, 120, 120), 2)
    return frame


def show(frame):
    cv2.imshow(WIN_NAME, frame)
    cv2.waitKey(1)


def pump():
    cv2.waitKey(1)


def live_countdown(shape, frame_no_timer, seconds, label):
    """Show frame_no_timer, overlay a countdown number bottom-right, for `seconds`."""
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        print(f"\r  {label}: {remaining:4.1f}s remaining ...", end="", flush=True)
        frame = frame_no_timer.copy()
        h, w = shape
        cv2.putText(frame, f"{remaining:4.1f}s", (w - 160, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 180, 100), 2, cv2.LINE_AA)
        show(frame)
        time.sleep(min(0.2, max(0.0, remaining)))
    print(f"\r  {label}: done.                              ")


# ============================================================
# ROUNDS
# ============================================================

def run_alone_round(shape, rng, base_range, delta_range, round_time):
    n_left, n_right, correct_side = make_choice_trial(rng, base_range, delta_range)
    left_img, _ = make_dot_image(420, n_left, rng)
    right_img, _ = make_dot_image(420, n_right, rng)
    frame = render_choice_frame(
        shape, "ALONE",
        "Silently decide: does LEFT or RIGHT have more dots? No talking.",
        left_img, right_img)
    live_countdown(shape, frame, round_time, "alone round")
    show(blank_frame(shape))
    return {"phase": "alone", "n_left": n_left, "n_right": n_right,
            "correct_side": correct_side}


def run_cooperative_round(shape, rng, base_range, delta_range, outlet):
    n_left, n_right, correct_side = make_choice_trial(rng, base_range, delta_range)
    left_img, _ = make_dot_image(420, n_left, rng)
    right_img, _ = make_dot_image(420, n_right, rng)
    frame = render_choice_frame(
        shape, "COOPERATIVE",
        "Both participants choose LEFT or RIGHT at the same time.",
        left_img, right_img)
    show(frame)
    print("\n  Participants: look at the screen and choose LEFT or RIGHT "
          "simultaneously.")
    raw = input("  Enter both choices as 'L R' (or 'left right'): ").strip()
    choices = parse_simultaneous_choices(raw)
    if raw and choices is None:
        print(f"  (didn't recognize '{raw}' as a pair of L/R responses -- logging as no answer)")

    trial_time = local_clock()
    outlet.push_sample(["COOPERATIVE_trial_start"], trial_time)
    if choices is not None:
        choice_marker = f"COOPERATIVE_choice_{choices['participant_a'][:1]}{choices['participant_b'][:1]}"
        outlet.push_sample([choice_marker], local_clock())

    participant_a = choices["participant_a"] if choices else None
    participant_b = choices["participant_b"] if choices else None
    agreement = (participant_a is not None and participant_b is not None and participant_a == participant_b)
    joint_correct = (
        participant_a is not None
        and participant_b is not None
        and participant_a == correct_side
        and participant_b == correct_side
    )

    show(render_feedback_frame(
        shape, n_left, n_right, correct_side,
        participant_choices=choices, joint_correct=joint_correct))
    time.sleep(2.5)
    pump()
    show(blank_frame(shape))
    return {"phase": "cooperative", "n_left": n_left, "n_right": n_right,
            "correct_side": correct_side,
            "participant_choices": choices,
            "agreement": agreement,
            "joint_correct": joint_correct,
            "correct": joint_correct}


def run(alone_trials, coop_trials, alone_round_time, reps, start_condition,
        audio_enabled, base_min, base_max, delta_min, delta_max,
        fullscreen, window_x, window_y, log_dir):
    order = ["alone", "cooperative"]
    if start_condition == "cooperative":
        order = ["cooperative", "alone"]

    rng = random.Random()
    base_range = (base_min, base_max)
    delta_range = (delta_min, delta_max)
    total_blocks = reps * len(order)

    print("=" * 60)
    print("Cooperative-task (ALONE vs COOPERATIVE) — 2AFC 'which side has "
          "more dots?' game")
    print(f"  alone: {alone_trials} rounds/block, {alone_round_time:.0f}s each")
    print(f"  cooperative: {coop_trials} rounds/block, open discussion each")
    print(f"  reps/condition: {reps}  ({total_blocks} blocks total)")
    print(f"  starting with : {order[0]}")
    print(f"  audio cues    : {'on' if audio_enabled and _HAVE_WINSOUND else 'off'}")
    if reps < 3:
        print("  NOTE: --reps below 3 gives the later block-permutation "
              "contrast test very little resolution -- consider --reps 3+.")
    print("=" * 60)
    print("Make sure record_both.py is already running for the full session "
          "length.\n")

    shape = make_window(fullscreen, window_x, window_y)
    show(blank_frame(shape))

    outlet = build_outlet()
    if not outlet.have_consumers():
        print("Waiting for record_both.py to connect to the marker stream "
              "(up to 30s) ...")
        waited = 0.0
        while waited < 30.0 and not outlet.have_consumers():
            pump()
            outlet.wait_for_consumers(1.0)
            waited += 1.0
    if outlet.have_consumers():
        print("  Recorder connected.\n")
    else:
        print("  WARNING: no recorder connected after 30s — markers will "
              "NOT be captured. Proceeding anyway.\n")

    print("How to play, in one line: look at the two dot patches, then "
          "either silently decide on your OWN which side has more "
          "(ALONE), or talk it out and agree on ONE side together "
          "(COOPERATIVE).\n")
    input("Press ENTER when both participants are ready to begin ...")

    session_log = {"started": datetime.now().isoformat(), "blocks": []}

    block_num = 0
    for rep in range(reps):
        for cond_key in order:
            block_num += 1
            cond = CONDITIONS[cond_key]
            t_marker = local_clock()
            outlet.push_sample([cond["marker"]], t_marker)
            print(f"\n[{block_num}/{total_blocks}] >> {cond_key.upper()}  "
                  f"(marker '{cond['marker']}' @ LSL t={t_marker:.3f})")
            beep(cond["beeps"], audio_enabled)

            rounds = []
            if cond_key == "alone":
                for _ in range(alone_trials):
                    rounds.append(run_alone_round(shape, rng, base_range,
                                                   delta_range, alone_round_time))
            else:
                n_correct = 0
                for _ in range(coop_trials):
                    r = run_cooperative_round(shape, rng, base_range, delta_range, outlet)
                    rounds.append(r)
                    if r["correct"]:
                        n_correct += 1
                print(f"  Cooperative block accuracy: {n_correct}/{coop_trials}")

            session_log["blocks"].append({
                "block_num": block_num, "condition": cond_key,
                "lsl_marker_time": t_marker, "rounds": rounds,
            })

    show(blank_frame(shape))
    print("\nAll blocks complete. Stop record_both.py now if it's still running.")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"cooperative_2afc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"Round-by-round log saved to: {log_path}")
    print("  (supplementary -- the EEG analysis in compare_conditions.py "
          "only needs the ALONE_start/COOPERATIVE_start LSL markers, "
          "already captured by record_both.py)")
    print("Analyze with:  python compare_conditions.py <A.csv> <B.csv>")

    cv2.waitKey(500)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description="Block-design cue script for an ALONE vs COOPERATIVE "
                     "joint two-alternative forced-choice task ('which side "
                     "has more dots?'), with an on-screen participant "
                     "display.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alone-trials", type=int, default=4,
                   help="trials per ALONE block (default 4)")
    p.add_argument("--coop-trials", type=int, default=4,
                   help="trials per COOPERATIVE block (default 4)")
    p.add_argument("--alone-round-time", type=float, default=12.0,
                   help="seconds per ALONE trial -- auto-advances, no input "
                        "needed (default 12)")
    p.add_argument("--reps", type=int, default=4,
                   help="repetitions of each condition block (default 4). "
                        "compare_conditions.py's block-permutation contrast "
                        "test needs several reps per condition to have any "
                        "resolution -- 1-2 reps is not enough.")
    p.add_argument("--n-dots-min", type=int, default=15, dest="base_min",
                   help="minimum base dot count per side (default 15)")
    p.add_argument("--n-dots-max", type=int, default=35, dest="base_max",
                   help="maximum base dot count per side (default 35)")
    p.add_argument("--delta-min", type=int, default=4,
                   help="minimum extra dots on the correct side -- smaller "
                        "= harder to tell apart (default 4)")
    p.add_argument("--delta-max", type=int, default=10,
                   help="maximum extra dots on the correct side (default 10)")
    p.add_argument("--start", choices=["alone", "cooperative"], default="alone",
                   help="which condition to start with (default alone)")
    p.add_argument("--no-audio", action="store_true",
                   help="disable beep cues (console text only)")
    p.add_argument("--fullscreen", action="store_true",
                   help="participant-facing window fullscreen")
    p.add_argument("--window-x", type=int, default=None,
                   help="x position (px) to place the participant window -- "
                        "use this to put it on a second monitor, e.g. "
                        "--window-x 1920 if your primary monitor is 1920px wide")
    p.add_argument("--window-y", type=int, default=None,
                   help="y position (px) to place the participant window")
    p.add_argument("--log-dir", default="cooperative_logs",
                   help="directory to save the round-by-round JSON log "
                        "(default cooperative_logs/)")
    args = p.parse_args()

    run(args.alone_trials, args.coop_trials, args.alone_round_time, args.reps,
        args.start, not args.no_audio, args.base_min, args.base_max,
        args.delta_min, args.delta_max, args.fullscreen, args.window_x,
        args.window_y, args.log_dir)


if __name__ == "__main__":
    main()
