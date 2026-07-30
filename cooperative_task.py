"""
cooperative_task.py — block-design cue script for an ALONE vs COOPERATIVE
joint multiple-choice quiz (trivia / logic / word-analogy mix), with an
on-screen participant display, using the same OpenCV window pattern as
play_stimulus.py.

Alternates ALONE / COOPERATIVE blocks, firing a distinct LSL marker at every
condition switch on the same "StimulusMarkers" stream that record_both.py
already watches for (same handshake as play_stimulus.py / stimulus_marker.py
/ the old eye_contact_task.py).

Task design (Bahrami et al. 2010, Science — "two heads better than one"
collective perceptual/cognitive decision-making paradigm, the standard
design behind the joint-selection literature):
  Each round shows one multiple-choice question (2-4 answers), randomly
  drawn from a mixed trivia/logic/analogy bank.
    ALONE block: both look and silently decide, on their own, which answer
      is correct -- no talking, no comparing. Individual-decision baseline.
    COOPERATIVE block: both look at a (new) question, discuss out loud,
      then each participant CLICKS their own answer in turn (Participant A
      clicks first, then Participant B) using the same on-screen buttons.
  Multiple-choice answers are much faster for a dyad to converge on than an
  open-ended answer, and give a clean correct/incorrect score per trial --
  so besides the EEG synchrony contrast, you also get joint accuracy as a
  bonus behavioral measure (the original "collective benefit" finding is
  whether joint accuracy beats the better individual's accuracy). Both
  conditions do the SAME kind of question, controlling for shared
  engagement/attention.

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
     for Participant A to click their answer, then Participant B to click
     theirs.
  4. record_both.py captures every marker into its _markers.json sidecars.
  5. Analyze with: python compare_conditions.py <A.csv> <B.csv>

Usage:
  python cooperative_task.py                       # 4 reps, 4 questions/block each condition
  python cooperative_task.py --reps 6               # more blocks -> better-powered contrast test
  python cooperative_task.py --alone-round-time 15   # slower alone rounds (more time to read)
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


# ============================================================
# QUESTION BANK (trivia / logic / word-analogy mix)
# ============================================================

QUESTION_BANK = [
    # -- trivia --------------------------------------------------------
    {"category": "trivia", "question": "What is the capital of Australia?",
     "options": ["Sydney", "Canberra", "Melbourne", "Perth"], "correct_index": 1},
    {"category": "trivia", "question": "Which planet is known as the Red Planet?",
     "options": ["Venus", "Mars", "Jupiter", "Saturn"], "correct_index": 1},
    {"category": "trivia", "question": "What is the largest ocean on Earth?",
     "options": ["Atlantic", "Indian", "Arctic", "Pacific"], "correct_index": 3},
    {"category": "trivia", "question": "Who wrote 'Romeo and Juliet'?",
     "options": ["Dickens", "Shakespeare", "Hemingway", "Austen"], "correct_index": 1},
    {"category": "trivia", "question": "What gas do plants absorb from the air to grow?",
     "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "correct_index": 2},
    {"category": "trivia", "question": "How many continents are there on Earth?",
     "options": ["5", "6", "7", "8"], "correct_index": 2},
    {"category": "trivia", "question": "What is the chemical symbol for gold?",
     "options": ["Ag", "Au", "Gd", "Go"], "correct_index": 1},
    {"category": "trivia", "question": "Which country is home to the kangaroo?",
     "options": ["New Zealand", "South Africa", "Australia", "Brazil"], "correct_index": 2},
    {"category": "trivia", "question": "What is the smallest prime number?",
     "options": ["0", "1", "2", "3"], "correct_index": 2},
    {"category": "trivia", "question": "Which organ pumps blood through the body?",
     "options": ["Lungs", "Liver", "Heart", "Kidney"], "correct_index": 2},
    {"category": "trivia", "question": "What year did World War II end?",
     "options": ["1943", "1945", "1947", "1950"], "correct_index": 1},
    {"category": "trivia", "question": "What is the tallest mountain in the world?",
     "options": ["K2", "Kilimanjaro", "Everest", "Denali"], "correct_index": 2},

    # -- logic -----------------------------------------------------------
    {"category": "logic", "question": "All Bloops are Razzles. All Razzles are Lazzles. "
     "Are all Bloops definitely Lazzles?",
     "options": ["Yes", "No", "Not enough info"], "correct_index": 0},
    {"category": "logic", "question": "It is true that 'No cats are dogs.' "
     "Is it also true that 'No dogs are cats'?",
     "options": ["Yes", "No", "Cannot say"], "correct_index": 0},
    {"category": "logic", "question": "Some fruits are sweet. All apples are fruits. "
     "Are all apples definitely sweet?",
     "options": ["Yes", "No", "Not enough info"], "correct_index": 2},
    {"category": "logic", "question": "A is taller than B. B is taller than C. "
     "Is A definitely taller than C?",
     "options": ["Yes", "No", "Not enough info"], "correct_index": 0},
    {"category": "logic", "question": "If today is Wednesday, what day was it 3 days ago?",
     "options": ["Sunday", "Monday", "Tuesday", "Saturday"], "correct_index": 0},
    {"category": "logic", "question": "Tom is older than Jane. Jane is older than Sam. "
     "Who is the youngest?",
     "options": ["Tom", "Jane", "Sam", "Cannot say"], "correct_index": 2},
    {"category": "logic", "question": "No birds are mammals. A robin is a bird. "
     "Is a robin a mammal?",
     "options": ["Yes", "No", "Cannot say"], "correct_index": 1},
    {"category": "logic", "question": "Every square is a rectangle. "
     "Is every rectangle a square?",
     "options": ["Yes", "No", "Sometimes"], "correct_index": 1},

    # -- word analogies --------------------------------------------------
    {"category": "analogy", "question": "HAND is to GLOVE as FOOT is to ___?",
     "options": ["Sock", "Shoe", "Leg", "Toe"], "correct_index": 1},
    {"category": "analogy", "question": "BIRD is to NEST as BEE is to ___?",
     "options": ["Flower", "Hive", "Wing", "Honey"], "correct_index": 1},
    {"category": "analogy", "question": "DOCTOR is to HOSPITAL as TEACHER is to ___?",
     "options": ["Book", "School", "Student", "Chalk"], "correct_index": 1},
    {"category": "analogy", "question": "PUPPY is to DOG as KITTEN is to ___?",
     "options": ["Cub", "Cat", "Kitty", "Feline"], "correct_index": 1},
    {"category": "analogy", "question": "PEN is to WRITE as KNIFE is to ___?",
     "options": ["Sharp", "Cut", "Kitchen", "Blade"], "correct_index": 1},
    {"category": "analogy", "question": "SUN is to DAY as MOON is to ___?",
     "options": ["Star", "Night", "Sky", "Light"], "correct_index": 1},
    {"category": "analogy", "question": "AUTHOR is to BOOK as DIRECTOR is to ___?",
     "options": ["Camera", "Movie", "Actor", "Script"], "correct_index": 1},
    {"category": "analogy", "question": "FISH is to WATER as BIRD is to ___?",
     "options": ["Nest", "Air", "Tree", "Wing"], "correct_index": 1},
]


class QuestionDeck:
    """Draws questions without repeats until the bank is exhausted, then
    reshuffles -- so a session doesn't repeat a question until every other
    one has been used."""

    def __init__(self, rng, bank=None):
        self.rng = rng
        self.bank = list(bank or QUESTION_BANK)
        self._remaining = []

    def draw(self):
        if not self._remaining:
            self._remaining = list(self.bank)
            self.rng.shuffle(self._remaining)
        return self._remaining.pop()


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
# PARTICIPANT-FACING WINDOW (OpenCV, same style as play_stimulus.py)
# ============================================================

_mouse_state = {"click": None, "pos": (-1, -1)}


def _mouse_callback(event, x, y, flags, param):
    _mouse_state["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_state["click"] = (x, y)


def get_screen_size():
    """(width, height) of the primary screen, or None if unavailable (e.g.
    non-Windows without a display)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return None


def make_window(fullscreen, window_x, window_y, width=1100, height=700):
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN_NAME, width, height)
        if window_x is None and window_y is None:
            # default: center on the primary screen
            screen_size = get_screen_size()
            if screen_size is not None:
                screen_w, screen_h = screen_size
                window_x = max(0, (screen_w - width) // 2)
                window_y = max(0, (screen_h - height) // 2)
        cv2.moveWindow(WIN_NAME, window_x or 0, window_y or 0)
    cv2.setMouseCallback(WIN_NAME, _mouse_callback)
    return (height, width)


def point_in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def draw_button(frame, rect, label, hover=False, font_scale=0.75):
    x0, y0, x1, y1 = rect
    fill = (90, 60, 60) if not hover else (60, 140, 220)
    border = (140, 140, 140) if not hover else (200, 220, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 3)
    centered_text(frame, label, (y0 + y1) // 2 + 8, font_scale, (255, 255, 255), 2,
                  x_center=(x0 + x1) // 2)


def wait_for_button_click(shape, base_frame, rects_by_key):
    """Show base_frame with hover-responsive buttons overlaid (rects_by_key:
    {key: (rect, label)}); block until one is clicked (clicks elsewhere are
    ignored); return the clicked key."""
    _mouse_state["click"] = None
    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("\n  Window closed -- aborting.")
            raise SystemExit(0)
        mx, my = _mouse_state["pos"]
        frame = base_frame.copy()
        for key, (rect, label) in rects_by_key.items():
            draw_button(frame, rect, label, hover=point_in_rect(mx, my, rect))
        show(frame)
        cv2.waitKey(15)
        if _mouse_state["click"] is not None:
            cx, cy = _mouse_state["click"]
            _mouse_state["click"] = None
            for key, (rect, _label) in rects_by_key.items():
                if point_in_rect(cx, cy, rect):
                    return key
            # clicked outside every button -- ignore and keep waiting


def blank_frame(shape):
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def centered_text(frame, text, y, scale, color, thick, x_center=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]
    cx = w // 2 if x_center is None else x_center
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (cx - tw // 2, y), font, scale, color, thick, cv2.LINE_AA)


def wrap_text(text, max_chars=56):
    words = text.split()
    lines, cur = [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if len(trial) > max_chars and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def option_rects(shape, n_options, top):
    """Answer-option button rects: a single centered row for 2-3 options,
    a 2x2 grid for 4."""
    w = shape[1]
    btn_h, gap_y = 75, 20
    if n_options <= 3:
        btn_w, gap_x = 320, 30
        total_w = n_options * btn_w + (n_options - 1) * gap_x
        x0 = (w - total_w) // 2
        return [(x0 + i * (btn_w + gap_x), top, x0 + i * (btn_w + gap_x) + btn_w, top + btn_h)
                for i in range(n_options)]
    btn_w, gap_x = 440, 30
    total_w = 2 * btn_w + gap_x
    x0 = (w - total_w) // 2
    rects = []
    for i in range(n_options):
        row, col = divmod(i, 2)
        x = x0 + col * (btn_w + gap_x)
        y = top + row * (btn_h + gap_y)
        rects.append((x, y, x + btn_w, y + btn_h))
    return rects


def render_question_frame(shape, title, subtitle, item):
    frame = blank_frame(shape)
    centered_text(frame, title, 45, 1.1, (255, 255, 255), 2)
    centered_text(frame, subtitle, 78, 0.6, (150, 150, 150), 2)
    centered_text(frame, f"[{item['category'].upper()}]", 112, 0.55, (0, 200, 200), 1)

    y = 160
    for line in wrap_text(item["question"]):
        centered_text(frame, line, y, 0.85, (255, 255, 255), 2)
        y += 42

    letters = "ABCD"
    rects = option_rects(shape, len(item["options"]), top=y + 30)
    labels = [f"{letters[i]}. {opt}" for i, opt in enumerate(item["options"])]
    return frame, rects, labels


def render_feedback_frame(shape, item, participant_choices, joint_correct):
    frame = blank_frame(shape)
    h = shape[0]
    correct_text = item["options"][item["correct_index"]]
    centered_text(frame, f"Correct answer: {correct_text}", h // 2 - 60, 1.0,
                  (120, 220, 120), 2)
    a_choice = item["options"][participant_choices["participant_a"]]
    b_choice = item["options"][participant_choices["participant_b"]]
    centered_text(frame, f"A: {a_choice}   B: {b_choice}", h // 2 + 10, 0.9,
                  (220, 220, 220), 2)
    color = (120, 220, 120) if joint_correct else (100, 100, 220)
    mark = "jointly correct" if joint_correct else "jointly incorrect"
    centered_text(frame, mark, h // 2 + 60, 0.95, color, 2)
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

def run_alone_round(shape, deck, round_time):
    item = deck.draw()
    frame, rects, labels = render_question_frame(
        shape, "ALONE", "Silently decide your OWN answer. No talking.", item)
    for rect, label in zip(rects, labels):
        draw_button(frame, rect, label, hover=False)
    live_countdown(shape, frame, round_time, "alone round")
    show(blank_frame(shape))
    return {"phase": "alone", "category": item["category"], "question": item["question"],
            "correct_answer": item["options"][item["correct_index"]]}


def run_cooperative_round(shape, deck, outlet):
    item = deck.draw()
    letters = "ABCD"

    def base_frame(subtitle):
        frame, rects, labels = render_question_frame(shape, "COOPERATIVE", subtitle, item)
        rects_by_key = {i: (rect, label) for i, (rect, label) in enumerate(zip(rects, labels))}
        return frame, rects_by_key

    print(f"\n  [{item['category'].upper()}] {item['question']}")
    for letter, opt in zip(letters, item["options"]):
        print(f"    {letter}. {opt}")
    print("  Participants: discuss out loud, then each click your own answer.")

    frame_a, rects_a = base_frame("Participant A: click your answer")
    print("  Waiting for Participant A to click ...")
    idx_a = wait_for_button_click(shape, frame_a, rects_a)
    print(f"  Participant A chose: {item['options'][idx_a]}")

    frame_b, rects_b = base_frame("Participant B: click your answer")
    print("  Waiting for Participant B to click ...")
    idx_b = wait_for_button_click(shape, frame_b, rects_b)
    print(f"  Participant B chose: {item['options'][idx_b]}")

    choices = {"participant_a": idx_a, "participant_b": idx_b}
    trial_time = local_clock()
    outlet.push_sample(["COOPERATIVE_trial_start"], trial_time)
    outlet.push_sample([f"COOPERATIVE_choice_{idx_a}{idx_b}"], local_clock())

    agreement = idx_a == idx_b
    joint_correct = idx_a == item["correct_index"] and idx_b == item["correct_index"]

    show(render_feedback_frame(shape, item, choices, joint_correct))
    time.sleep(2.5)
    pump()
    show(blank_frame(shape))
    return {"phase": "cooperative", "category": item["category"], "question": item["question"],
            "options": item["options"], "correct_answer": item["options"][item["correct_index"]],
            "participant_a_answer": item["options"][idx_a],
            "participant_b_answer": item["options"][idx_b],
            "agreement": agreement, "joint_correct": joint_correct, "correct": joint_correct}


def run(alone_trials, coop_trials, alone_round_time, reps, start_condition,
        audio_enabled, fullscreen, window_x, window_y, log_dir):
    order = ["alone", "cooperative"]
    if start_condition == "cooperative":
        order = ["cooperative", "alone"]

    rng = random.Random()
    deck = QuestionDeck(rng)
    total_blocks = reps * len(order)

    print("=" * 60)
    print("Cooperative-task (ALONE vs COOPERATIVE) — mixed trivia/logic/"
          "analogy quiz")
    print(f"  alone: {alone_trials} questions/block, {alone_round_time:.0f}s each")
    print(f"  cooperative: {coop_trials} questions/block, open discussion each")
    print(f"  reps/condition: {reps}  ({total_blocks} blocks total)")
    print(f"  starting with : {order[0]}")
    print(f"  audio cues    : {'on' if audio_enabled and _HAVE_WINSOUND else 'off'}")
    print(f"  question bank : {len(QUESTION_BANK)} questions")
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

    print("How to play, in one line: read the question, then either "
          "silently decide your OWN answer (ALONE), or talk it out and "
          "each click your answer (COOPERATIVE).\n")
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
                    rounds.append(run_alone_round(shape, deck, alone_round_time))
            else:
                n_correct = 0
                for _ in range(coop_trials):
                    r = run_cooperative_round(shape, deck, outlet)
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
        log_dir, f"cooperative_quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                     "joint multiple-choice quiz (trivia/logic/analogy "
                     "mix), with an on-screen participant display.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alone-trials", type=int, default=4,
                   help="questions per ALONE block (default 4)")
    p.add_argument("--coop-trials", type=int, default=4,
                   help="questions per COOPERATIVE block (default 4)")
    p.add_argument("--alone-round-time", type=float, default=12.0,
                   help="seconds per ALONE question -- auto-advances, no "
                        "input needed (default 12)")
    p.add_argument("--reps", type=int, default=4,
                   help="repetitions of each condition block (default 4). "
                        "compare_conditions.py's block-permutation contrast "
                        "test needs several reps per condition to have any "
                        "resolution -- 1-2 reps is not enough.")
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
        args.start, not args.no_audio, args.fullscreen, args.window_x,
        args.window_y, args.log_dir)


if __name__ == "__main__":
    main()
