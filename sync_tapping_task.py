"""
sync_tapping_task.py -- block-design cue script for a SOLO vs SYNC
finger-tapping task, the paradigm with among the strongest and most
replicated inter-brain synchrony (IBS) effects in the hyperscanning
literature (Konvalinka et al. 2010, "Follow you, follow me").

Why this task (see conversation notes / internship writeup):
  Passive shared-video viewing and even the ALONE-vs-COOPERATIVE quiz
  (cooperative_task.py) showed near-chance inter-brain PLV after the
  measurement pipeline was fixed. The quiz's problem for a 4-channel Muse:
  its most-coupled moments (out-loud discussion) are also its most
  artifact-laden (speech/jaw EMG, head motion), so continuous_bad_mask
  removes exactly the best data; and it is episodic (read/think/discuss/
  click), so whole-recording PLV averages a lot of dead time.

  Synchronized tapping fixes both:
    - SUSTAINED & CONTINUOUS: the coupled state runs for the whole block,
      so the PLV average isn't diluted by dead time.
    - LOW-ARTIFACT: silent, only a small finger movement -- the coupling
      survives artifact rejection instead of being masked out.
    - The natural coupling channel is auditory: two people at one keyboard
      HEAR each other's key clicks and mutually predict/adapt, exactly the
      self-paced mutual-adaptation design of Konvalinka et al. (2010), who
      used auditory tap feedback and no external metronome.

Task design (two conditions, contrasted within one continuous recording):
    SOLO block: each participant taps at their OWN comfortable steady pace,
      deliberately NOT trying to match the other. Independent-tapping
      baseline (any coupling here is incidental).
    SYNC block: both try to tap in synchrony with each other, WITHOUT a
      metronome -- each must predict and track the other's tempo. The
      coupled condition.
  Both conditions use an identical on-screen display (only the instruction
  text and the block marker differ), so the SOLO-vs-SYNC contrast isolates
  the coordination effect and any shared visual-flash activity cancels.

  Participant A taps the 'F' key, Participant B taps the 'J' key (left/
  right home-row keys, natural for two people side by side at one
  keyboard). Individual tap times are logged (and fired as TAP_A/TAP_B LSL
  markers) so you also get a BEHAVIORAL synchrony measure (mean tap
  asynchrony, lower in SYNC than SOLO) as a manipulation check -- if the
  behavioural synchrony didn't move, the neural contrast is uninterpretable.

Markers (on the same "StimulusMarkers" stream record_both.py watches):
    SOLO_start / SYNC_start   once per block  (the EEG-analysis markers,
                              recognized by compare_conditions.py)
    TAP_A / TAP_B             once per tap     (behavioural; ignored by
                              compare_conditions.py's 2-label segmentation)

IBS workflow
--------------------------------------------------------------------
  Terminal 1:  python record_both.py 600
  Terminal 2:  python sync_tapping_task.py --reps 4 --block-seconds 60

  1. Position the OpenCV window on the participant-facing monitor
     (--window-x/--window-y, or --fullscreen).
  2. Press ENTER in the terminal to begin.
  3. Blocks alternate automatically; a marker fires at the start of each.
     Participant A taps 'F', Participant B taps 'J'.
  4. record_both.py captures every marker into its _markers.json sidecars.
  5. Analyze with: python compare_conditions.py <A.csv> <B.csv>

Usage:
  python sync_tapping_task.py                        # 4 reps, 60s blocks
  python sync_tapping_task.py --reps 6               # more blocks -> better-powered contrast
  python sync_tapping_task.py --block-seconds 90     # longer blocks -> more clean data per condition
  python sync_tapping_task.py --window-x 1920        # participant window on a second monitor
  python sync_tapping_task.py --fullscreen           # participant window fullscreen
  python sync_tapping_task.py --start sync           # start with sync instead of solo
--------------------------------------------------------------------
"""

import argparse
import json
import os
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
WIN_NAME = "Synchronized Tapping Task"

CONDITIONS = {
    "solo": {"marker": "SOLO_start", "beeps": 1,
             "instruction": "Tap at your OWN steady pace -- do NOT match your partner."},
    "sync": {"marker": "SYNC_start", "beeps": 2,
             "instruction": "Tap TOGETHER, in sync with your partner. No metronome -- match each other."},
}

KEY_A = ord("f")   # Participant A
KEY_B = ord("j")   # Participant B
KEY_ESC = 27

TAP_FLASH_S = 0.18  # how long a tap indicator stays lit


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
# PARTICIPANT-FACING WINDOW (OpenCV, same style as cooperative_task.py)
# ============================================================

def get_screen_size():
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
            screen_size = get_screen_size()
            if screen_size is not None:
                screen_w, screen_h = screen_size
                window_x = max(0, (screen_w - width) // 2)
                window_y = max(0, (screen_h - height) // 2)
        cv2.moveWindow(WIN_NAME, window_x or 0, window_y or 0)
    return (height, width)


def blank_frame(shape):
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.uint8)


def centered_text(frame, text, y, scale, color, thick, x_center=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = frame.shape[1]
    cx = w // 2 if x_center is None else x_center
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (cx - tw // 2, y), font, scale, color, thick, cv2.LINE_AA)


def wrap_text(text, max_chars=60):
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


def show(frame):
    cv2.imshow(WIN_NAME, frame)
    cv2.waitKey(1)


def render_tapping_frame(shape, cond_key, remaining_s, total_s,
                          count_a, count_b, a_lit, b_lit):
    frame = blank_frame(shape)
    h, w = shape
    cond = CONDITIONS[cond_key]
    title = "SYNC -- tap together" if cond_key == "sync" else "SOLO -- your own pace"
    title_color = (120, 220, 120) if cond_key == "sync" else (200, 200, 200)
    centered_text(frame, title, 60, 1.2, title_color, 3)
    for i, line in enumerate(wrap_text(cond["instruction"])):
        centered_text(frame, line, 105 + i * 34, 0.7, (170, 170, 170), 2)

    # two tap indicators
    cy = h // 2 + 20
    r = 90
    cx_a, cx_b = w // 2 - 230, w // 2 + 230
    lit_col, dim_col = (90, 220, 90), (60, 60, 60)
    cv2.circle(frame, (cx_a, cy), r, lit_col if a_lit else dim_col, -1)
    cv2.circle(frame, (cx_a, cy), r, (200, 200, 200), 3)
    cv2.circle(frame, (cx_b, cy), r, lit_col if b_lit else dim_col, -1)
    cv2.circle(frame, (cx_b, cy), r, (200, 200, 200), 3)
    centered_text(frame, "A  [F]", cy + 8, 1.0, (255, 255, 255), 2, x_center=cx_a)
    centered_text(frame, "B  [J]", cy + 8, 1.0, (255, 255, 255), 2, x_center=cx_b)
    centered_text(frame, f"{count_a} taps", cy + r + 45, 0.7, (180, 180, 180), 2, x_center=cx_a)
    centered_text(frame, f"{count_b} taps", cy + r + 45, 0.7, (180, 180, 180), 2, x_center=cx_b)

    # countdown + progress bar
    centered_text(frame, f"{remaining_s:4.1f}s", h - 90, 1.0, (100, 180, 100), 2)
    bar_w = int((w - 200) * (1.0 - remaining_s / total_s)) if total_s > 0 else 0
    cv2.rectangle(frame, (100, h - 55), (100 + bar_w, h - 40), (80, 140, 200), -1)
    cv2.rectangle(frame, (100, h - 55), (w - 100, h - 40), (120, 120, 120), 2)
    return frame


def run_tapping_block(shape, cond_key, outlet, duration_s, session_t0):
    """Run one SOLO or SYNC block for duration_s, capturing taps. Returns a
    dict with per-tap times (LSL clock and seconds-since-session-start)."""
    cond = CONDITIONS[cond_key]
    t_marker = local_clock()
    outlet.push_sample([cond["marker"]], t_marker)

    taps_a, taps_b = [], []
    last_a = last_b = -1e9
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        remaining = duration_s - elapsed
        if remaining <= 0:
            break
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("\n  Window closed -- aborting.")
            raise SystemExit(0)

        key = cv2.waitKey(5) & 0xFF
        now = time.perf_counter()
        if key == KEY_A:
            t = local_clock()
            outlet.push_sample(["TAP_A"], t)
            taps_a.append(t - session_t0)
            last_a = now
        elif key == KEY_B:
            t = local_clock()
            outlet.push_sample(["TAP_B"], t)
            taps_b.append(t - session_t0)
            last_b = now
        elif key == KEY_ESC:
            print("\n  ESC pressed -- aborting.")
            raise SystemExit(0)

        frame = render_tapping_frame(
            shape, cond_key, remaining, duration_s, len(taps_a), len(taps_b),
            a_lit=(now - last_a) < TAP_FLASH_S, b_lit=(now - last_b) < TAP_FLASH_S)
        show(frame)

    return {"condition": cond_key, "marker": cond["marker"],
            "lsl_marker_time": t_marker,
            "taps_a_s": taps_a, "taps_b_s": taps_b,
            "n_taps_a": len(taps_a), "n_taps_b": len(taps_b)}


def mean_tap_asynchrony_ms(taps_a, taps_b):
    """For each of A's taps, distance to B's nearest tap; return the mean over
    all such pairings, in ms. A behavioural manipulation check: should be
    clearly lower in SYNC than SOLO. Returns None if either has < 2 taps."""
    if len(taps_a) < 2 or len(taps_b) < 2:
        return None
    b = np.asarray(taps_b)
    diffs = [float(np.min(np.abs(b - ta))) for ta in taps_a]
    return float(np.mean(diffs) * 1000.0)


def countdown_screen(shape, seconds, label):
    for k in range(seconds, 0, -1):
        frame = blank_frame(shape)
        centered_text(frame, label, shape[0] // 2 - 40, 1.0, (200, 200, 200), 2)
        centered_text(frame, str(k), shape[0] // 2 + 40, 2.0, (120, 200, 120), 3)
        show(frame)
        time.sleep(1.0)


def run(reps, block_seconds, rest_seconds, start_condition, audio_enabled,
        fullscreen, window_x, window_y, log_dir):
    order = ["solo", "sync"]
    if start_condition == "sync":
        order = ["sync", "solo"]
    total_blocks = reps * len(order)

    print("=" * 60)
    print("Synchronized-tapping task (SOLO vs SYNC) -- Konvalinka et al. 2010")
    print(f"  block length : {block_seconds:.0f}s   rest between: {rest_seconds:.0f}s")
    print(f"  reps/condition: {reps}  ({total_blocks} blocks total)")
    print(f"  starting with : {order[0]}")
    print(f"  keys          : Participant A = 'F',  Participant B = 'J'")
    print(f"  audio cues    : {'on' if audio_enabled and _HAVE_WINSOUND else 'off'}")
    if reps < 3:
        print("  NOTE: --reps below 3 gives compare_conditions.py's block-"
              "permutation contrast very little resolution -- consider 3+.")
    print("=" * 60)
    print("Make sure record_both.py is already running for the full session.\n")

    shape = make_window(fullscreen, window_x, window_y)
    show(blank_frame(shape))

    outlet = build_outlet()
    if not outlet.have_consumers():
        print("Waiting for record_both.py to connect to the marker stream "
              "(up to 30s) ...")
        waited = 0.0
        while waited < 30.0 and not outlet.have_consumers():
            cv2.waitKey(1)
            outlet.wait_for_consumers(1.0)
            waited += 1.0
    if outlet.have_consumers():
        print("  Recorder connected.\n")
    else:
        print("  WARNING: no recorder connected after 30s -- markers will "
              "NOT be captured. Proceeding anyway.\n")

    print("How to play: when a block starts, Participant A taps 'F' and "
          "Participant B taps 'J'. In SOLO tap your own steady pace; in SYNC "
          "tap together, matching each other by ear -- no metronome.\n")
    input("Press ENTER when both participants are ready to begin ...")

    session_t0 = local_clock()
    session_log = {"started": datetime.now().isoformat(),
                   "block_seconds": block_seconds, "blocks": []}

    block_num = 0
    for rep in range(reps):
        for cond_key in order:
            block_num += 1
            cond = CONDITIONS[cond_key]
            countdown_screen(shape, 3, f"[{block_num}/{total_blocks}] {cond_key.upper()} starting")
            beep(cond["beeps"], audio_enabled)
            print(f"\n[{block_num}/{total_blocks}] >> {cond_key.upper()}  "
                  f"(marker '{cond['marker']}')")

            block = run_tapping_block(shape, cond_key, outlet, block_seconds, session_t0)
            async_ms = mean_tap_asynchrony_ms(block["taps_a_s"], block["taps_b_s"])
            block["mean_tap_asynchrony_ms"] = async_ms
            session_log["blocks"].append({"block_num": block_num, **block})
            async_str = f"{async_ms:.0f}ms" if async_ms is not None else "n/a"
            print(f"  taps: A={block['n_taps_a']}  B={block['n_taps_b']}  "
                  f"mean tap asynchrony={async_str}")

            if block_num < total_blocks and rest_seconds > 0:
                countdown_screen(shape, int(rest_seconds), "rest")

    show(blank_frame(shape))

    # behavioural manipulation-check summary: SYNC should have LOWER asynchrony
    solo_async = [b["mean_tap_asynchrony_ms"] for b in session_log["blocks"]
                  if b["condition"] == "solo" and b["mean_tap_asynchrony_ms"] is not None]
    sync_async = [b["mean_tap_asynchrony_ms"] for b in session_log["blocks"]
                  if b["condition"] == "sync" and b["mean_tap_asynchrony_ms"] is not None]
    print("\nAll blocks complete. Stop record_both.py now if it's still running.")
    if solo_async and sync_async:
        ms_solo, ms_sync = float(np.mean(solo_async)), float(np.mean(sync_async))
        print(f"  Behavioural check -- mean tap asynchrony: "
              f"SOLO={ms_solo:.0f}ms  SYNC={ms_sync:.0f}ms")
        if ms_sync < ms_solo:
            print("  -> SYNC tighter than SOLO: the manipulation worked "
                  "(participants actually synchronized).")
        else:
            print("  -> SYNC NOT tighter than SOLO: manipulation may have "
                  "failed -- treat any neural contrast with caution.")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"sync_tapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2)
    print(f"Tap-by-tap log saved to: {log_path}")
    print("  (supplementary -- the EEG analysis in compare_conditions.py only "
          "needs the SOLO_start/SYNC_start LSL markers, already captured by "
          "record_both.py)")
    print("Analyze with:  python compare_conditions.py <A.csv> <B.csv>")

    cv2.waitKey(500)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description="Block-design cue script for a SOLO vs SYNC finger-"
                     "tapping inter-brain-synchrony task (Konvalinka et al. "
                     "2010).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reps", type=int, default=4,
                   help="repetitions of each condition block (default 4). "
                        "compare_conditions.py's block-permutation contrast "
                        "needs several reps per condition to have resolution.")
    p.add_argument("--block-seconds", type=float, default=60.0,
                   help="length of each tapping block in seconds (default 60). "
                        "Longer blocks give more clean data per condition for "
                        "the PLV estimate.")
    p.add_argument("--rest-seconds", type=float, default=10.0,
                   help="rest between blocks in seconds (default 10)")
    p.add_argument("--start", choices=["solo", "sync"], default="solo",
                   help="which condition to start with (default solo)")
    p.add_argument("--no-audio", action="store_true",
                   help="disable beep cues (console text only)")
    p.add_argument("--fullscreen", action="store_true",
                   help="participant-facing window fullscreen")
    p.add_argument("--window-x", type=int, default=None,
                   help="x position (px) to place the participant window -- "
                        "e.g. --window-x 1920 to put it on a second monitor")
    p.add_argument("--window-y", type=int, default=None,
                   help="y position (px) to place the participant window")
    p.add_argument("--log-dir", default="tapping_logs",
                   help="directory for the tap-by-tap JSON log (default tapping_logs/)")
    args = p.parse_args()

    run(args.reps, args.block_seconds, args.rest_seconds, args.start,
        not args.no_audio, args.fullscreen, args.window_x, args.window_y,
        args.log_dir)


if __name__ == "__main__":
    main()
