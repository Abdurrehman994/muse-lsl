"""
video_task.py — block-design cue script for an ALONE vs VIDEO short-clip
task, firing the same LSL block markers as cooperative_task.py so
compare_conditions.py can analyze either.

Alternates ALONE (silent baseline) / VIDEO (watch a short clip) blocks,
firing a distinct LSL marker at every condition switch on the same
"StimulusMarkers" stream that record_both.py already watches for (same
handshake as play_stimulus.py / stimulus_marker.py / cooperative_task.py).

Rationale (see conversation notes / internship writeup): Evan asked to
also test short video clips ("like the one you showed me" -- stimulus.mp4)
as a stimulus task. Note this is a different test than the earlier
long-form passive-video pilot sessions (which showed near-chance PLV/circ-
corr): short, single, well-defined clips are closer to the "neurocinematics"
inter-subject-correlation literature (Hasson et al. 2008), where clip
choice and length matter a lot for how much shared engagement/entrainment
they produce -- so this isn't assumed to reproduce the earlier null result.

VIDEO blocks play a random segment (with its original audio/voice) pulled
live from whatever full-length source videos sit in --sources-dir (default
stimuli/sources/) -- a fresh random start time is drawn every time a segment
is played, so re-running this script (or looping through more reps) shows
different footage instead of the same clip every time. Playback uses
ffpyplayer (MediaPlayer) instead of cv2.VideoCapture because OpenCV cannot
decode/output audio at all -- cv2 is still used only to display frames and
manage the window.

(make_video_clips.py's pre-cut silent clips are a separate, older workflow
for stimulus.mp4 and are unaffected by this.)

IBS workflow
────────────────────────────────────────────────────────────────
  Terminal 1:  python record_both.py 900
  Terminal 2:  python video_task.py --reps 3

  1. Position the OpenCV window on the participant-facing monitor
     (--window-x/--window-y, or --fullscreen).
  2. Press ENTER in the terminal to begin.
  3. Blocks alternate automatically; a marker fires at the start of each.
     ALONE blocks auto-advance (silent countdown). VIDEO blocks play one
     randomly-chosen segment (with audio) fully, then auto-advance.
  4. record_both.py captures every marker into its _markers.json sidecars.
  5. Analyze with: python compare_conditions.py <A.csv> <B.csv>

Usage:
  python video_task.py                          # 3 reps (one per source video in stimuli/sources/), 25s alone blocks
  python video_task.py --sources-dir stimuli/sources --reps 3
  python video_task.py --clip-length 60         # shorter video segments (default 90s)
  python video_task.py --alone-len 45
  python video_task.py --window-x 1920 --fullscreen
  python video_task.py --start video           # start with a video instead of alone
────────────────────────────────────────────────────────────────
"""

import argparse
import glob
import os
import random
import time
from datetime import datetime

import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
from pylsl import StreamInfo, StreamOutlet, local_clock

from cooperative_task import get_screen_size
from make_video_clips import get_duration_s

try:
    import winsound
    _HAVE_WINSOUND = True
except ImportError:
    _HAVE_WINSOUND = False

MARKER_STREAM_NAME = "StimulusMarkers"
WIN_NAME = "Video Task"

CONDITIONS = {
    "alone": {"marker": "ALONE_start", "beeps": 1},
    "video": {"marker": "VIDEO_start", "beeps": 2},
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


class SegmentDeck:
    """Draws (source_path, start_s) segments without repeating a SOURCE
    VIDEO until the pool is exhausted, then reshuffles (same pattern as
    cooperative_task.py's QuestionDeck) -- but unlike a fixed clip file,
    start_s is re-rolled fresh on every draw, so replays of the same source
    later in the session (or on the next run of this script) land on a
    different segment instead of repeating the exact same footage.

    Also guards the reshuffle boundary: with --reps greater than the number
    of source videos, a naive reshuffle can land on the same source twice
    in a row (the last draw of one cycle and the first of the next) --
    draw() swaps that case away so the same clip never plays back-to-back."""

    def __init__(self, rng, source_paths, clip_length):
        if not source_paths:
            raise SystemExit(
                "ERROR: no source videos found. Add some .mp4 files to "
                "stimuli/sources/ (e.g. the videos you just downloaded)."
            )
        self.rng = rng
        self.clip_length = clip_length
        self.pool = list(source_paths)
        self._remaining = []
        self._last_source = None
        self._durations = {p: get_duration_s(p) for p in source_paths}

    def draw(self):
        if not self._remaining:
            self._remaining = list(self.pool)
            self.rng.shuffle(self._remaining)
            # avoid a back-to-back repeat across the reshuffle boundary
            if (len(self._remaining) > 1 and self._last_source is not None
                    and self._remaining[-1] == self._last_source):
                swap_idx = self.rng.randrange(len(self._remaining) - 1)
                self._remaining[-1], self._remaining[swap_idx] = (
                    self._remaining[swap_idx], self._remaining[-1])
        source_path = self._remaining.pop()
        self._last_source = source_path
        duration_s = self._durations[source_path]
        max_start = max(0.0, duration_s - self.clip_length)
        start_s = self.rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        return source_path, start_s


def make_window(fullscreen, window_x, window_y, width=960, height=600):
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


def text_frame(shape, lines, font_scale=None):
    h, w = shape
    frame = blank_frame(shape)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = font_scale or min(h, w) / 500
    thick = max(1, int(scale * 2))
    line_h = int(scale * 50)
    total_h = line_h * len(lines)
    y0 = (h - total_h) // 2 + line_h
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, font, scale, thick)
        x = (w - tw) // 2
        cv2.putText(frame, line, (x, y0 + i * line_h), font, scale,
                    (220, 220, 220), thick, cv2.LINE_AA)
    return frame


def show(frame):
    cv2.imshow(WIN_NAME, frame)
    cv2.waitKey(1)


def pump():
    cv2.waitKey(1)


def run_alone_block(shape, alone_len):
    deadline = time.perf_counter() + alone_len
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        print(f"\r  alone: {remaining:4.1f}s remaining ...", end="", flush=True)
        show(text_frame(shape, ["ALONE", "sit quietly - no talking, no eye contact",
                                 f"{remaining:.0f}s remaining"]))
        time.sleep(min(0.2, max(0.0, remaining)))
    print("\r  alone: done.                              ")
    show(blank_frame(shape))


def run_video_block(shape, source_path, start_s, length_s, audio_enabled):
    """Plays [start_s, start_s+length_s) of source_path, audio included.

    Uses ffpyplayer's MediaPlayer (not cv2.VideoCapture) because OpenCV has
    no audio support at all -- it can only decode/display video frames.
    MediaPlayer decodes both and plays audio itself (via SDL) in real time;
    cv2 is used here only to draw the returned video frames into our window.
    """
    print(f"  Playing {os.path.basename(source_path)} "
          f"@ {start_s:.1f}s ({length_s:.0f}s, audio {'on' if audio_enabled else 'off'})")
    player = MediaPlayer(
        source_path,
        ff_opts={"ss": start_s, "t": length_s, "an": not audio_enabled},
    )
    fh, fw = shape
    try:
        while True:
            frame, val = player.get_frame()
            if val == "eof":
                break
            if frame is None:
                pump()
                time.sleep(0.01)
                continue
            img, pts = frame
            w, h = img.get_size()
            buf = bytes(img.to_memoryview()[0])
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            arr = cv2.resize(arr, (fw, fh))
            show(arr)
            if val:
                time.sleep(val)
    finally:
        player.close_player()
    show(blank_frame(shape))


def run(sources_dir, clip_length, alone_len, reps, start_condition, audio_enabled,
        fullscreen, window_x, window_y):
    order = ["alone", "video"]
    if start_condition == "video":
        order = ["video", "alone"]

    source_paths = sorted(glob.glob(os.path.join(sources_dir, "*.mp4")))
    rng = random.Random()
    deck = SegmentDeck(rng, source_paths, clip_length)
    total_blocks = reps * len(order)

    print("=" * 60)
    print("Video-task (ALONE vs VIDEO) block script")
    print(f"  alone block length: {alone_len:.0f}s")
    print(f"  video segment length: {clip_length:.0f}s")
    print(f"  source videos found: {len(source_paths)} in {sources_dir} "
          f"({', '.join(os.path.basename(p) for p in source_paths)})")
    print(f"  reps/condition: {reps}  ({total_blocks} blocks total)")
    print(f"  starting with : {order[0]}")
    print(f"  audio (beeps + video voice): {'on' if audio_enabled else 'off'}")
    if reps < 3:
        print("  NOTE: --reps below 3 gives compare_conditions.py's block-"
              "permutation contrast test very little resolution -- "
              "consider --reps 3+.")
    print("=" * 60)
    print("Make sure record_both.py is already running for the full "
          "session length.\n")

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

    input("Press ENTER when both participants are ready to begin ...")

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

            if cond_key == "alone":
                run_alone_block(shape, alone_len)
            else:
                source_path, start_s = deck.draw()
                run_video_block(shape, source_path, start_s, clip_length, audio_enabled)

    show(blank_frame(shape))
    print("\nAll blocks complete. Stop record_both.py now if it's still running.")
    print("Analyze with:  python compare_conditions.py <A.csv> <B.csv>")

    cv2.waitKey(500)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description="Block-design cue script for an ALONE vs VIDEO "
                     "short-clip task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sources-dir", default="stimuli/sources",
                   help="directory of full-length source .mp4 videos to pull "
                        "random VIDEO-block segments from, audio included "
                        "(default stimuli/sources)")
    p.add_argument("--clip-length", type=float, default=90.0,
                   help="seconds to play per VIDEO block, drawn from a fresh "
                        "random start point in the source video each time "
                        "(default 90)")
    p.add_argument("--alone-len", type=float, default=25.0,
                   help="seconds for each ALONE block (default 25)")
    p.add_argument("--reps", type=int, default=3,
                   help="repetitions of each condition block (default 3, "
                        "matching a 3-video pool). "
                        "compare_conditions.py's block-permutation contrast "
                        "test needs several reps per condition to have any "
                        "resolution -- 1-2 reps is not enough.")
    p.add_argument("--start", choices=["alone", "video"], default="alone",
                   help="which condition to start with (default alone)")
    p.add_argument("--no-audio", action="store_true",
                   help="disable beep cues AND video audio/voice (silent, "
                        "console text only)")
    p.add_argument("--fullscreen", action="store_true",
                   help="participant-facing window fullscreen")
    p.add_argument("--window-x", type=int, default=None,
                   help="x position (px) to place the participant window")
    p.add_argument("--window-y", type=int, default=None,
                   help="y position (px) to place the participant window")
    args = p.parse_args()

    run(args.sources_dir, args.clip_length, args.alone_len, args.reps, args.start,
        not args.no_audio, args.fullscreen, args.window_x, args.window_y)


if __name__ == "__main__":
    main()
