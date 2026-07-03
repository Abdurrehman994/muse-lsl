"""
play_stimulus.py — play a stimulus video and fire an LSL marker at the exact
moment the first frame appears on screen.

IBS workflow
────────────────────────────────────────────────────────────────
Both Muse headsets stream simultaneously to one recording:

  Terminal 1:  python record_both.py 180
  Terminal 2:  python play_stimulus.py clip.mp4

  1. When both scripts are running, press ENTER in the video window to
     start the countdown and then play the video.
  2. The LSL marker fires automatically when frame 0 hits the screen.
  3. record_both.py captures the marker into _markers.json sidecars
     for both headsets.

Analysis (after recording):
  python pipeline.py recordings/<stamp>_Muse.csv recordings/<stamp>_Muse_1.csv \\
         --surrogate 200

The pipeline reads each _markers.json, trims both recordings to their
stimulus onset (t=0), and computes PLV + circular correlation on the
stimulus-locked data.
────────────────────────────────────────────────────────────────

Requirements:
  pip install opencv-python pylsl

Optional (adds audio playback):
  pip install pygame

Usage:
  python play_stimulus.py clip.mp4
  python play_stimulus.py clip.mp4 --countdown 5
  python play_stimulus.py clip.mp4 --marker trial_1_start --fullscreen
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

try:
    from ffpyplayer.player import MediaPlayer
    _HAVE_FFPYPLAYER = True
except ImportError:
    _HAVE_FFPYPLAYER = False

MARKER_STREAM_NAME = "StimulusMarkers"


def build_outlet():
    info = StreamInfo(
        MARKER_STREAM_NAME, "Markers", 1, 0, "string", "stimulus_markers_001"
    )
    return StreamOutlet(info)


def text_frame(shape, lines, font_scale=None):
    """Solid black frame with centred lines of white text."""
    h, w = shape[:2]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
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


def play(video_path, marker_name, countdown_s, fullscreen):
    if not os.path.exists(video_path):
        print(f"ERROR: file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / fps if fps > 0 else 0

    print(f"\nVideo : {video_path}")
    print(f"        {width}x{height}  {fps:.1f} fps  {duration_s:.1f}s")

    # audio via ffpyplayer
    if not _HAVE_FFPYPLAYER:
        print("Audio : ffpyplayer not installed — video silent")
        print("        pip install ffpyplayer  to add audio")

    # LSL marker outlet
    outlet = build_outlet()
    print(f"\nLSL   : '{MARKER_STREAM_NAME}' stream live")
    print("        Make sure record_both.py is running in another terminal\n")

    win = "Stimulus"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, min(width, 1280), min(height, 720))

    shape = (height, width)

    # ── wait for experimenter ──────────────────────────────────
    cv2.imshow(win, text_frame(shape, ["Press ENTER to begin"]))
    print("Press ENTER in this window to start the countdown ...")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (13, 10):      # Enter
            break
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            print("Aborted.")
            return

    # ── countdown ─────────────────────────────────────────────
    if countdown_s > 0:
        print(f"Countdown: {countdown_s}s ...")
        for n in range(countdown_s, 0, -1):
            cv2.imshow(win, text_frame(shape, [str(n)], font_scale=min(height, width) / 150))
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline:
                cv2.waitKey(40)

    # ── marker + playback (as atomic as Python allows) ────────
    t_marker = local_clock()
    outlet.push_sample([marker_name], t_marker)

    # start audio player at the same moment as the marker
    audio_player = MediaPlayer(video_path) if _HAVE_FFPYPLAYER else None

    print(f"\n  >> Marker '{marker_name}' sent  (LSL t = {t_marker:.4f})")
    print(f"     Playing {duration_s:.1f}s of video — press Q to stop early\n")

    t_start = time.perf_counter()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(win, frame)
        frame_idx += 1

        # let ffpyplayer push audio for this frame
        if audio_player:
            audio_player.get_frame()

        # sleep just long enough to keep real-time playback
        elapsed  = time.perf_counter() - t_start
        target   = frame_idx / fps
        wait_ms  = max(1, int((target - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if audio_player:
        del audio_player
    cv2.destroyAllWindows()

    print(f"  Playback ended.")
    print(f"  Stimulus marker is saved in the recording's _markers.json sidecar.")
    print(f"  The pipeline will use it to align this session to t=0 = stimulus onset.")


def main():
    p = argparse.ArgumentParser(
        description="Play a stimulus video and send an LSL marker at playback start."
    )
    p.add_argument("video", help="path to the video file (mp4, avi, …)")
    p.add_argument("--countdown", type=int, default=3,
                   help="countdown seconds before playback (default 3)")
    p.add_argument("--marker", default="stimulus_start",
                   help="LSL marker string (default 'stimulus_start')")
    p.add_argument("--fullscreen", action="store_true",
                   help="play video fullscreen")
    args = p.parse_args()
    play(args.video, args.marker, args.countdown, args.fullscreen)


if __name__ == "__main__":
    main()
