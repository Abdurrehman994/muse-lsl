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

Two-monitor frequency-tagging (subject A at one rate, subject B at another --
the unambiguous inter-brain test; see pipeline.py --tag-hz-a/--tag-hz-b):
  Terminal 1: python record_both.py 65 Muse_A Muse_B
  Terminal 2: python play_stimulus.py stimuli/checker_6hz.mp4   --monitor 1 --fullscreen --marker tagA_6hz
  Terminal 3: python play_stimulus.py stimuli/checker_7.5hz.mp4 --monitor 2 --fullscreen --marker tagB_7.5hz
  (--monitor needs `pip install screeninfo`; otherwise use --pos X,Y with the
   second monitor's origin from your OS display settings, e.g. --pos 1920,0)

  Dry-run the placement first (no recording/clip needed):
    python play_stimulus.py --test-monitors                          # list monitors
    python play_stimulus.py --test-monitors --monitor 2 --fullscreen # test-card on screen 2
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


def resolve_monitor_origin(monitor, pos):
    """
    Work out the (x, y) top-left pixel where the stimulus window should be
    placed, for multi-monitor frequency-tagging (subject A's clip on one
    screen, subject B's on another).

    --pos "X,Y" is an explicit override (read the second monitor's origin from
    your OS display settings, e.g. "1920,0"). Otherwise --monitor N looks the
    geometry up via the optional `screeninfo` package (1-indexed, ordered as
    the OS reports them). Returns (x, y) or None to leave the window where the
    OS put it (primary screen).
    """
    if pos:
        try:
            x, y = (int(v) for v in pos.replace(" ", "").split(","))
            return x, y
        except ValueError:
            print(f"  WARNING: could not parse --pos '{pos}' (want 'X,Y') -- ignoring.")
            return None
    if monitor:
        try:
            from screeninfo import get_monitors
        except ImportError:
            print("  WARNING: --monitor needs the 'screeninfo' package "
                  "(pip install screeninfo). Use --pos X,Y instead for now -- ignoring.")
            return None
        mons = get_monitors()
        if not (1 <= monitor <= len(mons)):
            print(f"  WARNING: --monitor {monitor} out of range (found {len(mons)} "
                  f"monitor(s)) -- ignoring.")
            return None
        m = mons[monitor - 1]
        print(f"  target monitor {monitor}: {m.width}x{m.height} at ({m.x},{m.y})")
        return m.x, m.y
    return None


def list_monitors():
    """Print the monitors screeninfo can see, with the --monitor index to use
    for each. Returns the list, or None if screeninfo isn't installed."""
    try:
        from screeninfo import get_monitors
    except ImportError:
        print("  screeninfo not installed (pip install screeninfo) -- can't "
              "enumerate monitors. Use --pos X,Y instead.")
        return None
    mons = get_monitors()
    print(f"Detected {len(mons)} monitor(s):")
    for i, m in enumerate(mons, 1):
        prim = " [primary]" if getattr(m, "is_primary", False) else ""
        print(f"  --monitor {i}:  {m.width}x{m.height}  origin ({m.x},{m.y}){prim}")
    return mons


def test_monitor_placement(monitor, pos, fullscreen):
    """
    Dry-run for two-monitor frequency-tagging: enumerate the monitors and, if a
    target is given, pop a labelled test card on it so you can confirm each
    subject's clip will land on the right screen -- WITHOUT starting a recording
    or playing a clip. Run this once before a real session.
    """
    list_monitors()
    origin = resolve_monitor_origin(monitor, pos)
    if origin is None:
        if monitor is None and pos is None:
            print("\nNo --monitor/--pos given, so nothing to place. Re-run e.g. "
                  "'--test-monitors --monitor 2 --fullscreen' to check a target screen.")
        else:
            print("\nCouldn't resolve that target (see warning above) -- no test card shown.")
        return
    win = "MonitorTest"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if origin is not None:
        cv2.moveWindow(win, origin[0], origin[1])
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 800, 600)
    where = f"monitor {monitor}" if monitor is not None else f"pos {pos}"
    card = text_frame((600, 800, 3), [
        f"TEST CARD -- {where}",
        f"origin = {origin}",
        "Right screen for this subject?",
        "Press any key to close.",
    ])
    cv2.imshow(win, card)
    print(f"\nShowing a test card on {where} (origin={origin}). "
          "Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def play(video_path, marker_name, countdown_s, fullscreen, monitor=None, pos=None):
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
    # Move the window onto the target monitor BEFORE going fullscreen -- if you
    # fullscreen first and move after, OpenCV tends to snap back to the primary
    # display, which breaks two-monitor frequency-tagging (each subject's clip
    # must fill its own screen).
    origin = resolve_monitor_origin(monitor, pos)
    if origin is not None:
        cv2.moveWindow(win, origin[0], origin[1])
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

    # ── wait for a recorder to connect (record_both.py's own startup — EEG
    # stream scan + inlet connections — can easily take 10-15s) ───────────
    if not outlet.have_consumers():
        print("\nWaiting for record_both.py to connect to the marker stream "
              "(up to 30s) — make sure it's already running ...")
        waited = 0.0
        while waited < 30.0 and not outlet.have_consumers():
            cv2.imshow(win, text_frame(shape, ["Waiting for recorder...",
                                                f"{waited:.0f}s / 30s"]))
            cv2.waitKey(1)
            outlet.wait_for_consumers(1.0)
            waited += 1.0

    had_consumer = outlet.have_consumers()
    if had_consumer:
        print("  Recorder connected.")
    else:
        print("  WARNING: no recorder connected after 30s. Proceeding anyway, "
              "but the marker will NOT be captured.")

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
    if had_consumer:
        print(f"  A recorder was connected when the marker fired — check that "
              f"recording's _markers.json sidecar to confirm it was captured.")
        print(f"  The pipeline will use it to align this session to t=0 = stimulus onset.")
    else:
        print(f"  WARNING: no recorder was connected when the marker fired — "
              f"it was NOT saved anywhere. This session has no stimulus marker.")


def main():
    p = argparse.ArgumentParser(
        description="Play a stimulus video and send an LSL marker at playback start."
    )
    p.add_argument("video", nargs="?", default=None,
                   help="path to the video file (mp4, avi, …); optional with --test-monitors")
    p.add_argument("--countdown", type=int, default=3,
                   help="countdown seconds before playback (default 3)")
    p.add_argument("--marker", default="stimulus_start",
                   help="LSL marker string (default 'stimulus_start')")
    p.add_argument("--fullscreen", action="store_true",
                   help="play video fullscreen")
    p.add_argument("--monitor", type=int, default=None,
                   help="which monitor to show the clip on (1-indexed), for "
                        "two-monitor frequency-tagging. Needs the 'screeninfo' "
                        "package (pip install screeninfo). Combine with --fullscreen.")
    p.add_argument("--pos", default=None,
                   help="explicit window top-left 'X,Y' pixel (e.g. '1920,0' for a "
                        "second monitor to the right); overrides --monitor and needs "
                        "no extra package. Combine with --fullscreen.")
    p.add_argument("--test-monitors", action="store_true",
                   help="dry-run: list monitors and (with --monitor/--pos) pop a test "
                        "card on the target screen to confirm placement. No recording "
                        "or video needed -- run before a real frequency-tagging session.")
    args = p.parse_args()

    if args.test_monitors:
        test_monitor_placement(args.monitor, args.pos, args.fullscreen)
        return
    if args.video is None:
        p.error("a video path is required (unless using --test-monitors)")
    play(args.video, args.marker, args.countdown, args.fullscreen,
         monitor=args.monitor, pos=args.pos)


if __name__ == "__main__":
    main()
