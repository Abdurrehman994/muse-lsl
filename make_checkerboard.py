"""
make_checkerboard.py — generate a contrast-reversal checkerboard video, the
standard visual stimulus for driving an SSVEP (steady-state visually evoked
potential). Same idea as peterscarfe.com/contrastRevCheckerboard.html, baked
into an .mp4 so play_stimulus.py can play it while record_both.py records.

Two uses:
  1. Positive control (sanity check): both subjects watch the SAME clip.
     If the pipeline can't find strong PLV/circ-corr at the reversal
     frequency here, the problem is measurement, not "no synchrony".
  2. Frequency-tagging same-vs-different: give subject A a clip at one
     reversal rate and subject B a *different* rate (needs two monitors —
     see --pos below). Each brain should track its own screen's frequency;
     any coupling between the two subjects' signals at each other's
     frequency (or at a shared harmonic) is a real cross-brain effect, not
     just "both looking at the same flicker".

Usage:
  python make_checkerboard.py stimuli/checker_6hz.mp4 --reversal-hz 6 --duration 60
  python make_checkerboard.py stimuli/checker_8hz.mp4 --reversal-hz 8 --duration 60
  python make_checkerboard.py stimuli/checker_10hz.mp4 --reversal-hz 10 --duration 60 --fixation

Then, e.g. for the positive control:
  Terminal 1: python record_both.py 65 Muse_A Muse_B
  Terminal 2: python play_stimulus.py stimuli/checker_6hz.mp4 --marker checker_6hz_start
  ...
  python pipeline.py recordings/<stamp>_A.csv recordings/<stamp>_B.csv --surrogate 100
"""
import argparse

import cv2
import numpy as np


def make_checkerboard(rows, cols, cell_px, invert=False):
    board = np.zeros((rows, cols), dtype=np.uint8)
    board[0::2, 0::2] = 1
    board[1::2, 1::2] = 1
    if invert:
        board = 1 - board
    return np.kron(board, np.ones((cell_px, cell_px), dtype=np.uint8)) * 255


def to_bgr(gray, draw_fixation):
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if draw_fixation:
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 3)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 3)
    return frame


def main():
    p = argparse.ArgumentParser(
        description="Generate a contrast-reversal checkerboard video (SSVEP stimulus).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("out", help="output video path, e.g. stimuli/checker_6hz.mp4")
    p.add_argument("--reversal-hz", type=float, default=6.0,
                   help="pattern flips per second (default 6)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds (default 60)")
    p.add_argument("--fps", type=int, default=60,
                   help="video fps, must be >= 2x reversal-hz (default 60 -- "
                        "exactly represents common SSVEP rates: 5, 6, 7.5, "
                        "10, 15 Hz)")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--cell-px", type=int, default=90)
    p.add_argument("--fixation", action="store_true",
                   help="draw a small central fixation cross (recommended — "
                        "keeps gaze/eye-movement artifacts consistent)")
    args = p.parse_args()

    if args.fps < 2 * args.reversal_hz:
        new_fps = int(np.ceil(2 * args.reversal_hz))
        print(f"  fps={args.fps} can't represent {args.reversal_hz} Hz reversal "
              f"(need >= {2*args.reversal_hz:.0f}). Raising fps to {new_fps}.")
        args.fps = new_fps

    # frames_per_flip must be an integer, but actual_hz = fps / (2 * k) is
    # NOT linear in k -- rounding k to the nearest integer does not give the
    # nearest achievable Hz. Search the two neighboring integers and pick
    # whichever's resulting Hz is actually closest to what was requested.
    ideal_k = args.fps / (2 * args.reversal_hz)
    candidates = sorted({max(1, int(np.floor(ideal_k))), max(1, int(np.ceil(ideal_k)))})
    frames_per_flip = min(candidates, key=lambda k: abs(args.fps / (2 * k) - args.reversal_hz))
    actual_hz = args.fps / (2 * frames_per_flip)
    if abs(actual_hz - args.reversal_hz) > 0.05:
        print(f"  NOTE: {args.reversal_hz} Hz isn't exactly representable at "
              f"{args.fps} fps; using the closest achievable rate: {actual_hz:.3f} Hz "
              f"(exactly achievable rates near this fps: "
              f"{[round(args.fps/(2*k), 3) for k in range(1, 8)]})")

    board_a = make_checkerboard(args.rows, args.cols, args.cell_px, invert=False)
    board_b = make_checkerboard(args.rows, args.cols, args.cell_px, invert=True)
    frame_a = to_bgr(board_a, args.fixation)
    frame_b = to_bgr(board_b, args.fixation)
    h, w = frame_a.shape[:2]

    n_frames = int(args.duration * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"ERROR: could not open video writer for {args.out}")

    for i in range(n_frames):
        flip_idx = i // frames_per_flip
        writer.write(frame_a if flip_idx % 2 == 0 else frame_b)
    writer.release()

    print(f"Wrote {args.out}")
    print(f"  {w}x{h}  {args.fps}fps  {args.duration:.0f}s  "
          f"reversal rate ~{actual_hz:.2f} Hz  ({args.rows}x{args.cols} squares)")


if __name__ == "__main__":
    main()
