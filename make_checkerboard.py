"""
make_checkerboard.py — generate a contrast-reversal checkerboard video, the
standard visual stimulus for driving an SSVEP (steady-state visually evoked
potential). Same idea as peterscarfe.com/contrastRevCheckerboard.html, baked
into an .mp4 so play_stimulus.py can play it while record_both.py records.

Two use:
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


def make_radial_checkerboard(size_px, n_wedges=24, n_rings=8, invert=False):
    """
    Radial "dartboard" contrast-reversal checkerboard -- the classic clinical
    VEP stimulus (concentric rings x angular wedges, alternating parity).
    Rings are equal-AREA (radius ~ sqrt(ring fraction)), not equal-width, so
    each ring carries a similar amount of contrast/pixels despite squares
    visibly growing larger toward the edge -- matching the standard design.
    """
    h = w = size_px
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = h / 2.0, w / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # range (-pi, pi]

    r_max = min(h, w) / 2.0
    r_norm = np.clip(r / r_max, 0.0, 1.0)
    ring_idx = np.clip(np.floor(np.sqrt(r_norm) * n_rings).astype(int), 0, n_rings - 1)
    wedge_idx = np.clip(
        np.floor((theta + np.pi) / (2 * np.pi) * n_wedges).astype(int), 0, n_wedges - 1
    )

    parity = (ring_idx + wedge_idx) % 2
    if invert:
        parity = 1 - parity

    board = (parity * 255).astype(np.uint8)
    board = np.where(r <= r_max, board, 128).astype(np.uint8)  # mid-gray surround
    return board


def make_ring_pattern(size_px, n_rings=8, invert=False):
    """Concentric rings only (no angular wedges) -- a 'bullseye' pattern."""
    h = w = size_px
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = min(h, w) / 2.0
    r_norm = np.clip(r / r_max, 0.0, 1.0)
    ring_idx = np.clip(np.floor(np.sqrt(r_norm) * n_rings).astype(int), 0, n_rings - 1)
    parity = ring_idx % 2
    if invert:
        parity = 1 - parity
    board = (parity * 255).astype(np.uint8)
    return np.where(r <= r_max, board, 128).astype(np.uint8)


def make_wedge_pattern(size_px, n_wedges=24, invert=False):
    """Angular wedges only (no rings) -- a 'pinwheel' pattern."""
    h = w = size_px
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = h / 2.0, w / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    r_max = min(h, w) / 2.0
    wedge_idx = np.clip(
        np.floor((theta + np.pi) / (2 * np.pi) * n_wedges).astype(int), 0, n_wedges - 1
    )
    parity = wedge_idx % 2
    if invert:
        parity = 1 - parity
    board = (parity * 255).astype(np.uint8)
    return np.where(r <= r_max, board, 128).astype(np.uint8)


def make_stripe_pattern(size_px, n_stripes=10, orientation="vertical", invert=False):
    """Simple alternating-bar grating -- a full-field orientation stimulus."""
    h = w = size_px
    length = w if orientation == "vertical" else h
    idx = (np.arange(length) * n_stripes // length) % 2
    if invert:
        idx = 1 - idx
    line = (idx * 255).astype(np.uint8)
    if orientation == "vertical":
        return np.tile(line, (h, 1))
    return np.tile(line.reshape(-1, 1), (1, w))


def build_boards(args):
    """Dispatch to the chosen pattern, returning (board_a, board_b, description)."""
    if args.pattern == "radial":
        a = make_radial_checkerboard(args.size, args.wedges, args.rings, invert=False)
        b = make_radial_checkerboard(args.size, args.wedges, args.rings, invert=True)
        desc = f"radial {args.wedges} wedges x {args.rings} rings"
    elif args.pattern == "rings":
        a = make_ring_pattern(args.size, args.rings, invert=False)
        b = make_ring_pattern(args.size, args.rings, invert=True)
        desc = f"{args.rings} concentric rings (bullseye)"
    elif args.pattern == "wedges":
        a = make_wedge_pattern(args.size, args.wedges, invert=False)
        b = make_wedge_pattern(args.size, args.wedges, invert=True)
        desc = f"{args.wedges} angular wedges (pinwheel)"
    elif args.pattern == "stripes":
        a = make_stripe_pattern(args.size, args.n_stripes, args.orientation, invert=False)
        b = make_stripe_pattern(args.size, args.n_stripes, args.orientation, invert=True)
        desc = f"{args.n_stripes} {args.orientation} stripes"
    else:  # grid
        a = make_checkerboard(args.rows, args.cols, args.cell_px, invert=False)
        b = make_checkerboard(args.rows, args.cols, args.cell_px, invert=True)
        desc = f"{args.rows}x{args.cols} grid squares"
    return a, b, desc


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
    p.add_argument("--pattern",
                   choices=["grid", "radial", "rings", "wedges", "stripes"],
                   default="grid",
                   help="'grid' = square checkerboard (default), "
                        "'radial' = ring+wedge dartboard (classic clinical VEP), "
                        "'rings' = concentric rings only (bullseye), "
                        "'wedges' = angular wedges only (pinwheel), "
                        "'stripes' = alternating bar grating")
    p.add_argument("--rows", type=int, default=8, help="grid pattern only")
    p.add_argument("--cols", type=int, default=8, help="grid pattern only")
    p.add_argument("--cell-px", type=int, default=90, help="grid pattern only")
    p.add_argument("--size", type=int, default=720,
                   help="frame size in px, square (radial/rings/wedges/stripes, default 720)")
    p.add_argument("--wedges", type=int, default=24,
                   help="number of angular wedges (radial/wedges patterns, default 24)")
    p.add_argument("--rings", type=int, default=8,
                   help="number of concentric rings (radial/rings patterns, default 8)")
    p.add_argument("--n-stripes", type=int, default=10, help="stripes pattern only")
    p.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical",
                   help="stripes pattern only")
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

    board_a, board_b, pattern_desc = build_boards(args)

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
          f"reversal rate ~{actual_hz:.2f} Hz  ({pattern_desc})")


if __name__ == "__main__":
    main()
