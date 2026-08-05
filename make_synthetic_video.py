"""
make_synthetic_video.py — procedurally generate an abstract animated video
clip (colored particles bouncing around a slowly drifting-hue background),
for use as a short-video-watching stimulus (see video_task.py) when you
don't have real footage available.

CAVEAT (read before relying on this for the "engaging short video"
hypothesis): this is NOT narratively engaging footage like a movie clip.
The neurocinematics inter-subject-correlation literature (Hasson et al.
2008) that motivates testing short/engaging VIDEO clips specifically found
strong effects for NARRATIVE, emotionally engaging content -- plain
abstract motion is a different kind of visual stimulus (closer to a
sustained-attention/motion-tracking task) and isn't expected to produce
the same effect by the same mechanism. Use this as a stopgap for content
variety/pipeline testing, not a substitute for real footage if the actual
research question is about narrative engagement.

No ffmpeg dependency -- pure OpenCV, same as make_checkerboard.py.

Usage:
  python make_synthetic_video.py stimuli/clips/synth1.mp4
      # random pattern, different every run
  python make_synthetic_video.py stimuli/clips/synth2.mp4 --seed 5 --n-particles 40
  python make_synthetic_video.py stimuli/clips/synth3.mp4 --duration 150
"""
import argparse
import random

import cv2
import numpy as np


def hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


def make_particles(rng, n, w, h):
    particles = []
    for _ in range(n):
        particles.append({
            "x": rng.uniform(0, w), "y": rng.uniform(0, h),
            "vx": rng.uniform(-3, 3), "vy": rng.uniform(-3, 3),
            "r": rng.uniform(8, 30), "hue": rng.uniform(0, 180),
        })
    return particles


def step_particles(particles, w, h):
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        if p["x"] < p["r"] or p["x"] > w - p["r"]:
            p["vx"] *= -1
        if p["y"] < p["r"] or p["y"] > h - p["r"]:
            p["vy"] *= -1
        p["x"] = float(np.clip(p["x"], p["r"], w - p["r"]))
        p["y"] = float(np.clip(p["y"], p["r"], h - p["r"]))


def render_frame(particles, w, h, bg_hue):
    bg = hsv_to_bgr(bg_hue, 60, 40)
    frame = np.full((h, w, 3), bg, dtype=np.uint8)
    for p in particles:
        color = hsv_to_bgr(p["hue"], 200, 230)
        cv2.circle(frame, (int(p["x"]), int(p["y"])), int(p["r"]), color, -1, cv2.LINE_AA)
    return frame


def generate(out_path, duration_s, fps, width, height, n_particles, rng):
    particles = make_particles(rng, n_particles, width, height)
    n_frames = int(duration_s * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"ERROR: could not open video writer for {out_path}")

    bg_hue = rng.uniform(0, 180)
    hue_drift = rng.uniform(-0.05, 0.05)
    for _ in range(n_frames):
        step_particles(particles, width, height)
        bg_hue = (bg_hue + hue_drift) % 180
        writer.write(render_frame(particles, width, height, bg_hue))
    writer.release()


def main():
    p = argparse.ArgumentParser(
        description="Generate an abstract synthetic video clip (bouncing "
                     "colored particles, drifting background hue)."
    )
    p.add_argument("out", help="output path, e.g. stimuli/clips/synth1.mp4")
    p.add_argument("--duration", type=float, default=150.0,
                   help="clip length in seconds (default 150)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--n-particles", type=int, default=25)
    p.add_argument("--seed", type=int, default=None,
                   help="omit (default) for a fresh random pattern every "
                        "run; pass a fixed value for a reproducible one")
    args = p.parse_args()

    rng = random.Random(args.seed)
    generate(args.out, args.duration, args.fps, args.width, args.height,
             args.n_particles, rng)
    print(f"Wrote {args.out}  ({args.duration:.0f}s, {args.n_particles} particles, "
          f"{'seed=' + str(args.seed) if args.seed is not None else 'random'})")


if __name__ == "__main__":
    main()
