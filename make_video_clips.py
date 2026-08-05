"""
make_video_clips.py — cut short clips out of one or more source videos, for
use as short-video-watching stimuli (see video_task.py).

No ffmpeg dependency -- uses OpenCV (already a project dependency) to read
and re-encode the selected frame ranges.

Two modes:
  ONE source video  -> multiple clips from different points in it
      (repeated exposure to the SAME long video across a session risks
      habituation -- reduced engagement/entrainment on later viewings).
  MULTIPLE source videos -> one clip per video, pooled together
      (recommended for variety: video_task.py's ClipDeck already shuffles
      through whatever's in --out-dir regardless of original source, so
      dropping clips from several different videos into the same folder
      gives you a varied pool with no other code changes needed).

Random clip selection: if --starts is omitted, start times are chosen
RANDOMLY (a fresh, different random draw every run -- no fixed default and
no seed unless you pass --seed) rather than reusing the same fixed
timestamps every time, so re-running this between sessions gives you new
segments instead of the exact same clips.

Usage:
  python make_video_clips.py stimulus.mp4 --out-dir stimuli/clips
      # random clips every run (default 3, spread across the video)
  python make_video_clips.py stimulus.mp4 --n-clips 5
      # random, but 5 of them
  python make_video_clips.py stimulus.mp4 --seed 42
      # random but reproducible -- same seed = same clips every time
  python make_video_clips.py stimulus.mp4 --starts 60 400 700 --length 150
      # exact, manually chosen start times (old fixed behaviour)
  python make_video_clips.py video1.mp4 video2.mp4 video3.mp4 --length 150
      # one RANDOM clip from each of several source videos
  python make_video_clips.py video1.mp4 video2.mp4 --starts 30 90 --length 150
      # one exact start time per source video, when using multiple sources
"""
import argparse
import os
import random

import cv2


def get_duration_s(src_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames / fps


def random_starts(duration_s, length_s, n_clips, rng):
    """n_clips random start times, stratified across [0, duration_s] so
    they're spread out and (as long as the video is long enough) don't
    overlap, rather than independently drawn and possibly clustered."""
    max_start = max(0.0, duration_s - length_s)
    if max_start <= 0:
        return [0.0] * n_clips
    bin_w = max_start / n_clips
    if bin_w < length_s:
        print(f"  NOTE: {n_clips} clips of {length_s:.0f}s barely fit (or don't) "
              f"in {duration_s:.0f}s -- clips may overlap somewhat.")
    return [rng.uniform(i * bin_w, (i + 1) * bin_w) for i in range(n_clips)]


def extract_clip(src_path, start_s, length_s, out_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * fps)
    n_frames = min(int(length_s * fps), total_frames - start_frame)
    if n_frames <= 0:
        cap.release()
        raise SystemExit(f"ERROR: start_s={start_s} is beyond the end of "
                          f"{src_path} ({total_frames / fps:.1f}s long)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"ERROR: could not open video writer for {out_path}")

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    actual_s = n_frames / fps
    print(f"  Wrote {out_path}  ({actual_s:.1f}s, frames {start_frame}-{start_frame + n_frames})")
    return actual_s


def main():
    p = argparse.ArgumentParser(
        description="Cut short clips out of one or more source videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sources", nargs="+",
                   help="one or more source video paths, e.g. stimulus.mp4, "
                        "or video1.mp4 video2.mp4 video3.mp4 for variety")
    p.add_argument("--starts", type=float, nargs="+", default=None,
                   help="exact start times in seconds. With ONE source: one "
                        "or more clip start times from that video. With "
                        "MULTIPLE sources: either one value applied to "
                        "every source, or one value per source. If "
                        "omitted (the default), start times are chosen "
                        "RANDOMLY instead -- see --n-clips/--seed.")
    p.add_argument("--n-clips", type=int, default=3,
                   help="(single-source mode only, when --starts is "
                        "omitted) how many random clips to cut (default 3)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for clip selection. Omit (default) for "
                        "a fresh random draw every run; pass a fixed value "
                        "for a reproducible one.")
    p.add_argument("--length", type=float, default=150.0,
                   help="clip length in seconds (default 150 = 2.5 min)")
    p.add_argument("--out-dir", default="stimuli/clips",
                   help="output directory (default stimuli/clips)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    n_written = 0

    if len(args.sources) == 1:
        src = args.sources[0]
        if args.starts:
            starts = args.starts
        else:
            duration_s = get_duration_s(src)
            starts = random_starts(duration_s, args.length, args.n_clips, rng)
            print(f"  Randomly chosen start times (s): "
                  f"{[round(s, 1) for s in starts]}")
        base = os.path.splitext(os.path.basename(src))[0]
        for i, start_s in enumerate(starts, start=1):
            out_path = os.path.join(args.out_dir, f"{base}_clip{i}.mp4")
            extract_clip(src, start_s, args.length, out_path)
            n_written += 1
    else:
        if args.starts:
            starts = args.starts
            if len(starts) == 1:
                starts = starts * len(args.sources)
            if len(starts) != len(args.sources):
                raise SystemExit(
                    f"ERROR: --starts has {len(starts)} value(s) but "
                    f"{len(args.sources)} source videos were given -- pass "
                    "either one value (applied to all) or one per source."
                )
        else:
            starts = []
            for src in args.sources:
                duration_s = get_duration_s(src)
                starts.append(random_starts(duration_s, args.length, 1, rng)[0])
            print(f"  Randomly chosen start times (s): "
                  f"{[round(s, 1) for s in starts]}")
        for src, start_s in zip(args.sources, starts):
            base = os.path.splitext(os.path.basename(src))[0]
            out_path = os.path.join(args.out_dir, f"{base}_clip.mp4")
            extract_clip(src, start_s, args.length, out_path)
            n_written += 1

    print(f"\nDone. {n_written} clip(s) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
