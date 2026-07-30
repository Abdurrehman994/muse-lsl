"""
make_video_clips.py — cut a few short clips out of a longer source video,
for use as short-video-watching stimuli (see video_task.py).

No ffmpeg dependency -- uses OpenCV (already a project dependency) to read
and re-encode the selected frame ranges.

Usage:
  python make_video_clips.py stimulus.mp4 --out-dir stimuli/clips
  python make_video_clips.py stimulus.mp4 --starts 60 400 700 --length 150
"""
import argparse
import os

import cv2


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
        description="Cut a few short clips out of a longer source video."
    )
    p.add_argument("source", help="path to the source video, e.g. stimulus.mp4")
    p.add_argument("--starts", type=float, nargs="+", default=[60, 400, 700],
                   help="start times in seconds for each clip (default: 60 400 700)")
    p.add_argument("--length", type=float, default=150.0,
                   help="clip length in seconds (default 150 = 2.5 min)")
    p.add_argument("--out-dir", default="stimuli/clips",
                   help="output directory (default stimuli/clips)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.source))[0]
    for i, start_s in enumerate(args.starts, start=1):
        out_path = os.path.join(args.out_dir, f"{base}_clip{i}.mp4")
        extract_clip(args.source, start_s, args.length, out_path)

    print(f"\nDone. {len(args.starts)} clip(s) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
