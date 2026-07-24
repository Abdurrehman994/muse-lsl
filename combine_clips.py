"""
combine_clips.py - concatenate several video clips into a single .mp4, and
write a sidecar JSON describing each segment's time offsets within the
combined file.

Use this with play_stimulus.py's single marker + pipeline.py's stimulus
alignment: since segment durations are fixed and known in advance, one
stimulus_start marker plus this sidecar is enough to slice the recording by
condition afterward -- no per-segment LSL markers needed (that's the
play_sequence.py approach instead; use that one if you'd rather keep clips
as separate files with their own markers).

Usage:
  python combine_clips.py stimuli/checker_combined.mp4 \\
      stimuli/checker_grid_6hz.mp4 stimuli/checker_radial_6hz.mp4 \\
      stimuli/checker_rings_6hz.mp4 stimuli/checker_wedges_6hz.mp4 \\
      stimuli/checker_stripes_6hz.mp4

Writes:
  stimuli/checker_combined.mp4
  stimuli/checker_combined_segments.json
    [{"name": "checker_grid_6hz", "start_s": 0.0, "end_s": 30.0}, ...]

Then:
  Terminal 1: python record_both.py 155 A1 6F
  Terminal 2: python play_stimulus.py stimuli/checker_combined.mp4
  ...
  python pipeline.py <A.csv> <B.csv> --surrogate 100   # whole-clip analysis

For per-segment analysis, load the _segments.json sidecar and trim each
subject's Raw to [onset + start_s, onset + end_s] before epoching -- reuse
pipeline.py's load_csv_to_raw / preprocess / epoch_with_gap_rejection /
plv_manual building blocks per segment, same pattern as compare_conditions.py.
"""
import argparse
import json
import os

import cv2


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("out", help="output combined video path")
    p.add_argument("clips", nargs="+", help="clips to concatenate, in order")
    args = p.parse_args()

    for c in args.clips:
        if not os.path.exists(c):
            raise SystemExit(f"ERROR: file not found: {c}")

    caps = [cv2.VideoCapture(c) for c in args.clips]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    for c, cap in zip(args.clips, caps):
        cfps = cap.get(cv2.CAP_PROP_FPS)
        if abs(cfps - fps) > 0.01:
            raise SystemExit(
                f"ERROR: {c} has fps={cfps}, expected {fps} (all clips must "
                f"share the same fps -- regenerate with matching --fps)"
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"ERROR: could not open video writer for {args.out}")

    segments = []
    t = 0.0
    for clip_path, cap in zip(args.clips, caps):
        n_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            n_written += 1
        cap.release()
        duration = n_written / fps
        name = os.path.splitext(os.path.basename(clip_path))[0]
        segments.append({
            "name": name,
            "start_s": round(t, 4),
            "end_s": round(t + duration, 4),
        })
        print(f"  + {clip_path}: {n_written} frames ({duration:.1f}s)  ->  "
              f"[{segments[-1]['start_s']:.1f}s, {segments[-1]['end_s']:.1f}s)")
        t += duration

    writer.release()

    sidecar = os.path.splitext(args.out)[0] + "_segments.json"
    with open(sidecar, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"\nWrote {args.out}  ({t:.1f}s total, {fps:.0f}fps, {w}x{h})")
    print(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
