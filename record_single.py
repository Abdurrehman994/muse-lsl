"""
record_single.py — record ONE Muse EEG stream, then put it onto a clean,
uniform-timestamp grid WITH gap detection.

The Muse hardware always streams EEG at 256 Hz. This script records the
stream at native rate, shows a live single-headset EEG view the whole time,
then resamples onto a TARGET_FS timeline (anti-aliased first only if the
source is meaningfully faster than TARGET_FS — at the default TARGET_FS=256
this is a no-op resample/grid-snap, not a downsample, so you keep full
256 Hz resolution).

Any grid point that falls inside a real timestamp gap on the source stream
is flagged in an `is_gap` column and the EEG values are set to NaN there, so
HyPyP/MNE won't compute phase on invented data.

LIVE VIEW:
While recording, a single-panel plot (TP9/AF7/AF8/TP10) shows the live
signal so you can check contact/quality in real time. Close the plot
window any time to stop the recording early.

Usage:
    python record_single.py                  # 60s, auto-detect stream
    python record_single.py 120               # 120s
    python record_single.py 120 Muse          # custom duration + stream name

Start the Muse stream before running this:
    muselsl stream --address <MAC> --name Muse

Output (in recordings/):
    <timestamp>_<STREAM_NAME>.csv     <- TARGET_FS Hz grid, relative time, is_gap + NaNs
"""

import sys
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import resolve_streams, StreamInlet, local_clock

# scipy is optional; if missing we fall back to plain interpolation (no anti-alias)
try:
    from scipy.signal import butter, filtfilt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ---- config ----
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 60          # seconds
STREAM_NAME = sys.argv[2] if len(sys.argv) > 2 else "Muse"
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
OUT_DIR = "recordings"
TARGET_FS = 256               # output sampling rate (Hz) — 256 = native Muse rate,
                               # so this just grids/cleans the data without downsampling

# live single-headset view settings
WIN_SECONDS = 5                          # rolling window shown on the live plot
OFFSETS = np.array([300, 200, 100, 0])   # vertical spacing so channels don't overlap

# Anything bigger than this between consecutive raw samples is a "real" gap.
# At a nominal 256 Hz the expected spacing is ~3.9 ms. 20 ms means we missed
# ~5 samples in a row — well past "minor jitter" territory.
GAP_THRESHOLD_MS = 20.0


def find_one_stream(name, wait_time=5):
    print(f"\nScanning for an EEG stream matching '{name}' ...")
    streams = [s for s in resolve_streams(wait_time=wait_time) if s.type() == "EEG"]
    for s in streams:
        if name.lower() in s.name().lower() or name.lower() in s.source_id().lower():
            print(f"  matched -> '{s.name()}'")
            return s
    if streams:
        print("  No name match. EEG stream(s) seen instead:")
        for s in streams:
            print(f"     {s.name()}")
    else:
        print("  No EEG streams found at all.")
    return None


def record_one(name, duration):
    """
    Connect to a single named stream, record it for `duration` seconds, and
    show a live single-headset EEG view the whole time so you can check
    signal quality as it happens. Close the plot window any time to stop
    the recording early.
    """
    stream = find_one_stream(name)
    if stream is None:
        raise RuntimeError(
            f"No EEG stream found for '{name}'. Is muselsl running with --name {name}?"
        )

    inlet = StreamInlet(stream, max_buflen=duration + 5)
    fs = int(inlet.info().nominal_srate())
    print(f"Connected to '{stream.name()}' ({fs} Hz). Recording {duration}s ...")
    print("Sit still for the first few seconds so the signal settles.")
    print("Close the plot window any time to stop this recording early.")

    data = []
    marker_events = []
    win_samples = max(int(WIN_SECONDS * fs), 1)
    buf_holder = {"buf": np.full((win_samples, 4), np.nan)}
    state = {"start": None, "marker_inlet": None, "last_scan": 0.0}

    print("  Waiting for play_stimulus.py to start ...\n")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"Live EEG — {name}")

    def update(_):
        if state["start"] is None:
            state["start"] = local_clock()
        elapsed = local_clock() - state["start"]

        try:
            samples, ts = inlet.pull_chunk(timeout=0.0)
        except Exception as e:
            print(f"\n  pull_chunk error: {e}")
            samples, ts = None, None

        if samples:
            s = np.array(samples, dtype=float)[:, :4]
            n = len(s)
            for samp, t in zip(samples, ts):
                data.append([t] + list(samp[:4]))
            buf = buf_holder["buf"]
            buf = np.roll(buf, -n, axis=0)
            buf[-n:, :] = s
            buf_holder["buf"] = buf

        # retry connecting to marker stream every 2s until found
        if state["marker_inlet"] is None:
            now = local_clock()
            if now - state["last_scan"] > 2.0:
                state["last_scan"] = now
                streams = [s for s in resolve_streams(wait_time=0.2)
                           if s.name() == "StimulusMarkers"]
                if streams:
                    state["marker_inlet"] = StreamInlet(streams[0])
                    print(f"\n  [MARKER STREAM] Connected at {elapsed:.1f}s — "
                          f"ready to capture stimulus onset.")

        # poll for stimulus markers
        if state["marker_inlet"]:
            sample, ts_m = state["marker_inlet"].pull_sample(timeout=0.0)
            if sample:
                rel_t = ts_m - state["start"] if state["start"] else 0.0
                marker_events.append({
                    "marker": sample[0],
                    "lsl_timestamp": ts_m,
                    "rel_time_s": rel_t,
                })
                print(f"\n  [MARKER] '{sample[0]}' at {rel_t:.3f}s into recording")

        buf = buf_holder["buf"]
        t_axis = np.linspace(-WIN_SECONDS, 0, len(buf))

        ax.clear()
        for i in range(4):
            col = buf[:, i]
            valid = ~np.isnan(col)
            base = col[valid].mean() if valid.any() else 0.0
            ax.plot(t_axis, col - base + OFFSETS[i], lw=0.7)
        ax.set_yticks(OFFSETS)
        ax.set_yticklabels(CH_NAMES)
        ax.set_xlabel("Time (s)")

        marker_str = (f"  |  marker @ {marker_events[-1]['rel_time_s']:.1f}s"
                      if marker_events else "")
        ax.set_title(f"{elapsed:5.1f}s / {duration}s  |  {len(data)} samples{marker_str}")

        if elapsed >= duration:
            plt.close(fig)

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()  # blocks until duration elapses or the window is closed

    inlet.close_stream()
    df = pd.DataFrame(data, columns=["lsl_timestamp"] + CH_NAMES)
    return df, fs, marker_events


def effective_fs(df):
    if len(df) < 2:
        return 0.0
    dur = df["lsl_timestamp"].iloc[-1] - df["lsl_timestamp"].iloc[0]
    return len(df) / dur if dur > 0 else 0.0


def build_gap_mask(rel_ts, grid, gap_threshold_s):
    """
    For each point on `grid` (relative time), decide whether it sits inside a
    real gap in `rel_ts` (also relative time, same stream).

    A "real gap" is any raw inter-sample interval larger than gap_threshold_s.
    Grid points before the first sample or after the last are also marked bad.

    Returns: boolean array, True where the grid point is INSIDE a gap.
    """
    rel_ts = np.asarray(rel_ts, dtype=float)
    mask = np.zeros(len(grid), dtype=bool)

    mask |= grid < rel_ts[0]
    mask |= grid > rel_ts[-1]

    idx = np.searchsorted(rel_ts, grid, side="right")
    idx = np.clip(idx, 1, len(rel_ts) - 1)

    interval_size = rel_ts[idx] - rel_ts[idx - 1]
    mask |= interval_size > gap_threshold_s

    return mask


def resample_to_grid(df, eff_fs, grid, target_fs, gap_threshold_s):
    """
    Put the stream onto `grid` (uniform target_fs timeline, in RELATIVE
    time — seconds since the recording's own start).

    If the stream is meaningfully faster than target (the clean ~256 Hz
    case), anti-alias first: interpolate to a uniform grid at native rate,
    low-pass below the target Nyquist, then resample onto the grid. A
    starved stream close to target rate just gets aligned directly.

    Also returns a boolean is_gap mask: True wherever a grid point sits
    inside a real gap on the source stream. EEG values there are set to NaN.
    """
    rel_ts = df["rel_time"].values.astype(float)
    out = {"time_s": grid}
    do_antialias = _HAVE_SCIPY and eff_fs > target_fs * 1.5

    gap_mask = build_gap_mask(rel_ts, grid, gap_threshold_s)

    if do_antialias:
        n_uni = max(int((rel_ts[-1] - rel_ts[0]) * eff_fs), 2)
        uni_t = rel_ts[0] + np.arange(n_uni) / eff_fs
        b, a = butter(4, (target_fs * 0.45) / (eff_fs / 2.0), btype="low")

    for ch in CH_NAMES:
        x = df[ch].values.astype(float)
        if do_antialias:
            xu = np.interp(uni_t, rel_ts, x)
            xu = filtfilt(b, a, xu)
            y = np.interp(grid, uni_t, xu)
        else:
            y = np.interp(grid, rel_ts, x)

        y[gap_mask] = np.nan
        out[ch] = y

    out["is_gap"] = gap_mask
    return pd.DataFrame(out), gap_mask


def gap_report(name, df):
    if len(df) < 3:
        return
    dt = np.diff(df["lsl_timestamp"].values) * 1000.0  # ms
    n_big = int((dt > GAP_THRESHOLD_MS).sum())
    flag = (f"  <-- {n_big} GAP(S) > {GAP_THRESHOLD_MS:.0f} ms"
            if n_big > 0 else "  (clean)")
    print(f"      spacing: median {np.median(dt):.1f} ms | "
          f"std {dt.std():.1f} ms | max {dt.max():.1f} ms{flag}")


def report(name, df, fs):
    if len(df) < 2:
        print(f"  {name}: NO DATA - check the stream!")
        return
    dur = df["lsl_timestamp"].iloc[-1] - df["lsl_timestamp"].iloc[0]
    eff = len(df) / dur if dur > 0 else 0
    print(f"  {name} captured: {len(df)} samples over {dur:.1f}s "
          f"(~{eff:.0f} Hz, nominal {fs} Hz)")


def longest_clean_run_s(bad_mask, fs):
    if bad_mask.all():
        return 0.0
    runs = np.diff(np.concatenate(([1], bad_mask.astype(int), [1])))
    starts = np.where(runs == -1)[0]
    ends = np.where(runs == 1)[0]
    if len(starts) == 0:
        return 0.0
    return float((ends - starts).max() / fs)


# ============================================================
# 1. Record the single stream
# ============================================================
print("=" * 60)
print(f"Recording '{STREAM_NAME}' for {DURATION}s")
print(f"Make sure {STREAM_NAME}'s muselsl stream is running.")
print("=" * 60)
df, fs, marker_events = record_one(STREAM_NAME, DURATION)
df["rel_time"] = df["lsl_timestamp"] - df["lsl_timestamp"].iloc[0]

# ============================================================
# 2. Resample onto a relative-time TARGET_FS grid
# ============================================================
print("\nProcessing ...")
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = os.path.join(OUT_DIR, f"{stamp}_{STREAM_NAME}.csv")

eff = effective_fs(df)
gap_threshold_s = GAP_THRESHOLD_MS / 1000.0
saved = False

if len(df) > 1:
    rec_dur = df["rel_time"].iloc[-1]
    n_grid = int(rec_dur * TARGET_FS)

    if n_grid >= 2:
        grid = np.arange(n_grid) / TARGET_FS
        df64, mask = resample_to_grid(df, eff, grid, TARGET_FS, gap_threshold_s)
        df64.to_csv(path, index=False)
        saved = True

        if marker_events:
            sidecar = path.replace(".csv", "_markers.json")
            with open(sidecar, "w") as f:
                json.dump(marker_events, f, indent=2)
            print(f"  Stimulus markers saved: {sidecar}")
    else:
        print("  Recording too short to resample - nothing saved.")
else:
    print("  Not enough data recorded - nothing saved.")

# ============================================================
# 3. Reports
# ============================================================
if saved:
    print(f"\nSaved ({TARGET_FS} Hz, relative time, with gap flags):\n  {path}\n")

report(STREAM_NAME, df, fs)
gap_report(STREAM_NAME, df)

if not _HAVE_SCIPY:
    print("\n  NOTE: scipy not found - skipped anti-alias filtering on the fast "
          "stream.\n  Install it (pip install scipy) or 50 Hz mains can alias into "
          "your bands.")

if saved:
    clean_frac = float((~mask).mean())
    clean_s = clean_frac * rec_dur
    longest_s = longest_clean_run_s(mask, TARGET_FS)

    print(f"\n  Recording window: {rec_dur:.1f}s -> {TARGET_FS} Hz grid")
    print(f"  Gap-free:                     {clean_s:.1f}s "
          f"({clean_frac*100:.1f}% of window)")
    print(f"  Longest contiguous clean run: {longest_s:.1f}s")

    if clean_frac < 0.5:
        print("\n  WARNING: more than half the recording is in a gap. "
              "Check headset fit/battery and BLE distance.")
    elif longest_s < 10.0:
        print("\n  NOTE: longest clean run is short. Epoch-based analysis (PLV, "
              "coherence)\n  may struggle to find enough usable windows.")