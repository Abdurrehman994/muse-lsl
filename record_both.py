"""
ICA
record_both.py — record TWO simultaneous Muse EEG streams for hyperscanning.

Both headsets must be streaming via muselsl BEFORE running this script:

    Terminal 1:  muselsl stream --address <MAC_A> --name Muse_A
    Terminal 2:  muselsl stream --address <MAC_B> --name Muse_B
    Terminal 3:  python stimulus_marker.py          (optional, for IBS alignment)
    Terminal 4:  python record_both.py 120

Both streams share the same LSL clock, so their timestamps are directly
comparable and no post-hoc alignment is needed (beyond trimming to the
stimulus marker if you use one).

Usage:
    python record_both.py                        # 60s, Muse_A + Muse_B
    python record_both.py 120                    # 120s
    python record_both.py 120 Muse_A Muse_B      # custom stream names

Output (in recordings/):
    <stamp>_Muse_A.csv   <stamp>_Muse_A_markers.json
    <stamp>_Muse_B.csv   <stamp>_Muse_B_markers.json

Feed directly into pipeline.py:
    python pipeline.py recordings/<stamp>_Muse_A.csv recordings/<stamp>_Muse_B.csv
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

try:
    from scipy.signal import butter, filtfilt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ---- config ----
DURATION         = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NAME_A           = sys.argv[2] if len(sys.argv) > 2 else "Muse_A"
NAME_B           = sys.argv[3] if len(sys.argv) > 3 else "Muse_B"
CH_NAMES         = ["TP9", "AF7", "AF8", "TP10"]
OUT_DIR          = "recordings"
TARGET_FS        = 256
WIN_SECONDS      = 5
GAP_THRESHOLD_MS = 20.0
STREAM_DROP_TIMEOUT_S = 3.0
STREAM_DROP_GRACE_S   = 10.0

# vertical offsets so channels don't overlap on the live plot
OFFSETS = np.array([300, 200, 100, 0])


# ============================================================
# Stream discovery
# ============================================================

def find_eeg_streams(wait_time=5):
    print(f"\nScanning for EEG streams (up to {wait_time}s) ...")
    streams = [s for s in resolve_streams(wait_time=wait_time) if s.type() == "EEG"]
    print(f"  Found {len(streams)} EEG stream(s):")
    for s in streams:
        print(f"    '{s.name()}'  source_id={s.source_id()}")
    return streams


def pick_stream(streams, name, skip=0):
    """Return the (skip+1)-th stream whose name or source_id contains `name`."""
    name_l = name.lower()
    count = 0
    for s in streams:
        if name_l in s.name().lower() or name_l in s.source_id().lower():
            if count == skip:
                return s
            count += 1
    return None


# ============================================================
# Gap detection + grid resampling  (same logic as record_single.py)
# ============================================================

def build_gap_mask(rel_ts, grid, gap_threshold_s):
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


def gap_summary(name, df_raw, df_grid, mask, target_fs):
    dur = df_raw["lsl_timestamp"].iloc[-1] - df_raw["lsl_timestamp"].iloc[0]
    eff = len(df_raw) / dur if dur > 0 else 0
    dt_ms = np.diff(df_raw["lsl_timestamp"].values) * 1000
    n_big = int((dt_ms > GAP_THRESHOLD_MS).sum())
    clean_frac = float((~mask).mean())
    longest_clean = 0.0
    if not mask.all():
        runs = np.diff(np.concatenate(([1], mask.astype(int), [1])))
        starts = np.where(runs == -1)[0]
        ends = np.where(runs == 1)[0]
        if len(starts):
            longest_clean = float((ends - starts).max() / target_fs)
    print(f"  {name}: {len(df_raw)} raw samples  {dur:.1f}s  ~{eff:.0f} Hz")
    print(f"     gaps>{GAP_THRESHOLD_MS:.0f}ms: {n_big}  |  "
          f"clean: {clean_frac*100:.1f}%  |  "
          f"longest clean run: {longest_clean:.1f}s")
    if clean_frac < 0.5:
        print(f"  WARNING {name}: >50% gap — check headset fit/battery/BLE distance.")
    elif longest_clean < 10.0:
        print(f"  NOTE {name}: longest clean run < 10s — epoch analysis may struggle.")


# ============================================================
# Main recording loop
# ============================================================

def record_both(name_a, name_b, duration):
    all_streams = find_eeg_streams(wait_time=8)

    stream_a = pick_stream(all_streams, name_a)
    stream_b = pick_stream(all_streams, name_b, skip=1 if name_a == name_b else 0)

    missing = []
    if stream_a is None:
        missing.append(name_a)
    if stream_b is None:
        missing.append(name_b)
    if missing:
        print(f"\nERROR: could not find stream(s): {missing}")
        if all_streams:
            print("Available EEG streams right now:")
            for s in all_streams:
                print(f"  - '{s.name()}'  source_id={s.source_id()}")
        else:
            print("No EEG streams were discovered at all.")
        print("Make sure muselsl stream is running for each headset, e.g.:")
        print(f"  muselsl stream --address <MAC_A> --name {name_a}")
        print(f"  muselsl stream --address <MAC_B> --name {name_b}")
        print("If you are unsure of the stream names, run muselsl list in the terminals that start the streams.")
        raise SystemExit(1)

    inlet_a = StreamInlet(stream_a, max_buflen=duration + 5)
    inlet_b = StreamInlet(stream_b, max_buflen=duration + 5)
    fs_a = int(inlet_a.info().nominal_srate())
    fs_b = int(inlet_b.info().nominal_srate())
    print(f"\nConnected to '{stream_a.name()}' ({fs_a} Hz) as Subject A")
    print(f"Connected to '{stream_b.name()}' ({fs_b} Hz) as Subject B")
    print(f"Recording {duration}s. Close the plot window to stop early.\n")

    # optional stimulus marker stream — checked now, and re-checked
    # throughout the whole recording in case it starts later
    marker_inlet = None
    marker_streams = [s for s in resolve_streams(wait_time=1.0)
                      if s.name() == "StimulusMarkers"]
    if marker_streams:
        marker_inlet = StreamInlet(marker_streams[0])
        print("  Stimulus marker stream connected — press Enter in stimulus_marker.py when ready.\n")
    else:
        print("  No stimulus marker stream yet — will keep watching for one for the "
              "rest of this recording. Run stimulus_marker.py or play_stimulus.py "
              "any time before firing the marker, in any order relative to this script.\n")

    data_a, data_b = [], []
    marker_events = []
    win_samples = max(int(WIN_SECONDS * max(fs_a, fs_b)), 1)
    buf = {
        "a": np.full((win_samples, 4), np.nan),
        "b": np.full((win_samples, 4), np.nan),
    }
    state = {
        "start": None,
        "last_marker_check": -999.0,
        "last_a": None,
        "last_b": None,
        "stop_reason": None,
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Live EEG — {name_a} (top) vs {name_b} (bottom)")

    def pull_into(inlet, store, buf_key, n_ch=4):
        try:
            samples, ts = inlet.pull_chunk(timeout=0.0)
        except Exception:
            return 0
        if samples:
            s = np.array(samples, dtype=float)[:, :n_ch]
            n = len(s)
            for samp, t in zip(samples, ts):
                store.append([t] + list(samp[:n_ch]))
            b = buf[buf_key]
            b = np.roll(b, -n, axis=0)
            b[-n:, :] = s
            buf[buf_key] = b
            return n
        return 0

    def update(_):
        nonlocal marker_inlet
        if state["start"] is None:
            state["start"] = local_clock()
        elapsed = local_clock() - state["start"]

        pulled_a = pull_into(inlet_a, data_a, "a")
        pulled_b = pull_into(inlet_b, data_b, "b")
        if pulled_a:
            state["last_a"] = elapsed
        if pulled_b:
            state["last_b"] = elapsed

        if elapsed >= STREAM_DROP_GRACE_S:
            stale = []
            if state["last_a"] is None or elapsed - state["last_a"] > STREAM_DROP_TIMEOUT_S:
                stale.append(name_a)
            if state["last_b"] is None or elapsed - state["last_b"] > STREAM_DROP_TIMEOUT_S:
                stale.append(name_b)
            if stale:
                state["stop_reason"] = (
                    f"stream timeout detected for {', '.join(stale)} "
                    f"(no samples for > {STREAM_DROP_TIMEOUT_S:.1f}s)"
                )
                print(f"\n  WARNING: stream dropout detected — {state['stop_reason']}.")
                print("           Stopping early to avoid a one-sided tail.")
                plt.close(fig)
                return

        if (marker_inlet is None
                and elapsed - state["last_marker_check"] >= 1.0):
            state["last_marker_check"] = elapsed
            found = [s for s in resolve_streams(wait_time=0.2)
                     if s.name() == "StimulusMarkers"]
            if found:
                marker_inlet = StreamInlet(found[0])
                print(f"\n  Stimulus marker stream connected at {elapsed:.1f}s "
                      f"into recording.\n")

        if marker_inlet:
            sample, ts_m = marker_inlet.pull_sample(timeout=0.0)
            if sample:
                rel_t = ts_m - state["start"] if state["start"] else 0.0
                marker_events.append({
                    "marker": sample[0],
                    "lsl_timestamp": ts_m,
                    "rel_time_s": rel_t,
                })
                print(f"\n  [MARKER] '{sample[0]}' at {rel_t:.3f}s")

        t_axis = np.linspace(-WIN_SECONDS, 0, win_samples)
        for ax, bk, label in zip(axes, ["a", "b"], [name_a, name_b]):
            ax.clear()
            for i in range(4):
                col = buf[bk][:, i]
                valid = ~np.isnan(col)
                base = col[valid].mean() if valid.any() else 0.0
                ax.plot(t_axis, col - base + OFFSETS[i], lw=0.6)
            ax.set_yticks(OFFSETS)
            ax.set_yticklabels(CH_NAMES)
            ax.set_ylabel(label)

        marker_str = (f"  |  marker @ {marker_events[-1]['rel_time_s']:.1f}s"
                      if marker_events else "")
        axes[0].set_title(
            f"{elapsed:5.1f}s / {duration}s  |  A:{len(data_a)} B:{len(data_b)} samples"
            f"{marker_str}"
        )
        axes[-1].set_xlabel("Time (s)")

        if elapsed >= duration:
            plt.close(fig)

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    if state["stop_reason"]:
        print(f"  Recording stopped early due to stream dropout: {state['stop_reason']}")

    inlet_a.close_stream()
    inlet_b.close_stream()

    df_a = pd.DataFrame(data_a, columns=["lsl_timestamp"] + CH_NAMES)
    df_b = pd.DataFrame(data_b, columns=["lsl_timestamp"] + CH_NAMES)
    return df_a, df_b, fs_a, fs_b, marker_events, state["stop_reason"]


# ============================================================
# Post-processing + save
# ============================================================

def process_and_save(df_raw, fs_nominal, stream_name, stamp, marker_events):
    if len(df_raw) < 2:
        print(f"  {stream_name}: not enough data — nothing saved.")
        return None, None

    df_raw["rel_time"] = df_raw["lsl_timestamp"] - df_raw["lsl_timestamp"].iloc[0]
    dur = df_raw["rel_time"].iloc[-1]
    eff_fs = len(df_raw) / dur if dur > 0 else 0

    n_grid = int(dur * TARGET_FS)
    if n_grid < 2:
        print(f"  {stream_name}: recording too short — nothing saved.")
        return None, None

    grid = np.arange(n_grid) / TARGET_FS
    gap_threshold_s = GAP_THRESHOLD_MS / 1000.0
    df_grid, mask = resample_to_grid(df_raw, eff_fs, grid, TARGET_FS, gap_threshold_s)

    path = os.path.join(OUT_DIR, f"{stamp}_{stream_name}.csv")
    df_grid.to_csv(path, index=False)
    print(f"  Saved: {path}")

    if marker_events:
        sidecar = path.replace(".csv", "_markers.json")
        with open(sidecar, "w") as f:
            json.dump(marker_events, f, indent=2)
        print(f"  Markers: {sidecar}")

    gap_summary(stream_name, df_raw, df_grid, mask, TARGET_FS)
    return path, mask


# ============================================================
# Entry point
# ============================================================

print("=" * 60)
print(f"Recording '{NAME_A}' and '{NAME_B}' simultaneously for {DURATION}s")
print(f"Make sure both muselsl streams are running:")
print(f"  muselsl stream --address <MAC_A> --name {NAME_A}")
print(f"  muselsl stream --address <MAC_B> --name {NAME_B}")
print("=" * 60)
input("Press Enter when ready to start recording ...")

df_a, df_b, fs_a, fs_b, marker_events, stop_reason = record_both(NAME_A, NAME_B, DURATION)

print("\nProcessing ...")
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

label_a = NAME_A if NAME_A != NAME_B else f"{NAME_A}_A"
label_b = NAME_B if NAME_A != NAME_B else f"{NAME_B}_B"
path_a, mask_a = process_and_save(df_a, fs_a, label_a, stamp, marker_events)
path_b, mask_b = process_and_save(df_b, fs_b, label_b, stamp, marker_events)

session_summary_path = os.path.join(OUT_DIR, f"{stamp}_session_summary.txt")
with open(session_summary_path, "w") as f:
    f.write(f"Recording summary {datetime.now().isoformat()}\n")
    f.write(f"A: {label_a}  samples={len(df_a)}  fs={fs_a}\n")
    f.write(f"B: {label_b}  samples={len(df_b)}  fs={fs_b}\n")
    if stop_reason:
        f.write(f"dropout: {stop_reason}\n")
    else:
        f.write("dropout: none\n")
print(f"  Session summary: {session_summary_path}")

if not _HAVE_SCIPY:
    print("\n  NOTE: scipy not found — anti-alias filter skipped. "
          "Install it: pip install scipy")

if path_a and path_b:
    print(f"\nRun the analysis pipeline with:")
    print(f"  python pipeline.py {path_a} {path_b} --surrogate 200")
