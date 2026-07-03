"""
stimulus_marker.py — broadcast an LSL marker at stimulus onset.

Run this in a separate terminal BEFORE starting record_both.py.
When you're ready to start the video, press Enter here — the marker
is sent instantly and record_both.py will capture it automatically.

Usage:
    python stimulus_marker.py              # sends "stimulus_start"
    python stimulus_marker.py MyClip       # custom marker name
    python stimulus_marker.py MyClip 3    # send 3 markers (e.g. multiple trials)

Workflow:
    Terminal 1:  python stimulus_marker.py
    Terminal 2:  python record_both.py 120
    -- when ready --
    Press Enter in Terminal 1 at the exact moment you start the video.
"""

import sys
import time
from pylsl import StreamInfo, StreamOutlet, local_clock

MARKER_NAME = sys.argv[1] if len(sys.argv) > 1 else "stimulus_start"
N_MARKERS   = int(sys.argv[2]) if len(sys.argv) > 2 else 1

info = StreamOutlet(
    StreamInfo("StimulusMarkers", "Markers", 1, 0, "string", "stimulus_markers_001")
)

print("=" * 50)
print("Stimulus marker stream is LIVE.")
print(f"  Marker name : '{MARKER_NAME}'")
print(f"  Markers left: {N_MARKERS}")
print("=" * 50)
print("Make sure record_both.py is already running.")
print()

for i in range(N_MARKERS):
    if N_MARKERS > 1:
        prompt = f"Press Enter to send marker {i+1}/{N_MARKERS} ('{MARKER_NAME}') ..."
    else:
        prompt = f"Press Enter when the stimulus starts ..."
    input(prompt)
    t = local_clock()
    info.push_sample([MARKER_NAME], t)
    print(f"  -> Marker '{MARKER_NAME}' sent at LSL time {t:.4f}\n")
    if i < N_MARKERS - 1:
        time.sleep(0.1)

print("Done. You can close this window.")
