"""
visualize_pipeline.py -- stage-by-stage visualization of the hyperscanning
pipeline (pipeline.py), so you can SEE what the data looks like at every step
instead of only the final connectivity matrices.

It reuses pipeline.py's own functions, so what you see here is exactly what the
pipeline computes -- no re-implementation that could drift out of sync.

Stages rendered (one PNG each, numbered in processing order):

    01_raw.png              raw microvolt traces, both subjects, gap segments shaded
    02_filter_before_after  raw vs (band-pass 1-40 Hz + re-reference), zoomed
    03_psd_before_after     power spectrum before vs after filtering, per subject
    04_artifact_mask.png    filtered signal with per-subject bad stretches, the
                            derived blink/saccade channels underneath each
                            subject (so you can see WHAT was rejected), + the
                            JOINT good_mask actually used for connectivity
    05_band_decomposition   the signal split into theta / alpha / beta, both subjects
    06_phase_and_plv.png    instantaneous phase (Hilbert) of one channel pair + the
                            polar phase-difference histogram that PLV summarizes
    07_connectivity.png     final PLV (0..1) and signed circular-corr (-1..1) matrices
    08_overview.png         ALL SEVEN stages condensed onto one page, each panel
                            paired with a written explanation of what it shows and
                            what to look for -- a standalone summary you can hand to
                            someone without also handing them the other 7 PNGs.

Usage:
    python visualize_pipeline.py <subject_a.csv> <subject_b.csv>
    python visualize_pipeline.py a.csv b.csv --band alpha        # band for stages 5/6/8
    python visualize_pipeline.py a.csv b.csv --window 12         # zoom-window length (s)
    python visualize_pipeline.py a.csv b.csv --reference mastoid  # see pipeline.py point 8
    python visualize_pipeline.py a.csv b.csv --out-dir myfigs
    python visualize_pipeline.py a.csv b.csv --no-overview       # skip the 08 poster

Outputs land in out/<timestamp>_viz/ by default.
"""
import argparse
import os
import textwrap
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline as P

# ---------------------------------------------------------------------------
# Palette (design-system aware)
#   channels  -> Okabe-Ito categorical (colorblind-safe, fixed order)
#   subjects  -> two well-separated categorical hues
#   PLV       -> viridis (sequential, one hue light->dark)
#   circ-corr -> RdBu_r (diverging, neutral-gray/white midpoint at 0)
# ---------------------------------------------------------------------------
CH_COLORS = {
    "TP9":  "#0072B2",  # blue
    "AF7":  "#E69F00",  # orange
    "AF8":  "#009E73",  # green
    "TP10": "#CC79A7",  # pink
}
SUBJ_COLORS = {"A": "#0072B2", "B": "#D55E00"}  # blue / vermillion
BAD_COLOR = "#D55E00"
GOOD_COLOR = "#009E73"


def _ch_short(name):
    return name.split("_")[-1]


def _find_window(good_mask, fs, window_s, mode="cleanest"):
    """
    Pick the window_s stretch the zoomed panels display.

    A cleanest-window zoom is not arbitrary, but it is not representative
    either: if only half the recording is usable, the cleanest 10s tells you
    what the data looks like at its best, not what it usually looks like. So
    the mode is selectable and the caller is told how the chosen window
    compares to the recording as a whole:

      cleanest  the cleanest available window (best case)
      typical   the window whose clean fraction is closest to the WHOLE
                recording's clean fraction -- representative by construction
      worst     the dirtiest available window (worst case)

    Returns (start, end, window_clean_fraction).
    """
    win_n = int(round(window_s * fs))
    n = len(good_mask)
    if win_n >= n:
        return 0, n, float(good_mask.mean())
    cum = np.concatenate(([0], np.cumsum(good_mask.astype(int))))
    stride = max(1, win_n // 20)
    starts = np.arange(0, n - win_n + 1, stride)
    fracs = (cum[starts + win_n] - cum[starts]) / float(win_n)
    if mode == "worst":
        idx = int(np.argmin(fracs))
    elif mode == "typical":
        idx = int(np.argmin(np.abs(fracs - good_mask.mean())))
    else:
        idx = int(np.argmax(fracs))
    s0 = int(starts[idx])
    return s0, s0 + win_n, float(fracs[idx])


# ===========================================================================
# STAGE 1 -- raw signal + gaps
# ===========================================================================
def stage_raw(raw_a, raw_b, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, raw, label in zip(axes, [raw_a, raw_b], ["A", "B"]):
        data = raw.get_data() * 1e6
        t = np.arange(data.shape[1]) / raw.info["sfreq"]
        offsets = np.arange(len(raw.ch_names)) * 250.0
        for i, ch in enumerate(raw.ch_names):
            short = _ch_short(ch)
            ax.plot(t, data[i] + offsets[i], lw=0.4,
                    color=CH_COLORS.get(short, "#555"), label=short)
        for ann in raw.annotations:
            if "BAD" in ann["description"]:
                ax.axvspan(ann["onset"], ann["onset"] + ann["duration"],
                           color=BAD_COLOR, alpha=0.18, lw=0)
        ax.set_yticks(offsets)
        ax.set_yticklabels([_ch_short(c) for c in raw.ch_names])
        ax.set_ylabel(f"Subject {label}")
        ax.margins(x=0)
        gap_s = sum(a["duration"] for a in raw.annotations if "BAD" in a["description"])
        dur = data.shape[1] / raw.info["sfreq"]
        ax.set_title(f"Subject {label}   {dur:.0f}s   dropped/gap = {gap_s:.1f}s",
                     fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("STAGE 1  -  Raw signal (uV)   |   shaded = BAD_gap (lost BLE packets)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# STAGE 2 -- filter + reference, before vs after (zoomed)
# ===========================================================================
def stage_filter(raw_a, raw_a_pp, raw_b, raw_b_pp, fs, win, out_path,
                 zoom_note=""):
    s0, s1 = win
    t = np.arange(s0, s1) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, raw, raw_pp, label in zip(axes, [raw_a, raw_b],
                                      [raw_a_pp, raw_b_pp], ["A", "B"]):
        d_raw = raw.get_data() * 1e6
        d_pp = raw_pp.get_data() * 1e6
        offsets = np.arange(len(raw.ch_names)) * 120.0
        for i, ch in enumerate(raw.ch_names):
            short = _ch_short(ch)
            ax.plot(t, d_raw[i, s0:s1] + offsets[i], lw=0.6, color="#BBBBBB",
                    label="raw" if i == 0 else None, zorder=1)
            ax.plot(t, d_pp[i, s0:s1] + offsets[i], lw=0.8,
                    color=CH_COLORS.get(short, "#333"),
                    label="filtered" if i == 0 else None, zorder=2)
        ax.set_yticks(offsets)
        ax.set_yticklabels([_ch_short(c) for c in raw.ch_names])
        ax.set_ylabel(f"Subject {label}")
        ax.margins(x=0)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(zoom_note + "STAGE 2  -  Band-pass 1-40 Hz + re-reference   "
                 "(grey = raw, color = cleaned)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# STAGE 3 -- PSD before vs after
# ===========================================================================
def stage_psd(raw_a, raw_a_pp, raw_b, raw_b_pp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, raw, raw_pp, label in zip(axes, [raw_a, raw_b],
                                      [raw_a_pp, raw_b_pp], ["A", "B"]):
        fmax = min(60.0, raw.info["sfreq"] / 2 * 0.99)
        for r, color, name in [(raw, "#BBBBBB", "raw"),
                               (raw_pp, SUBJ_COLORS[label], "filtered")]:
            psd = r.compute_psd(fmin=0.5, fmax=fmax, verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)
            mean_db = 10 * np.log10(psds.mean(axis=0) + 1e-30)
            ax.plot(freqs, mean_db, color=color, lw=1.6, label=name)
        for f0, f1 in P.FREQ_BANDS.values():
            ax.axvspan(f0, f1, color="#009E73", alpha=0.06, lw=0)
        ax.set_title(f"Subject {label}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB, uV^2/Hz)")
        ax.legend(fontsize=9)
        ax.margins(x=0)
    fig.suptitle("STAGE 3  -  Power spectrum before vs after filtering   "
                 "(shaded = theta/alpha/beta)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# STAGE 4 -- artifact masking
# ===========================================================================
OCULAR_COLORS = {"blink": "#0072B2", "saccade": "#D55E00"}


def _plot_ocular(ax, oc, n, t, bad, fs, thresholds=None):
    """Derived blink/saccade traces under a subject's EEG panel.

    Paired with the rejection shading above it, this answers the question the
    EEG panel alone cannot: whether a rejected stretch was actually ocular,
    or whether the amplitude detector fired on something else.

    thresholds (from pipeline.ocular_thresholds) are drawn as dashed lines at
    +/- half the peak-to-peak threshold around each trace's baseline, so a
    deflection spanning both lines is one the peak-to-peak criterion would
    reject. They are per-participant, so the two subjects' lines sit at
    different heights -- that separation IS the point of estimating them
    per subject.
    """
    traces = [(nm, tr) for nm, tr in (("blink", oc.blink),
                                      ("saccade", oc.saccade)) if tr is not None]
    if not traces:
        ax.text(0.5, 0.5, "no ocular channels (electrodes flagged bad)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.set_yticks([])
        return
    scale = max(np.percentile(np.abs(tr), 99) for _, tr in traces) or 1.0
    step = 4.0 * scale
    for i, (name, trace) in enumerate(traces):
        ax.plot(t, trace[:n] + i * step, lw=0.4, color=OCULAR_COLORS[name])
        thr = (thresholds or {}).get(name)
        if thr:
            for sign in (+1, -1):
                ax.axhline(i * step + sign * thr / 2.0, color=OCULAR_COLORS[name],
                           ls="--", lw=0.7, alpha=0.55)
    _shade_runs(ax, bad[:n], fs, BAD_COLOR, 0.20)
    ax.set_yticks([i * step for i in range(len(traces))])
    ax.set_yticklabels([nm for nm, _ in traces], fontsize=8)
    ax.margins(x=0)
    if thresholds:
        label = "  ".join(f"{nm} p2p thr {thresholds[nm]:.0f} uV"
                          for nm, _ in traces if thresholds.get(nm))
        if label:
            ax.set_title(label, fontsize=8, loc="right", color="#666")


def stage_artifacts(raw_a_pp, bad_a, raw_b_pp, bad_b, good_mask, fs,
                    oc_a, oc_b, out_path, thr_a=None, thr_b=None):
    n = len(good_mask)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4, 3, 1.4, 1]})
    eeg_axes = (axes[0], axes[2])
    ocular_axes = (axes[1], axes[3])
    for ax, raw_pp, bad, label in zip(eeg_axes, [raw_a_pp, raw_b_pp],
                                      [bad_a, bad_b], ["A", "B"]):
        data = raw_pp.get_data()[:, :n] * 1e6
        offsets = np.arange(data.shape[0]) * 120.0
        for i, ch in enumerate(raw_pp.ch_names):
            short = _ch_short(ch)
            ax.plot(t, data[i] + offsets[i], lw=0.4,
                    color=CH_COLORS.get(short, "#555"))
        _shade_runs(ax, bad[:n], fs, BAD_COLOR, 0.20)
        ax.set_yticks(offsets)
        ax.set_yticklabels([_ch_short(c) for c in raw_pp.ch_names])
        ax.set_ylabel(f"Subject {label}")
        ax.margins(x=0)
        ax.set_title(f"Subject {label}: {100 * (~bad[:n]).mean():.1f}% clean "
                     "(shaded = rejected)", fontsize=10, loc="left")
    for ax, oc, bad, label, thr in zip(ocular_axes, [oc_a, oc_b],
                                       [bad_a, bad_b], ["A", "B"],
                                       [thr_a, thr_b]):
        _plot_ocular(ax, oc, n, t, bad, fs, thresholds=thr)
        ax.set_ylabel(f"{label} ocular", fontsize=9)
    # joint good_mask ribbon
    ribbon = good_mask.astype(float)[None, :]
    axes[4].imshow(ribbon, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=(0, t[-1], 0, 1))
    axes[4].set_yticks([])
    axes[4].set_ylabel("joint")
    axes[4].set_xlabel("Time (s)")
    good_s = good_mask.sum() / fs
    axes[4].set_title(f"JOINT good_mask (green = usable in BOTH) -- "
                      f"{good_s:.1f}s / {n / fs:.1f}s = {100 * good_mask.mean():.1f}% "
                      "used for connectivity", fontsize=10, loc="left")
    fig.suptitle("STAGE 4  -  Continuous artifact rejection "
                 "(sliding-window peak-to-peak), with derived ocular channels",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _shade_runs(ax, mask, fs, color, alpha):
    if not mask.any():
        return
    m = mask.astype(int)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(s / fs, e / fs, color=color, alpha=alpha, lw=0)


# ===========================================================================
# STAGE 5 -- band decomposition
# ===========================================================================
def stage_bands(raw_a_pp, raw_b_pp, fs, win, out_path, zoom_note=""):
    s0, s1 = win
    t = np.arange(s0, s1) / fs
    bands = list(P.FREQ_BANDS.items())
    fig, axes = plt.subplots(len(bands), 1, figsize=(14, 8), sharex=True)
    # use one representative channel: AF7 (frontal, present on both)
    for ax, (band_name, band) in zip(axes, bands):
        for raw_pp, label in [(raw_a_pp, "A"), (raw_b_pp, "B")]:
            raw_band = P.prefilter_raw_for_band(raw_pp, band)
            ch_idx = next(i for i, c in enumerate(raw_pp.ch_names)
                          if _ch_short(c) == "AF7")
            d = raw_band.get_data()[ch_idx, s0:s1] * 1e6
            ax.plot(t, d, lw=1.0, color=SUBJ_COLORS[label],
                    label=f"Subject {label}")
        ax.set_ylabel(f"{band_name}\n{band[0]:.0f}-{band[1]:.0f} Hz")
        ax.axhline(0, color="#999", lw=0.5)
        ax.margins(x=0)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(zoom_note + "STAGE 5  -  Signal decomposed into frequency "
                 "bands  (channel AF7)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# STAGE 6 -- phase relationship + PLV
# ===========================================================================
def stage_phase(raw_a_pp, raw_b_pp, good_mask, n_common, fs, band_name, band,
                win, out_path, zoom_note=""):
    raw_a_band = P.prefilter_raw_for_band(raw_a_pp, band)
    raw_b_band = P.prefilter_raw_for_band(raw_b_pp, band)
    analytic_a = P.analytic_signal(raw_a_band, n_samples=n_common)
    analytic_b = P.analytic_signal(raw_b_band, n_samples=n_common)
    plv = P.plv_masked(analytic_a, analytic_b, good_mask)

    # pick the channel pair with the strongest PLV for illustration
    i, j = np.unravel_index(np.argmax(plv), plv.shape)
    ch_a = _ch_short(raw_a_pp.ch_names[i])
    ch_b = _ch_short(raw_b_pp.ch_names[j])

    phase_a = np.angle(analytic_a[i])
    phase_b = np.angle(analytic_b[j])
    dphi = np.angle(np.exp(1j * (phase_a - phase_b)))  # wrapped to [-pi, pi]
    dphi_good = dphi[good_mask]

    # time panels: zoom tighter (<=3s) so individual cycles are legible;
    # the polar histogram below still uses ALL jointly-clean samples.
    s0, s_end = win
    s1 = min(s_end, s0 + int(round(3.0 * fs)))
    t = np.arange(s0, s1) / fs
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1])

    ax_pa = fig.add_subplot(gs[0, 0])
    ax_pa.plot(t, phase_a[s0:s1], lw=1.0, color=SUBJ_COLORS["A"], label=f"A:{ch_a}")
    ax_pa.plot(t, phase_b[s0:s1], lw=1.0, color=SUBJ_COLORS["B"], label=f"B:{ch_b}")
    ax_pa.set_ylabel("Phase (rad)")
    ax_pa.set_title(f"Instantaneous phase ({band_name})", fontsize=10, loc="left")
    ax_pa.legend(loc="upper right", fontsize=8)
    ax_pa.margins(x=0)

    ax_dphi = fig.add_subplot(gs[1, 0], sharex=ax_pa)
    ax_dphi.plot(t, dphi[s0:s1], lw=1.0, color="#333")
    ax_dphi.axhline(0, color="#999", lw=0.5)
    ax_dphi.set_ylabel("A - B phase diff (rad)")
    ax_dphi.set_xlabel("Time (s)")
    ax_dphi.set_ylim(-np.pi, np.pi)
    ax_dphi.set_title("Phase difference (flat = locked)", fontsize=10, loc="left")
    ax_dphi.margins(x=0)

    # polar histogram of phase difference over ALL good samples + PLV vector
    ax_polar = fig.add_subplot(gs[:, 1], projection="polar")
    counts, edges = np.histogram(dphi_good, bins=36, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax_polar.bar(centers, counts, width=width, color="#0072B2", alpha=0.6,
                 edgecolor="white", linewidth=0.5)
    mean_vec = np.mean(np.exp(1j * dphi_good))
    r = np.abs(mean_vec) * counts.max()
    ax_polar.annotate("", xy=(np.angle(mean_vec), r), xytext=(0, 0),
                      arrowprops=dict(color=BAD_COLOR, width=2.5, headwidth=9))
    ax_polar.set_title(f"Phase-diff distribution\nPLV = {plv[i, j]:.3f}\n"
                       f"(arrow length = PLV)", fontsize=10)
    ax_polar.set_yticklabels([])

    fig.suptitle(zoom_note + f"STAGE 6  -  What PLV measures: phase locking of the strongest "
                 f"pair  A:{ch_a} <-> B:{ch_b}  ({band_name})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# STAGE 7 -- final connectivity matrices
# ===========================================================================
def stage_connectivity(raw_a_pp, raw_b_pp, good_mask, n_common, circ_fn, out_path):
    bands = list(P.FREQ_BANDS.items())
    fig, axes = plt.subplots(2, len(bands), figsize=(4.2 * len(bands), 8))
    # from the data, not P.CH_NAMES: the recorded channels and the analysed
    # ones diverge once --analysis-channels restricts the metric set
    ch = [c.split("_")[-1] for c in raw_a_pp.ch_names]
    for col, (band_name, band) in enumerate(bands):
        raw_a_band = P.prefilter_raw_for_band(raw_a_pp, band)
        raw_b_band = P.prefilter_raw_for_band(raw_b_pp, band)
        aa = P.analytic_signal(raw_a_band, n_samples=n_common)
        ab = P.analytic_signal(raw_b_band, n_samples=n_common)
        plv = P.plv_masked(aa, ab, good_mask)
        cc = circ_fn(aa, ab, good_mask)

        ax = axes[0, col]
        im = ax.imshow(plv, cmap="viridis", vmin=0, vmax=1)
        _annotate_matrix(ax, plv, ch, thresh=0.5)
        ax.set_title(f"PLV -- {band_name}\nmean={plv.mean():.3f}", fontsize=10)
        if col == 0:
            fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, label="PLV")

        ax = axes[1, col]
        im2 = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
        _annotate_matrix(ax, cc, ch, thresh=0.6, diverging=True)
        ax.set_title(f"circ-corr -- {band_name}\nmean={cc.mean():.3f}", fontsize=10)
        if col == 0:
            fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.7, label="r (circ)")

    fig.suptitle("STAGE 7  -  Inter-brain connectivity  (rows A:ch, cols B:ch)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _annotate_matrix(ax, mat, ch, thresh, diverging=False):
    ax.set_xticks(range(len(ch)))
    ax.set_yticks(range(len(ch)))
    ax.set_xticklabels([f"B:{c}" for c in ch], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"A:{c}" for c in ch], fontsize=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if diverging:
                # RdBu_r: light background near 0 -> black text; dark poles -> white
                color = "white" if abs(v) > thresh else "black"
            else:
                # viridis: dark at low values -> white text; bright high -> black
                color = "white" if v < thresh else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)


# ===========================================================================
# STAGE 8 -- one-page overview poster (all 7 stages + written explanations)
# ===========================================================================
#
# Every panel above is deliberately full-size and single-purpose so it can be
# inspected closely. This stage instead builds ONE page that condenses all
# seven into small multiples, each paired with a plain-language description of
# what the panel shows, why that step exists in the pipeline, and what pattern
# in the plot would actually mean something -- so the poster is readable on
# its own, without the other seven PNGs open alongside it.

STAGE_DESCRIPTIONS = {
    "raw": (
        "1. RAW SIGNAL",
        "Voltage read straight off each electrode, in microvolts, before any "
        "cleaning. One representative channel (AF7) is shown per subject here "
        "-- see 01_raw.png for all four.\n\n"
        "Shaded red = BAD_gap: stretches where Bluetooth packets were dropped "
        "and no real data exists there (the pipeline fills them with 0 and "
        "annotates them so every later step skips them). On genuinely clean "
        "data, like this recording, gap coverage should sit near 0%. Large or "
        "frequent gaps usually mean a poor Bluetooth connection or the headband "
        "moving out of range."
    ),
    "filter": (
        "2. FILTER + REFERENCE",
        "Raw traces (grey) carry slow drift and a per-channel DC offset that "
        "have nothing to do with brain activity. A 1-40 Hz band-pass removes "
        "both, keeping only the frequency range the connectivity analysis "
        "actually tests (theta/alpha/beta all fall inside it). Average-"
        "referencing then re-centers each subject's own channels against each "
        "other, canceling out shared electrical noise picked up by all four "
        "electrodes at once.\n\n"
        "The colored trace is what every downstream stage operates on. Compare "
        "it to the grey raw trace here: it should look smoother (drift gone) "
        "but keep the same fast wiggles -- if the colored trace looks flat or "
        "unrecognizable, the filter cutoff may be too aggressive for this data."
    ),
    "psd": (
        "3. POWER SPECTRUM (PSD)",
        "The same recording viewed as power vs. frequency instead of voltage "
        "vs. time. Before filtering (grey) power is dominated by low-frequency "
        "drift on the left edge and a sharp spike at 50 Hz (mains electrical "
        "noise). After filtering (color), both are suppressed and the curve "
        "rolls off past 40 Hz.\n\n"
        "The shaded bands mark theta/alpha/beta -- where the pipeline actually "
        "measures inter-brain coupling. A healthy EEG spectrum slopes downward "
        "(1/f-like) with a soft alpha bump around 10 Hz; a flat or noisy "
        "spectrum in-band suggests electrode contact problems rather than "
        "brain signal."
    ),
    "artifact": (
        "4. ARTIFACT MASKING",
        "A short window (0.5s default) slides across each subject's cleaned "
        "signal; any window where a channel's peak-to-peak amplitude exceeds "
        "the threshold (500 uV default) gets marked bad -- typically an eye "
        "blink, jaw clench, or the headband shifting. One channel (AF7) is "
        "shown per subject; full detail in 04_artifact_mask.png.\n\n"
        "The bottom green/red ribbon is the INTERSECTION of both subjects' bad "
        "masks -- only timepoints clean in BOTH brains at once. That matters "
        "because PLV and circular correlation need a paired sample from each "
        "subject at every instant; a moment that's clean in A but contaminated "
        "in B still has to be dropped from both."
    ),
    "bands": (
        "5. FREQUENCY-BAND DECOMPOSITION",
        "The cleaned signal (channel AF7), split by band-pass filtering into "
        "the three rhythms the pipeline tests separately: theta (4-8 Hz, "
        "often linked to attention/memory), alpha (8-13 Hz, relaxed/idling "
        "visual cortex), and beta (13-30 Hz, active processing/motor "
        "engagement).\n\n"
        "Connectivity is computed independently per band because coupling "
        "that shows up in one rhythm can be completely absent in another -- "
        "collapsing them into one broadband signal would wash that out. "
        "Watch for envelope bursts (amplitude swelling and shrinking): those "
        "are the events that matter for phase estimation."
    ),
    "phase": (
        "6. INSTANTANEOUS PHASE & WHAT PLV MEASURES",
        "The Hilbert transform turns each band-passed channel into an "
        "instantaneous PHASE -- where in its oscillation cycle it currently "
        "sits, from -pi to +pi. The left panel is subject A minus subject B's "
        "phase, for the channel pair with the single strongest measured "
        "coupling; a flat, constant line would mean the two brains' "
        "oscillations stay locked in step.\n\n"
        "The polar plot on the right collapses that same phase difference "
        "across the WHOLE recording into a histogram. A tight cluster pointing "
        "in one direction means strong, consistent phase-locking (long arrow = "
        "high PLV, PLV=1 is a single point). A histogram spread evenly around "
        "the circle, like this one, means the phase relationship is random -- "
        "PLV near 0, no detectable locking."
    ),
    "connectivity": (
        "7. INTER-BRAIN CONNECTIVITY (the result)",
        "For every electrode pair (one from subject A, one from subject B) "
        "and every frequency band: PLV (Phase-Locking Value, 0 to 1, magnitude "
        "only -- shown here) and circular correlation (-1 to 1, signed, see "
        "07_connectivity.png) summarize how phase-locked that pair was across "
        "all jointly-clean data.\n\n"
        "This is the pipeline's main output, but a raw number here is NOT "
        "evidence of real coupling by itself -- both metrics are biased "
        "upward by short recordings and can be inflated by two brains "
        "independently locking onto a shared external rhythm (e.g. a "
        "flickering screen) rather than to each other. Run pipeline.py with "
        "--surrogate and/or --pool-dir to get the null distribution these "
        "numbers need to be compared against before calling anything "
        "'significant'."
    ),
}


def _wrap_body(text, width=44):
    return "\n\n".join(textwrap.fill(p, width=width) for p in text.split("\n\n"))


def _desc_panel(fig, rect, key):
    title, body = STAGE_DESCRIPTIONS[key]
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.text(0, 1.0, title, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", color="#111111")
    ax.text(0, 0.90, _wrap_body(body), transform=ax.transAxes, fontsize=8.4,
            va="top", ha="left", linespacing=1.55, color="#333333",
            family="sans-serif")


def _split_rect(rect, n, gap=0.012):
    left, bottom, width, height = rect
    w = (width - gap * (n - 1)) / n
    return [(left + i * (w + gap), bottom, w, height) for i in range(n)]


def stage_overview(csv_a, csv_b, raw_a, raw_a_pp, raw_b, raw_b_pp,
                    bad_a, bad_b, good_mask, fs, n_common, zoom,
                    band_name, band, circ_fn, out_path,
                    raw_a_ana=None, raw_b_ana=None):
    # raw_*_ana are the analysis-channel-restricted raws used for the
    # connectivity row only; every other row shows the full montage. Default
    # to the unrestricted raws so the signature stays usable on its own.
    raw_a_ana = raw_a_ana if raw_a_ana is not None else raw_a_pp
    raw_b_ana = raw_b_ana if raw_b_ana is not None else raw_b_pp
    fig = plt.figure(figsize=(20, 36))

    dur = raw_a.get_data().shape[1] / fs
    clean_pct = 100 * good_mask.mean()
    meta = (
        f"A: {os.path.basename(csv_a)}    B: {os.path.basename(csv_b)}    "
        f"duration: {dur:.0f}s    fs: {fs:.0f} Hz    "
        f"jointly-clean: {clean_pct:.1f}%    "
        f"zoomed panels use band: {band_name} ({band[0]:.0f}-{band[1]:.0f} Hz)    "
        f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    fig.text(0.02, 0.992, "HYPERSCANNING PIPELINE -- STAGE-BY-STAGE OVERVIEW",
             fontsize=20, fontweight="bold", va="top", ha="left")
    fig.text(0.02, 0.978, meta, fontsize=10, va="top", ha="left", color="#444")

    n_rows = 7
    top, bottom_lim = 0.960, 0.015
    gap = 0.028  # generous: rows below draw a title ABOVE their axes and rows
                 # above draw an xlabel BELOW theirs -- both encroach into this
                 # gap from opposite sides, so it has to fit both.
    row_h = (top - bottom_lim - gap * (n_rows - 1)) / n_rows
    text_w, plot_left, plot_w = 0.205, 0.245, 0.735
    title_margin = 0.014  # extra headroom carved from the TOP of a row's plot
                          # rect when that row draws matplotlib titles (which
                          # render above the axes box, i.e. outside the rect)

    def row_rects(i, has_titles=False):
        row_top = top - i * (row_h + gap)
        row_bot = row_top - row_h
        h = row_h - title_margin if has_titles else row_h
        return (0.02, row_bot, text_w, row_h), (plot_left, row_bot, plot_w, h)

    s0, s_end = zoom
    s1 = min(s_end, s0 + int(round(3.0 * fs)))
    t_zoom = np.arange(s0, s1) / fs
    ch = "AF7"
    ch_idx_a = next(i for i, c in enumerate(raw_a.ch_names) if _ch_short(c) == ch)
    ch_idx_b = next(i for i, c in enumerate(raw_b.ch_names) if _ch_short(c) == ch)

    # ---- row 0: raw ----------------------------------------------------
    text_rect, plot_rect = row_rects(0)
    _desc_panel(fig, text_rect, "raw")
    ax = fig.add_axes(plot_rect)
    for raw, idx, label in [(raw_a, ch_idx_a, "A"), (raw_b, ch_idx_b, "B")]:
        data = raw.get_data()[idx] * 1e6
        tt = np.arange(len(data)) / raw.info["sfreq"]
        ax.plot(tt, data, lw=0.35, color=SUBJ_COLORS[label], label=f"{label}:{ch}")
        for ann in raw.annotations:
            if "BAD" in ann["description"]:
                ax.axvspan(ann["onset"], ann["onset"] + ann["duration"],
                           color=BAD_COLOR, alpha=0.15, lw=0)
    ax.set_ylabel("uV")
    ax.set_xlabel("Time (s)")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- row 1: filter before/after ------------------------------------
    text_rect, plot_rect = row_rects(1)
    _desc_panel(fig, text_rect, "filter")
    ax = fig.add_axes(plot_rect)
    for raw, raw_pp, idx, label, off in [
        (raw_a, raw_a_pp, ch_idx_a, "A", 120.0), (raw_b, raw_b_pp, ch_idx_b, "B", 0.0)
    ]:
        d_raw = raw.get_data()[idx, s0:s1] * 1e6
        d_pp = raw_pp.get_data()[idx, s0:s1] * 1e6
        ax.plot(t_zoom, d_raw + off, lw=0.7, color="#BBBBBB",
                label="raw" if label == "A" else None)
        ax.plot(t_zoom, d_pp + off, lw=1.0, color=SUBJ_COLORS[label],
                label=f"{label}:{ch} filtered")
    ax.set_ylabel("uV (offset)")
    ax.set_xlabel("Time (s)")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # ---- row 2: PSD before/after ----------------------------------------
    text_rect, plot_rect = row_rects(2, has_titles=True)
    _desc_panel(fig, text_rect, "psd")
    sub_rects = _split_rect(plot_rect, 2)
    for rect, raw, raw_pp, label in [
        (sub_rects[0], raw_a, raw_a_pp, "A"), (sub_rects[1], raw_b, raw_b_pp, "B")
    ]:
        ax = fig.add_axes(rect)
        fmax = min(60.0, raw.info["sfreq"] / 2 * 0.99)
        for r, color, name in [(raw, "#BBBBBB", "raw"), (raw_pp, SUBJ_COLORS[label], "filtered")]:
            psd = r.compute_psd(fmin=0.5, fmax=fmax, verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)
            ax.plot(freqs, 10 * np.log10(psds.mean(axis=0) + 1e-30), color=color, lw=1.3, label=name)
        for f0, f1 in P.FREQ_BANDS.values():
            ax.axvspan(f0, f1, color="#009E73", alpha=0.06, lw=0)
        ax.set_title(f"Subject {label}", fontsize=9)
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB" if label == "A" else "")
        ax.legend(fontsize=7)
        ax.margins(x=0)

    # ---- row 3: artifact mask -------------------------------------------
    text_rect, plot_rect = row_rects(3)
    _desc_panel(fig, text_rect, "artifact")
    sub_top = (plot_rect[0], plot_rect[1] + plot_rect[3] * 0.28, plot_rect[2], plot_rect[3] * 0.72)
    sub_bot = (plot_rect[0], plot_rect[1], plot_rect[2], plot_rect[3] * 0.20)
    ax = fig.add_axes(sub_top)
    n = len(good_mask)
    t_full = np.arange(n) / fs
    for raw_pp, idx, bad, label in [(raw_a_pp, ch_idx_a, bad_a, "A"), (raw_b_pp, ch_idx_b, bad_b, "B")]:
        data = raw_pp.get_data()[idx, :n] * 1e6
        ax.plot(t_full, data, lw=0.3, color=SUBJ_COLORS[label], label=f"{label}:{ch}")
        _shade_runs(ax, bad[:n], fs, BAD_COLOR, 0.12)
    ax.set_ylabel("uV")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xticklabels([])
    ax2 = fig.add_axes(sub_bot)
    ax2.imshow(good_mask.astype(float)[None, :], aspect="auto", cmap="RdYlGn",
              vmin=0, vmax=1, extent=(0, t_full[-1], 0, 1))
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("joint", fontsize=8)

    # ---- row 4: band decomposition ---------------------------------------
    text_rect, plot_rect = row_rects(4, has_titles=True)
    _desc_panel(fig, text_rect, "bands")
    sub_rects = _split_rect(plot_rect, 3)
    for rect, (bname, bband) in zip(sub_rects, P.FREQ_BANDS.items()):
        ax = fig.add_axes(rect)
        for raw_pp, idx, label in [(raw_a_pp, ch_idx_a, "A"), (raw_b_pp, ch_idx_b, "B")]:
            raw_band = P.prefilter_raw_for_band(raw_pp, bband)
            d = raw_band.get_data()[idx, s0:s1] * 1e6
            ax.plot(t_zoom, d, lw=0.9, color=SUBJ_COLORS[label], label=f"Sub {label}")
        ax.axhline(0, color="#999", lw=0.4)
        ax.set_title(f"{bname} ({bband[0]:.0f}-{bband[1]:.0f} Hz)", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.margins(x=0)
        if rect is sub_rects[0]:
            ax.legend(fontsize=7, loc="upper right")

    # ---- row 5: phase & PLV -----------------------------------------------
    text_rect, plot_rect = row_rects(5, has_titles=True)
    _desc_panel(fig, text_rect, "phase")
    left_rect, right_rect = _split_rect(plot_rect, 2)
    raw_a_band = P.prefilter_raw_for_band(raw_a_pp, band)
    raw_b_band = P.prefilter_raw_for_band(raw_b_pp, band)
    analytic_a = P.analytic_signal(raw_a_band, n_samples=n_common)
    analytic_b = P.analytic_signal(raw_b_band, n_samples=n_common)
    plv = P.plv_masked(analytic_a, analytic_b, good_mask)
    i, j = np.unravel_index(np.argmax(plv), plv.shape)
    phase_a = np.angle(analytic_a[i])
    phase_b = np.angle(analytic_b[j])
    dphi = np.angle(np.exp(1j * (phase_a - phase_b)))
    dphi_good = dphi[good_mask]

    ax = fig.add_axes(left_rect)
    ax.plot(t_zoom, dphi[s0:s1], lw=1.0, color="#333")
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_ylabel("A-B phase diff (rad)")
    ax.set_xlabel("Time (s)")
    ax.margins(x=0)
    ax.set_title(f"strongest pair: A:{_ch_short(raw_a_pp.ch_names[i])} <-> "
                f"B:{_ch_short(raw_b_pp.ch_names[j])}", fontsize=9)

    ax_polar = fig.add_axes(right_rect, projection="polar")
    counts, edges = np.histogram(dphi_good, bins=36, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax_polar.bar(centers, counts, width=width, color="#0072B2", alpha=0.6,
                edgecolor="white", linewidth=0.4)
    mean_vec = np.mean(np.exp(1j * dphi_good))
    ax_polar.annotate("", xy=(np.angle(mean_vec), np.abs(mean_vec) * counts.max()),
                      xytext=(0, 0),
                      arrowprops=dict(color=BAD_COLOR, width=2, headwidth=7))
    ax_polar.set_yticklabels([])
    ax_polar.set_title(f"PLV = {plv[i, j]:.3f}", fontsize=9)

    # ---- row 6: connectivity -----------------------------------------------
    text_rect, plot_rect = row_rects(6, has_titles=True)
    _desc_panel(fig, text_rect, "connectivity")
    sub_rects = _split_rect(plot_rect, 3)
    ch_names = [c.split("_")[-1] for c in raw_a_ana.ch_names]
    for rect, (bname, bband) in zip(sub_rects, P.FREQ_BANDS.items()):
        ax = fig.add_axes(rect)
        raw_a_b = P.prefilter_raw_for_band(raw_a_ana, bband)
        raw_b_b = P.prefilter_raw_for_band(raw_b_ana, bband)
        aa = P.analytic_signal(raw_a_b, n_samples=n_common)
        ab = P.analytic_signal(raw_b_b, n_samples=n_common)
        plv_b = P.plv_masked(aa, ab, good_mask)
        im = ax.imshow(plv_b, cmap="viridis", vmin=0, vmax=1)
        _annotate_matrix(ax, plv_b, ch_names, thresh=0.5)
        ax.set_title(f"PLV -- {bname}  (mean={plv_b.mean():.3f})", fontsize=9)

    fig.savefig(out_path, dpi=115)
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--band", default="alpha", choices=list(P.FREQ_BANDS.keys()),
                    help="band used for the band-decomposition zoom (stage 5) and "
                         "the phase/PLV illustration (stage 6). default: alpha")
    ap.add_argument("--window", type=float, default=10.0,
                    help="length (s) of the zoomed time window in stages 2/5/6 "
                         "(default 10)")
    ap.add_argument("--zoom-mode", choices=["cleanest", "typical", "worst"],
                    default="typical",
                    help="which stretch the zoomed panels (stages 2/5/6) "
                         "show. 'cleanest' is the best-case window and can "
                         "flatter a mostly-dirty recording; 'typical' "
                         "(default) picks the window whose clean fraction "
                         "matches the whole recording's, so the zoom is "
                         "representative; 'worst' shows the worst case. "
                         "Every figure is labelled with both fractions.")
    ap.add_argument("--analysis-channels", choices=["auto", "all", "frontal"],
                    default="auto",
                    help="channels the connectivity panels (stages 6/7/8) "
                         "use, mirroring pipeline.py's flag of the same name "
                         "(default auto: all four under --reference average, "
                         "AF7/AF8 under --reference mastoid). The earlier "
                         "stages always show the full montage.")
    ap.add_argument("--ocular-k", type=float, default=5.0,
                    help="k for the per-participant ocular thresholds drawn "
                         "in stage 4 (threshold = median + k*MAD; default 5). "
                         "Display only -- rejection here still uses the EEG "
                         "amplitude criterion.")
    ap.add_argument("--reference", choices=["average", "mastoid"],
                    default="average",
                    help="re-reference scheme, passed through to "
                         "pipeline.preprocess (default average). See point 8 "
                         "of pipeline.py's docstring.")
    ap.add_argument("--artifact-threshold", type=float, default=500.0)
    ap.add_argument("--artifact-window", type=float, default=0.5)
    ap.add_argument("--artifact-step", type=float, default=0.1)
    ap.add_argument("--artifact-pad", type=float, default=0.3)
    ap.add_argument("--circ-corr-method", choices=["adjusted", "classic"],
                    default="adjusted")
    ap.add_argument("--no-overview", dest="overview", action="store_false",
                    default=True,
                    help="skip 08_overview.png (the condensed, annotated "
                         "single-page poster of all 7 stages; on by default)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S") + "_viz")
    os.makedirs(out_dir, exist_ok=True)

    circ_fn = (P.circ_corr_adjusted_masked if args.circ_corr_method == "adjusted"
               else P.circ_corr_masked)

    print("=" * 60)
    print("LOADING")
    print("=" * 60)
    onset_a = P.load_stimulus_onset(args.csv_a)
    onset_b = P.load_stimulus_onset(args.csv_b)
    if onset_a is None or onset_b is None:
        onset_a = onset_b = None
    raw_a, fs = P.load_csv_to_raw(args.csv_a, "A", onset_s=onset_a)
    raw_b, fs_b = P.load_csv_to_raw(args.csv_b, "B", onset_s=onset_b)
    if fs != fs_b:
        print(f"  WARNING: sampling rates differ ({fs} vs {fs_b})")

    print("\nPREPROCESSING")
    nyq = fs / 2
    h_freq = min(40.0, nyq * 0.95)
    raw_a_pp = P.preprocess(raw_a, h_freq=h_freq, subject_label="A",
                            reference=args.reference)
    raw_b_pp = P.preprocess(raw_b, h_freq=h_freq, subject_label="B",
                            reference=args.reference)
    # derived ocular channels, from the UNFILTERED raw (pipeline.py point 9)
    oc_a = P.make_ocular_channels(raw_a, subject_label="A",
                                  bads=raw_a_pp.info["bads"])
    oc_b = P.make_ocular_channels(raw_b, subject_label="B",
                                  bads=raw_b_pp.info["bads"])

    print("\nARTIFACT DETECTION")
    bad_a = P.continuous_bad_mask(raw_a_pp, window_s=args.artifact_window,
                                  step_s=args.artifact_step,
                                  threshold_uv=args.artifact_threshold,
                                  pad_s=args.artifact_pad)
    bad_b = P.continuous_bad_mask(raw_b_pp, window_s=args.artifact_window,
                                  step_s=args.artifact_step,
                                  threshold_uv=args.artifact_threshold,
                                  pad_s=args.artifact_pad)
    n_common = min(len(bad_a), len(bad_b))
    bad_a, bad_b = bad_a[:n_common], bad_b[:n_common]
    good_mask = ~(bad_a | bad_b)
    print(f"  A {100 * (~bad_a).mean():.1f}% clean   "
          f"B {100 * (~bad_b).mean():.1f}% clean   "
          f"joint {100 * good_mask.mean():.1f}%")

    if good_mask.sum() == 0:
        print("  No jointly-clean samples -- nothing to visualize. "
              "Loosen --artifact-threshold.")
        return

    # Restrict to the analysis channels for the connectivity stages, after
    # the masks are built from the full montage -- same ordering as
    # pipeline.main(). Stages 1-4 keep using the unrestricted raws, which is
    # what you want: they are there to show what was recorded and rejected.
    analysis_ch = P.resolve_analysis_channels(args.analysis_channels,
                                              args.reference)
    P.set_analysis_channels(analysis_ch)
    raw_a_ana = P.restrict_to_analysis(raw_a_pp, analysis_ch, "A")
    raw_b_ana = P.restrict_to_analysis(raw_b_pp, analysis_ch, "B")
    if len(analysis_ch) < len(P.CH_NAMES):
        print(f"  connectivity panels use {analysis_ch} "
              f"({len(analysis_ch) ** 2} pairs)")

    # zoom window, and an honest label saying how representative it is
    z0, z1, zfrac = _find_window(good_mask, fs, args.window, args.zoom_mode)
    zoom = (z0, z1)
    overall = float(good_mask.mean())
    zoom_note = (f"[zoom: {args.zoom_mode} {args.window:g}s window at "
                 f"t={z0 / fs:.0f}s, {100 * zfrac:.0f}% clean vs "
                 f"{100 * overall:.0f}% for the whole recording]\n")
    print(f"  zoom window ({args.zoom_mode}): t={z0 / fs:.1f}-{z1 / fs:.1f}s, "
          f"{100 * zfrac:.1f}% clean (recording overall {100 * overall:.1f}%)")
    band = P.FREQ_BANDS[args.band]

    print("\nRENDERING FIGURES")
    stage_raw(raw_a, raw_b, os.path.join(out_dir, "01_raw.png"))
    print("  01_raw.png")
    stage_filter(raw_a, raw_a_pp, raw_b, raw_b_pp, fs, zoom,
                 os.path.join(out_dir, "02_filter_before_after.png"),
                 zoom_note=zoom_note)
    print("  02_filter_before_after.png")
    stage_psd(raw_a, raw_a_pp, raw_b, raw_b_pp,
              os.path.join(out_dir, "03_psd_before_after.png"))
    print("  03_psd_before_after.png")
    thr_a = P.ocular_thresholds(oc_a, window_s=args.artifact_window,
                                step_s=args.artifact_step, k=args.ocular_k)
    thr_b = P.ocular_thresholds(oc_b, window_s=args.artifact_window,
                                step_s=args.artifact_step, k=args.ocular_k)
    stage_artifacts(raw_a_pp, bad_a, raw_b_pp, bad_b, good_mask, fs,
                    oc_a, oc_b,
                    os.path.join(out_dir, "04_artifact_mask.png"),
                    thr_a=thr_a, thr_b=thr_b)
    print("  04_artifact_mask.png")
    stage_bands(raw_a_pp, raw_b_pp, fs, zoom,
                os.path.join(out_dir, "05_band_decomposition.png"),
                zoom_note=zoom_note)
    print("  05_band_decomposition.png")
    stage_phase(raw_a_pp, raw_b_pp, good_mask, n_common, fs, args.band, band,
                zoom, os.path.join(out_dir, "06_phase_and_plv.png"),
                zoom_note=zoom_note)
    print("  06_phase_and_plv.png")
    stage_connectivity(raw_a_ana, raw_b_ana, good_mask, n_common, circ_fn,
                       os.path.join(out_dir, "07_connectivity.png"))
    print("  07_connectivity.png")

    if args.overview:
        stage_overview(args.csv_a, args.csv_b, raw_a, raw_a_pp, raw_b, raw_b_pp,
                       bad_a, bad_b, good_mask, fs, n_common, zoom,
                       args.band, band, circ_fn,
                       os.path.join(out_dir, "08_overview.png"),
                       raw_a_ana=raw_a_ana, raw_b_ana=raw_b_ana)
        print("  08_overview.png")

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
