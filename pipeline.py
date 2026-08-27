"""
pipeline.py — hyperscanning pipeline for two-Muse recordings.

Takes the two CSVs produced by record_both.py and runs the full chain:

    raw CSV   ->   MNE Raw
                   |
                   v
              annotations from is_gap column   (BAD_gap, MNE will skip these)
                   |
                   v
              bandpass filter 1-40 Hz, re-reference (average or linked
              mastoid -- see point 8, --reference)
                   |
                   v
              continuous sliding-window artifact detection (default, see
              point 6 below) -- OR fixed-length epochs (--legacy-epochs,
              see point 5)
                   |
                   v
              PLV + circular correlation per frequency band (theta, alpha, beta)
                   |
                   v
              inter-brain connectivity matrices + plots

Outputs land in `out/<timestamp>/`:
    plv_<band>.npy              raw 4x4 inter-brain connectivity matrices per band
    plv_interbrain_<band>.png   4x4 inter-brain matrices (the main result)
    raw_with_gaps.png           sanity check: signal + gap markers
    psd.png                     power spectrum per subject (QC)
    summary.txt                 numerical summary

Usage:
    python pipeline.py <subject_a.csv> <subject_b.csv>
    python pipeline.py a.csv b.csv                           # default: continuous sliding-window artifact rejection (see point 6 below)
    python pipeline.py a.csv b.csv --legacy-epochs           # old fixed-epoch pipeline (see point 5), for comparison
    python pipeline.py a.csv b.csv --bands alpha             # one band only
    python pipeline.py a.csv b.csv --surrogate 100           # within-dyad null (epoch shuffle)
    python pipeline.py a.csv b.csv --pool-dir recordings/other_dyads/   # cross-dyad null (pseudo-pairs)
    python pipeline.py a.csv b.csv --stim-hz 6.0 --pool-dir recordings/other_dyads/
        # positive control: both subjects watched the SAME 6 Hz flicker.
        # Raw magnitude here is expected to be high but AMBIGUOUS -- see
        # point 4 below. Needs --pool-dir (not just --surrogate) to mean
        # anything, since a constant stimulus barely responds to epoch shuffling.
    python pipeline.py a.csv b.csv --tag-hz-a 6.0 --tag-hz-b 8.0 --surrogate 100 --pool-dir recordings/other_dyads/
        # frequency-tagging: A watched 6 Hz, B watched 8 Hz (two monitors,
        # see make_checkerboard.py --pos). Unambiguous cross-brain test --
        # see point 4 below.

-----------------------------------------------------------------------------
CHANGES IN THIS VERSION (see conversation notes / internship writeup)
-----------------------------------------------------------------------------
1. PSEUDO-PAIR (CROSS-DYAD) SURROGATE, in addition to within-dyad epoch shuffle.
   Within-dyad epoch shuffling only destroys coupling that varies from epoch to
   epoch. A continuous, unchanging stimulus (e.g. a 6 Hz flicker running the
   whole recording) produces a near-constant per-epoch phase relationship, so
   shuffling epoch order barely changes the metric -- the "surrogate" ends up
   just as high as the real value, and you get a non-significant result even
   though the raw PLV/circ-corr is high. That is NOT a null result about
   coupling; it means the within-dyad shuffle test is the wrong tool for a
   stimulus-locked, stationary signal like SSVEP.
   Pseudo-pairing (comparing subject A against OTHER, non-partner people who
   watched the same or a comparable stimulus) tests the thing you actually
   care about: is this dyad's coupling any different from what you'd get by
   chance-pairing two people who were never in the same session together?
   This is the standard hyperscanning validity check in the literature
   (e.g. Burgess 2013 "cautionary note"; pseudo-pair designs used throughout
   the hyperscanning field). Use --pool-dir to point at a folder of OTHER
   subjects' single-person recordings (any CSVs with the same column format)
   and the pipeline will build a cross-dyad null distribution from them.

2. FDR CORRECTION instead of raw per-pair p<0.05 counts.
   With 16 channel pairs tested per band (4x4; fewer when
   --analysis-channels restricts the set, see point 11) at uncorrected p<0.05, you'd expect
   ~0.8 false positives per band by chance alone even under pure noise. The
   old "1/16" and "2/16" significant-pair counts were not meaningfully
   different from chance. This version applies Benjamini-Hochberg FDR
   correction across the 16 pairs (per band) before counting "significant"
   pairs, and reports both raw and corrected counts so you can see the
   difference.

3. OPTIONAL PRE-FILTERING before epoching (--prefilter).
   Previously, each 2-second epoch was independently narrowband-filtered
   inside HyPyP's compute_freq_bands (filter_signal=True). For a narrow
   band like stim_6hz (1 Hz wide) a 4th-order IIR filter's transient/edge
   response can occupy a large fraction of a 2-second window, which can
   inflate apparent phase consistency across epochs independent of any real
   coupling. --prefilter instead band-passes the CONTINUOUS raw signal
   BEFORE epoching (so filter transients only occur once, at the start/end
   of the whole recording, not at every epoch boundary), then epochs the
   already-filtered signal and passes filter_signal=False into HyPyP.
   This is now the default; use --no-prefilter to restore the old
   per-epoch-filtering behavior for comparison.

4. FREQUENCY-TAGGING support (--tag-hz-a / --tag-hz-b).
   Even a validated --stim-hz positive control (both subjects watching the
   SAME flicker) is fundamentally ambiguous: elevated inter-brain PLV/circ-
   corr there is exactly what you'd see BOTH if the two people are really
   coupled to each other AND if they're simply two independent visual
   systems separately locking onto one shared external clock. A shuffle
   test can't tell these apart for a constant, non-varying stimulus (see
   point 1), and even a --pool-dir pseudo-pair test only bounds the size of
   the shared-clock effect -- it doesn't eliminate the ambiguity in
   principle.
   Frequency-tagging removes the ambiguity by construction: give subject A
   and subject B DIFFERENT reversal rates (needs two monitors -- see --pos
   in make_checkerboard.py). Subject B is never driven at A's frequency, so
   any inter-brain coupling at A's tag frequency cannot be explained by "both
   locked to the same clock" -- there was no clock at that frequency for B.
   This is the strongest single test available and should supersede --stim-hz
   once you have a two-monitor setup; --stim-hz remains useful only as a
   cheap single-monitor sanity check that the measurement pipeline is
   sensitive at all.

5. DEFAULT EPOCH LENGTH increased from 2s to 30s (--epoch-len), overlap
   from 1s to 0s (--epoch-overlap).
   Short epochs inflate PLV/circ-corr: on one real dyad recording, mean
   theta PLV dropped from 0.286 (2s epochs) to 0.131 (10s) to 0.075 (30s)
   to 0.022 (one ~180s window covering the whole recording) -- a ~10x
   difference in the SAME data depending only on epoch length, exactly the
   bias reported by Zimmermann et al. (2024) and Cassioli et al. (2025),
   who estimate ~55s (+20% for artifact loss) of clean data is needed for
   adequately powered detection of a moderate effect at 256 Hz. 30s was
   chosen as the new default over 55s because at 150uV amplitude
   rejection, whole-epoch rejection makes very long epochs fragile (a
   single blink anywhere in the window drops the whole window); 30s
   still yields multiple epochs per session at typical recording lengths.
   Pass --epoch-len 2.0 --epoch-overlap 1.0 to restore the old behavior.

6. FIXED EPOCHS REPLACED BY CONTINUOUS SLIDING-WINDOW ARTIFACT REJECTION,
   as the new default (--legacy-epochs restores the point-5 fixed-epoch
   pipeline).
   Suggested by Evan (project supervisor, 2026-07-29): even the 30s fixed
   epochs from point 5 have the same flaw at a different scale -- a single
   1s blink inside a 30s window throws out the OTHER 29s of perfectly good
   data too, since MNE's epoch rejection is all-or-nothing per epoch.
   Instead, continuous_bad_mask() slides a short window (500ms, 100ms step
   by default -- see --artifact-window/--artifact-step/--artifact-threshold)
   across each subject's continuous signal and marks only the actual bad
   stretch, in BOTH subjects, at the matching timepoints (PLV/circ-corr need
   paired samples from both brains at the same instant). The Hilbert
   transform (instantaneous phase) is computed ONCE on the full continuous
   band-passed recording -- so there's no splicing discontinuity anywhere --
   and only afterward are the jointly-clean samples selected for the PLV /
   circular-correlation sum. See plv_masked()/circ_corr_masked() and the
   "3a2. CONTINUOUS (MASKED) CONNECTIVITY" section below; this bypasses
   HyPyP's epoch-based API, which has no way to mask out individual
   timepoints within an epoch.

7b. NULL-DISTRIBUTION SAMPLE-SIZE LOGGING + LENGTH MATCHING (--match-null-length).
   Diagnosed 2026-08-07: a real dyad's joint-clean window can be MUCH longer
   than what individual pseudo-pair (--pool-dir) draws end up with. E.g. one
   run had a real dyad joint good_mask of 372.5s (95,372 samples @ 256 Hz),
   but pseudo_pair_continuous() intersects the real subject's bad-mask with
   EACH POOL MEMBER'S OWN bad-mask after a random circular shift -- if a pool
   member's clean stretches don't line up well with the target's after
   shifting, that intersection can leave far less usable overlap per draw.
   Per Zimmermann et al. (2024), SHORTER windows systematically INFLATE PLV
   for weakly/uncoupled signals -- so a null built from short pool draws can
   end up biased upward relative to a long, well-converged real value, making
   real coupling look "below baseline" for a reason that has nothing to do
   with actual inter-brain coupling: an N mismatch between the real value and
   its null.
   circular_shift_surrogates_continuous() and pseudo_pair_continuous() now
   both return the per-draw sample count (Ns) alongside the null values, and
   this is logged to console + summary.txt regardless of anything else, so
   an N mismatch like the one above is visible without needing to add ad hoc
   print statements.
   --match-null-length (default: on) additionally computes a LENGTH-MATCHED
   comparison: a target sample count is chosen as the smaller of (a) the
   real dyad's own good_mask size and (b) a robust (10th percentile) size
   across the null draws, floored at --min-null-seconds. Both the null draws
   AND a matched "observed" value (averaged over several random subsamples
   of the real dyad's good_mask down to that same target size) are then
   recomputed at that common N, so the p-value comparison is apples-to-
   apples. The ORIGINAL full-length real PLV/circ-r (computed on ALL
   available clean data, still the best point estimate) is still reported
   unchanged as the headline number -- only the null-comparison/p-value uses
   the length-matched version. Use --no-match-null-length to restore the old
   (potentially N-mismatched) behaviour.

7. ADJUSTED CIRCULAR CORRELATION now implemented in the continuous path too.
   Point 6 originally shipped with a caveat: the continuous path fell back
   to the classic Fisher & Lee (1983) circular correlation because the
   bias-adjusted CCorradj formula from Zimmermann et al. (2024) "wasn't
   confidently reproducible from the secondary source available." That
   formula is now confirmed against a known, tested, open-source
   implementation: pingouin.circ_corrcc(correction_uniform=True) (Vallat,
   2018), whose source directly implements Jammalamadaka & Sengupta (2001,
   p.177):

       r_minus = |sum_t exp(i*(phase_a_t - phase_b_t))|
       r_plus  = |sum_t exp(i*(phase_a_t + phase_b_t))|
       denom   = 2 * sqrt(sum(sin(phase_a - mean_a)^2) * sum(sin(phase_b - mean_b)^2))
       r_adj   = (r_minus - r_plus) / denom

   circ_corr_adjusted_masked() below is a vectorized, sample-masked version
   of exactly this formula (verified against pingouin's scalar
   implementation to float precision). This is the SAME correction
   HyPyP's 'accorr' mode already applies in the --legacy-epochs path, so
   both paths now report a consistent, literature-matched measure instead
   of two different circular correlation definitions. The classic
   (non-adjusted) version is kept as circ_corr_masked() and is still
   available via --circ-corr-method classic for comparison; adjusted is
   now the default (--circ-corr-method adjusted), consistent with
   Zimmermann et al.'s recommendation that adjusted circular correlation
   should be preferred for continuous EEG data, which does not have a
   well-defined circular mean.

8. SELECTABLE RE-REFERENCE (--reference average|mastoid); average is still
   the default.
   The Muse's ONLINE reference is FPZ, the electrode in the centre of the
   forehead. That is a poor choice for ocular artifact specifically: FPZ
   sits directly over the orbital/frontalis region, so the reference signal
   itself carries blink and EOG activity. Because every channel is recorded
   as (X - FPZ), that ocular signal is injected into ALL FOUR channels with
   the same sign and a similar amplitude -- i.e. as common mode. This is a
   large part of why ICA (remove_blink_component) separates blinks so poorly
   here: the artifact is not spatially localised in the data, it is spread
   evenly across every channel, which looks like signal rather than like a
   separable component.

   Any re-reference cancels FPZ algebraically, since it appears in every
   term:

       (X - FPZ) - mean(TP9 - FPZ, TP10 - FPZ)  =  X - mean(TP9, TP10)

   so both --reference average and --reference mastoid remove the FPZ
   contamination. The difference is what they leave behind:

     average  -- mixes all four electrodes into every channel (each channel
                 becomes X - mean of all 4). With only 4 electrodes that is
                 a heavy mix: within-subject channels become artificially
                 correlated and every channel carries a share of every
                 other. Fine for inspection, not ideal for connectivity.
     mastoid  -- references to TP9/TP10, which are far from the eyes. The
                 frontal channels stay comparatively independent of each
                 other, which is what the cross-brain metrics actually need.

   Cost of the mastoid option: TP9/TP10 are spent as the reference, and they
   are the electrodes carrying most of the usable posterior alpha. Under a
   linked-mastoid reference TP9 and TP10 become +/-(TP9 - TP10)/2, i.e.
   exact mirror images of each other, so their cross-brain metrics are
   redundant by construction and only AF7/AF8 carry independent information
   (preprocess() warns about this at run time). Restricting the metrics to
   AF7/AF8 is a separate, later change; for now both references are kept
   available so the same recording can be analysed each way and compared --
   the console and summary.txt report clean fraction AND longest continuous
   clean run, which is the number that says whether a reference change
   actually bought usable data.

9. DERIVED BIPOLAR OCULAR CHANNELS (make_ocular_channels).
   The Muse has no EOG electrode, which is why blink handling here has been
   stuck between two bad options: ICA that cannot separate a common-mode
   artifact from 4 channels (point 8), and amplitude thresholding that
   throws away good data along with the blink. A third option is to build
   the missing EOG channels out of the four electrodes we do have:

       saccade = AF7 - AF8
       blink   = mean(AF7, AF8) - mean(TP9, TP10)

   The logic is ordinary bipolar EOG, improvised from this montage.
   Horizontal eye movement drives the two frontal sites to OPPOSITE
   polarity, so differencing them adds the ocular signal and largely
   cancels the brain signal (which is common to both). Blinks drive both
   frontal sites to the SAME polarity, so averaging them keeps the blink,
   and referencing that to the mastoids -- which are far from the eyes --
   turns it into a quasi-bipolar vertical EOG.

   Two properties worth knowing:

   a) Both are REFERENCE-INVARIANT. Every channel is recorded as (X - R)
      for a common R, and R cancels in both expressions:
          (AF7 - R) - (AF8 - R) = AF7 - AF8
          mean(AF7-R, AF8-R) - mean(TP9-R, TP10-R)
              = mean(AF7, AF8) - mean(TP9, TP10)
      So it does not matter where in the chain they are computed, and they
      are identical under --reference average and --reference mastoid. This
      is checked in test_ocular_channels.py.

   b) They get a DIFFERENT filter band from the analysis signal: 0.1-15 Hz
      by default (OCULAR_L_FREQ / OCULAR_H_FREQ, --ocular-band), against
      1-40 Hz for analysis. The reason is detection SNR, not amplitude.
      Measured on a synthetic 300ms blink (test_ocular_channels.py, check 3),
      peak blink amplitude over baseline noise:

          0.1-15 Hz (ocular band)     peak 197 uV   baseline sd  8.2   SNR 24.0
          1-40 Hz   (analysis band)   peak 176 uV   baseline sd 10.7   SNR 16.4

      So the analysis band still retains ~87% of the blink amplitude -- the
      1 Hz high-pass does NOT gut a 300ms blink, whose energy is centred
      nearer 3 Hz than DC. What the ocular band actually buys is a ~1.5x
      better SNR, mostly from the 15 Hz low-pass removing muscle/EMG noise
      that the 40 Hz analysis band lets through, plus a little extra peak
      from the lower high-pass. Worth having for a threshold-based detector,
      but the analysis band is not unusable, and this is not the reason the
      current amplitude thresholding performs poorly.

   This point only CONSTRUCTS the channels and plots them for inspection
   (ocular_channels.png). Point 10 uses them for detection.

10. OCULAR-BASED ARTIFACT DETECTION WITH PER-PARTICIPANT THRESHOLDS
    (--artifact-source eeg|ocular|both, default eeg = unchanged behaviour).

    Two problems with the existing detector, both visible in real data:

    a) It thresholds peak-to-peak amplitude on the EEG channels at a FIXED
       500 uV for everyone. Ocular amplitude varies enormously between
       people -- on session 20260806_125527 the derived blink channel has
       std 28.8 uV for subject A and 93.1 uV for subject B, a 3x difference.
       One fixed number either misses everything in the quieter participant
       or rejects most of the louder one. On that session the 500 uV EEG
       threshold rejects 0.3% of subject A while their blink channel shows
       regular, obvious blinks throughout.

    b) Peak-to-peak over a sliding window fires on anything large, so it
       cannot distinguish a blink from a jaw clench from a cable tug. That
       matters because the right response differs, and because it gives no
       way to check whether the detector is finding what you think it is.

    The fix is to detect on the derived ocular channels (point 9) instead,
    with a threshold computed PER PARTICIPANT from their own data:

        threshold = median(window p2p) + k * 1.4826 * MAD(window p2p)

    MAD rather than sd because the artifacts themselves would inflate an sd
    estimate and drag the threshold up above the very events it is meant to
    catch. k is --ocular-k (default 5). Explicit per-subject overrides are
    available (--blink-threshold-a/-b, --saccade-threshold-a/-b) for when
    the automatic value is visibly wrong.

    Two detectors run over the ocular channels (--ocular-detector):

      ptp       sliding-window peak-to-peak, as before but on the ocular
                channels and with a per-participant threshold. Catches
                anything large.
      velocity  thresholds |d/dt| of the blink channel and then keeps only
                events lasting between --blink-min-dur and --blink-max-dur
                (default 0.05-0.6s). This is the part that is specifically a
                BLINK detector rather than a generic large-thing detector:
                slow drift fails the velocity test, and sustained muscle
                tone fails the duration test.

    Both run by default ("both") and their masks are OR-ed: the velocity
    detector adds blink specificity, the p2p detector keeps coverage of the
    non-blink junk that a blink detector would ignore. Joint masking across
    the dyad is unchanged -- a sample is used only if BOTH subjects are
    clean at that moment.

    --artifact-source stays at 'eeg' by default so this is opt-in and
    directly comparable against the existing numbers. 'ocular' replaces the
    EEG-amplitude criterion, 'both' ORs them.

11. SELECTABLE ANALYSIS CHANNELS (--analysis-channels auto|all|frontal).
    Which channels the CONNECTIVITY METRICS run on, as distinct from which
    channels were recorded. Everything upstream -- referencing, bad-channel
    detection, the derived ocular channels, artifact detection -- still uses
    all four electrodes; only the PLV / circular-correlation step is
    restricted.

    This exists because of the reference artifact in point 8. Under a linked
    mastoid reference, TP9 and TP10 become +(TP9-TP10)/2 and -(TP9-TP10)/2:
    exact mirror images, correlation -1.000. Their cross-brain metrics are
    then redundant by construction. Measured on a real dyad (20260811_113201,
    --reference mastoid --analysis-channels all), the TP9 and TP10 ROWS of
    the 4x4 matrix agree to 2e-16, and so do the columns, for BOTH metrics:
    PLV and the adjusted circular correlation are each invariant to the pi
    phase shift that separates the two channels, so the duplication is exact
    rather than sign-flipped. Of the 16 cells only 9 are distinct -- 7 are
    literal copies. Reporting "16 tests" under that reference therefore
    overstates how many independent tests were run, which feeds straight
    into the FDR correction of point 2. Only AF7 and AF8 carry independent
    information there.

    'auto' (the default) follows the reference: all four channels under
    --reference average, AF7/AF8 under --reference mastoid. 'all' and
    'frontal' force the choice either way, so the same recording can be run
    both ways for comparison.

    A useful side effect: restricting to the frontal pair drops the number
    of cross-brain tests per band from 16 to 4, which is a real reduction in
    the multiple-comparison burden (point 2) rather than a cosmetic one --
    the discarded tests were the redundant ones.

    The pair count is derived from the data everywhere it is reported, so
    the significance counts and the FDR correction follow automatically.

12. REGRESSION-BASED OCULAR CORRECTION (--ocular-correction regress),
    the continuous form of Gratton, Coles & Donchin (1983).

    WHY, in one number. Point 10's detector works -- it finds blinks at a
    normal 7-39/min -- but masking what it finds cannot give long windows.
    analyze_blink_ceiling.py measures the ceiling directly: masking ONLY
    blinks, with no other criterion and no padding, still caps the longest
    JOINTLY-clean stretch at 3.5-12.8s across our dyads (median 8.2s), even
    though individual subjects have blink-free stretches up to 53s. The
    usable mask is the INTERSECTION of two people's blink schedules, and at
    a normal blink rate one of the two is essentially always blinking. Since
    short windows inflate PLV badly (point 5: 0.286 at 2s vs 0.022 whole-
    recording), masking trades one bias for another. Correcting the blink
    and keeping the samples is the only route that preserves long windows.

    HOW. For each EEG channel, fit

        EEG_channel(t) = beta * ocular(t) + residual(t)

    by least squares over the whole continuous recording, then subtract
    beta * ocular(t). One coefficient per channel, no epoching required.

    The MATLAB reference implementation (gratton_emcp.m) epochs the data
    first, but that step exists to subtract the condition-average ERP before
    estimating beta, so that stimulus-locked brain activity is not soaked up
    into the regression weight. There is no ERP to protect in a continuous
    connectivity analysis, so the epoching is unnecessary here and the
    continuous fit is the simpler, more appropriate form.

    AN HONEST CAVEAT ABOUT THIS MONTAGE. Classic Gratton regresses against a
    DEDICATED EOG electrode, which is a separate sensor and therefore not a
    linear combination of the EEG channels. The Muse has no such electrode,
    so the regressor here is itself built from the same four electrodes
    (point 9). Regressing it out is therefore a projection: it removes a
    fixed one-dimensional spatial pattern from a four-dimensional channel
    space. That is closer to removing one ICA component with a FIXED, chosen
    topography than to classic Gratton -- with the advantage over ICA that
    the topography is chosen on physiological grounds rather than estimated
    from 4 channels, and the disadvantage that any brain signal sharing that
    spatial pattern is removed along with the blink. It is not a free
    correction, and it should not be described as equivalent to EOG-based
    Gratton.

    DEGENERACY GUARD. The regressors are linear combinations of the
    electrodes, so with few enough channels they can span the whole space.
    Under --reference mastoid the mastoids become mirror images and
    mean(TP9', TP10') = 0, which makes

        blink   = mean(AF7', AF8')
        saccade = AF7' - AF8'

    Those two together span exactly span{AF7', AF8'} -- so regressing BOTH
    out of a frontal-only analysis set annihilates the data completely.
    gratton_regress() checks the residual variance of every analysis channel
    and refuses the correction if any channel loses essentially all of its
    variance, rather than silently returning zeros. Use one regressor
    (--regress-channels blink) with the frontal set, or keep all four
    channels if you want both.

    MEASURED OUTCOME on this data. The correction does what it was asked to
    do for window length: on 20260806_125527 it took the jointly-clean
    fraction from 91.3% to 95.0% and the longest continuous jointly-clean
    run from 152s to 189s, where blink MASKING gave 10s (point 10). That is
    the whole point of preferring regression, and it works.

    But the caveat above is not hypothetical, and the numbers say how large
    it is. The fitted betas come out at almost exactly the regressor's own
    algebraic coefficients rather than at the artifact's propagation gains:

        subject A, --regress-channels blink
        fit on whole recording   TP9 -0.432  AF7 +0.551  AF8 +0.452  TP10 -0.571
        fit on blink periods     TP9 -0.467  AF7 +0.536  AF8 +0.465  TP10 -0.534
        algebraic d(blink)/d(ch) TP9 -0.5    AF7 +0.5    AF8 +0.5    TP10 -0.5

    Those are the coefficients in the definition of the blink channel, not
    a measurement of how a blink propagated. Restricting the fit to blink
    periods -- the obvious remedy, since blinks are only ~2.5% of samples --
    barely moves them, because the regressor bears the same fixed spatial
    relationship to each channel during a blink as outside one.

    The cost is correspondingly large. Splitting the variance reduction by
    whether a blink was present (subject A):

        channel   variance kept IN blinks   OUTSIDE blinks   ratio
        TP9              26.7%                  71.0%         2.66
        AF7              29.4%                  37.2%         1.26
        AF8              40.6%                  47.8%         1.18
        TP10             21.4%                  50.9%         2.38

    Ratios above 1 mean the correction does preferentially remove
    blink-time variance -- it is not pure shrinkage. But it also removes
    29-63% of the variance at times when no blink is occurring, and for one
    subject's TP10 the ratio was 0.75-1.00, i.e. no blink specificity at
    all. On one channel the residual variance came out ABOVE 100%, meaning
    the subtraction added signal rather than removing it.

    THE FAITHFUL PROCEDURE WAS ALSO IMPLEMENTED (--ocular-correction emcp,
    gratton_emcp / gratton_blink_mask), ported from gratton_emcp.m. It adds
    the two things the simplified version above leaves out: Gratton's own
    matched-filter blink detector, and SEPARATE propagation factors for
    blink and non-blink samples applied piecewise. Two results:

    a) The template detector is the better detector. It found 16.4 and 23.4
       blinks/min on the two subjects of 20260806_125527, against 33/min for
       the velocity detector of point 10 at k=5 -- i.e. squarely in the
       normal 10-20/min range without per-subject tuning. Its shape is what
       does it: the template responds only to a sustained deflection flanked
       by baseline on both sides, so drift and steps score near zero however
       large. If any part of this is worth keeping, it is this detector.

    b) The two-regime structure -- the actual novelty of Gratton -- buys
       nothing here. The blink-regime and saccade-regime factors come out
       the same:

           channel   blink<-vert  sacc<-vert   blink<-horiz  sacc<-horiz
           AF7            0.501       0.503          0.501        0.504
           AF8            0.500       0.504         -0.501       -0.505

       They agree to ~0.003 because, as above, the coefficients are
       structural rather than artifact-dependent, and the structure is the
       same during a blink as outside one.

    Worse, running it in its usual TWO-EOG configuration makes the
    degeneracy sharply worse: with both derived regressors, AF7/AF8 retain
    only 11-12% of their variance on one subject and the guard refuses the
    correction outright on the other. The original assumes vertical and
    horizontal EOG are separate sensors; ours are two linear combinations of
    the same four electrodes, so together they span most of the space. Run
    with --emcp-channels blink (the original's single-vertical-channel mode,
    its example 5) it matches the simplified gratton_regress to within
    rounding -- 95.0% clean, 189s longest run, same variance retained --
    which is what a) and b) predict.

    So the honest summary: with an independent EOG electrode this procedure
    is correct and clean (verified in test_gratton_regression.py check 1,
    where it recovers known gains to ~0.05 and drops artifact correlation
    from 0.53 to 0.01). On the Muse's four electrodes, with the regressor
    necessarily built from those same electrodes, it degenerates into
    subtracting a fixed spatial projection. It buys the long windows that
    masking cannot, at the price of a large amount of genuine signal. Which
    trade is preferable is a judgement call about the specific analysis, not
    something this file should decide -- hence --ocular-correction defaults
    to none, and both paths are available for comparison.
-----------------------------------------------------------------------------
"""
import argparse
import glob
import json
import os
import sys
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from hypyp import analyses


CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
FREQ_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# Pass band for the derived ocular channels (point 9). Deliberately lower
# than the 1-40 Hz analysis band: blink energy is mostly below 3 Hz and the
# analysis high-pass would throw most of it away.
OCULAR_L_FREQ = 0.1
OCULAR_H_FREQ = 15.0

# Channels the connectivity metrics run on (point 11). Set once in main() via
# set_analysis_channels(); everything upstream of the metrics keeps using all
# four electrodes. Module-level because the plotting helpers need the labels
# and threading them through four plot functions buys nothing.
ANALYSIS_CH = list(CH_NAMES)


def set_analysis_channels(names):
    global ANALYSIS_CH
    ANALYSIS_CH = list(names)


def resolve_analysis_channels(selection, reference):
    """
    Which channels the metrics should use, given --analysis-channels and the
    active reference. See point 11: under a linked-mastoid reference TP9 and
    TP10 are mirror images of each other, so only the frontal pair carries
    independent information.
    """
    if selection == "all":
        return list(CH_NAMES)
    if selection == "frontal":
        return ["AF7", "AF8"]
    return ["AF7", "AF8"] if reference == "mastoid" else list(CH_NAMES)


def restrict_to_analysis(inst, analysis_ch, subject_label="", quiet=False):
    """
    Drop channels outside analysis_ch from a Raw or Epochs.

    Channel names are subject-prefixed ("A_TP9"), so match on the suffix.
    Called AFTER referencing, ocular-channel construction and artifact
    detection, all of which need the full montage.
    """
    keep = [ch for ch in inst.ch_names if ch.split("_")[-1] in analysis_ch]
    if not keep:
        raise ValueError(
            f"{subject_label}: no channels left after restricting to "
            f"{analysis_ch} (have {inst.ch_names})")
    if len(keep) == len(inst.ch_names):
        return inst
    dropped = [ch for ch in inst.ch_names if ch not in keep]
    bad_kept = [ch for ch in keep if ch in inst.info["bads"]]
    if not quiet:
        print(f"     {subject_label}: metrics on {[c.split('_')[-1] for c in keep]} "
              f"(dropped {[c.split('_')[-1] for c in dropped]})")
        if bad_kept:
            print(f"     {subject_label}: WARNING analysis channel(s) "
                  f"{[c.split('_')[-1] for c in bad_kept]} are flagged bad -- "
                  "the metrics for this subject rest on unreliable data")
    return inst.copy().pick(keep)

# saccade/blink traces in uV, plus the band they were built in. Either trace
# may be None when the electrodes it needs were flagged bad.
OcularChannels = namedtuple(
    "OcularChannels", "saccade blink fs l_freq h_freq label")


# ============================================================
# 1. LOADING
# ============================================================

def load_stimulus_onset(csv_path):
    """
    Look for a _markers.json sidecar next to the CSV (written by record_single.py).
    Returns the rel_time_s of the first stimulus marker, or None if no sidecar found.
    """
    sidecar = csv_path.replace(".csv", "_markers.json")
    if not os.path.exists(sidecar):
        return None
    # utf-8-sig, not utf-8: some sidecars were written with a BOM, which
    # json.load rejects outright. That silently crashed the whole run for
    # any such session (20260729_220724 was excluded from every comparison
    # table this way before it was noticed), so decode the BOM rather than
    # die on it.
    with open(sidecar, encoding="utf-8-sig") as f:
        markers = json.load(f)
    for m in markers:
        if "stimulus" in m.get("marker", "").lower():
            return float(m["rel_time_s"])
    if markers:
        return float(markers[0]["rel_time_s"])
    return None


def load_csv_to_raw(csv_path, subject_label, onset_s=None):
    """
    CSV (lsl_timestamp + 4 channels + is_gap) -> MNE Raw with BAD_gap annotations.

    Each channel gets prefixed with the subject label (e.g. 'A_TP9') so when we
    later concatenate the two subjects' data into one Raw for HyPyP, we can tell
    them apart.
    """
    df = pd.read_csv(csv_path)
    # accept either lsl_timestamp (record_both.py) or time_s (record_single.py)
    ts_col = "lsl_timestamp" if "lsl_timestamp" in df.columns else "time_s"
    required = [ts_col] + CH_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    # IBS stimulus alignment: trim to [onset, end] and re-zero timestamps
    if onset_s is not None:
        df = df[df[ts_col] >= onset_s].copy().reset_index(drop=True)
        df[ts_col] = df[ts_col] - onset_s
        print(f"     stimulus alignment: trimmed to onset at {onset_s:.3f}s")

    # infer sampling rate from timestamps
    dt = np.diff(df[ts_col].values)
    fs = 1.0 / float(np.median(dt))
    fs_rounded = round(fs)
    if abs(fs - fs_rounded) > 0.5:
        print(f"  WARNING: {csv_path} has irregular sampling (~{fs:.2f} Hz)")
    fs = float(fs_rounded)

    # build MNE Raw
    data = df[CH_NAMES].values.T  # shape (n_channels, n_samples), microvolts
    # MNE wants volts; convert from microvolts
    data = data * 1e-6
    # if there are NaN gaps, MNE filtering will explode - fill with 0 for now,
    # the annotations below tell downstream code to ignore those segments anyway
    nan_mask_per_channel = np.isnan(data)
    data = np.where(nan_mask_per_channel, 0.0, data)

    ch_names_prefixed = [f"{subject_label}_{c}" for c in CH_NAMES]
    info = mne.create_info(ch_names=ch_names_prefixed, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # attach standard 10-20 electrode coordinates so MNE knows where TP9/AF7/
    # AF8/TP10 sit on the scalp (enables spatial-color PSD plots, topomaps,
    # and any future montage-aware steps). Renaming is needed because the
    # montage's built-in names ("TP9") don't match our subject-prefixed ones
    # ("A_TP9") -- MNE matches montage points to raw channels by exact name.
    montage = mne.channels.make_standard_montage("standard_1020")
    rename_map = {f"{subject_label}_{c}": c for c in CH_NAMES}
    raw.rename_channels(rename_map)
    raw.set_montage(montage, on_missing="warn", verbose=False)
    raw.rename_channels({v: k for k, v in rename_map.items()})

    # annotations from is_gap column
    if "is_gap" in df.columns:
        gap = df["is_gap"].values.astype(bool)
    else:
        # fall back to any-NaN-across-channels
        gap = nan_mask_per_channel.any(axis=0)

    onsets, durations = gap_runs_to_annotations(gap, fs,
                                                start_time=df[ts_col].iloc[0])
    if len(onsets) > 0:
        anns = mne.Annotations(
            onset=onsets - df[ts_col].iloc[0],   # relative to raw start
            duration=durations,
            description=["BAD_gap"] * len(onsets),
        )
        raw.set_annotations(anns)

    clean_frac = float((~gap).mean())
    print(f"  {subject_label}: {csv_path}")
    print(f"     samples={len(df)}  fs={fs:.0f} Hz  duration={len(df)/fs:.1f}s")
    print(f"     clean fraction={clean_frac:.1%}  gap annotations={len(onsets)}")
    return raw, fs


def gap_runs_to_annotations(gap_mask, fs, start_time=0.0):
    """Turn a boolean is_gap array into (onsets, durations) absolute-time arrays."""
    if not gap_mask.any():
        return np.array([]), np.array([])
    # find run boundaries
    g = gap_mask.astype(int)
    edges = np.diff(np.concatenate(([0], g, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    onsets = start_time + starts / fs
    durations = (ends - starts) / fs
    return onsets, durations


# ============================================================
# 2. PREPROCESS + EPOCH
# ============================================================

def remove_blink_component(raw, subject_label, random_state=42):
    """
    Best-effort blink removal via ICA.

    Muse has no dedicated EOG channel, so the frontal channel (AF7 or AF8,
    most exposed to blinks) is used as a proxy target for
    ICA.find_bads_eog(). Only 4 EEG channels are available in total, so
    separation is weak compared to a real multi-channel montage -- at most
    the single most blink-correlated component is removed, to avoid
    discarding real brain signal along with the artifact.
    """
    proxy_ch = next((ch for ch in raw.ch_names if ch.endswith("AF7") or ch.endswith("AF8")), None)
    if proxy_ch is None:
        print(f"     {subject_label}: ICA skipped -- no frontal channel found for blink proxy")
        return raw

    ica = ICA(max_iter="auto", random_state=random_state)
    ica.fit(raw, verbose=False)

    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=proxy_ch, verbose=False)
    except Exception as e:
        print(f"     {subject_label}: ICA blink-detection via {proxy_ch} failed ({e}) -- no components removed")
        return raw

    if not eog_indices:
        print(f"     {subject_label}: ICA found no clear blink component -- no components removed")
        return raw

    ica.exclude = eog_indices[:1]
    print(f"     {subject_label}: ICA removed component {ica.exclude} (blink-correlated via {proxy_ch})")
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


def detect_bad_channels(raw, railed_uv=990.0, railed_frac_thresh=0.02, flat_std_uv=5.0,
                        strict_channels=(), strict_railed_frac_thresh=0.005,
                        strict_flat_std_uv=8.0):
    """
    Flag channels that are chronically railed (stuck near the Muse's ADC
    limit -- poor scalp contact or motion) or abnormally flat (near-zero
    variance -- a loose/disconnected electrode). Checked on the raw
    (pre-filter) signal, since railing is clearest there: band-pass
    filtering smooths sharp rail transitions and can hide them.

    railed_uv=990 sits just inside the Muse's +/-1000 uV rail. flat_std_uv=5
    is well below typical scalp EEG (tens of uV), so it only catches
    channels that are essentially dead, not just quiet.

    strict_channels holds channels that are about to be used AS THE
    REFERENCE (the mastoids, under --reference mastoid). Those get tighter
    thresholds, because a bad reference channel does not merely make itself
    unusable -- it subtracts its own noise into EVERY other channel for that
    subject. A mildly flaky mastoid would pass the ordinary gate and quietly
    contaminate the whole montage, so it is worth catching earlier than a
    mildly flaky AF7 would be.

    Returns a list of channel names to mark bad.
    """
    data_uv = raw.get_data(picks="eeg") * 1e6
    bad = []
    for ch_name, ch_data in zip(raw.ch_names, data_uv):
        is_strict = ch_name in strict_channels
        railed_thresh = strict_railed_frac_thresh if is_strict else railed_frac_thresh
        flat_thresh = strict_flat_std_uv if is_strict else flat_std_uv
        note = "  [reference channel: strict threshold]" if is_strict else ""
        railed_frac = float((np.abs(ch_data) >= railed_uv).mean())
        std_uv = float(ch_data.std())
        if railed_frac > railed_thresh:
            print(f"     {ch_name}: railed {railed_frac:.1%} of samples "
                  f"(>= {railed_uv:.0f} uV) -- marking bad{note}")
            bad.append(ch_name)
        elif std_uv < flat_thresh:
            print(f"     {ch_name}: abnormally flat (std={std_uv:.1f} uV, "
                  f"< {flat_thresh:.0f} uV) -- marking bad{note}")
            bad.append(ch_name)
    return bad


def mastoid_channels(raw):
    """This subject's mastoid channels (TP9, TP10), in that order.

    Channel names are subject-prefixed ('A_TP9'), so match on the suffix.
    Returns whichever of the two are actually present.
    """
    out = []
    for suffix in ("TP9", "TP10"):
        out += [ch for ch in raw.ch_names if ch.split("_")[-1] == suffix]
    return out


def longest_clean_run_s(good_mask, fs):
    """Longest run of consecutive True samples in good_mask, in seconds.

    Reported alongside overall clean fraction because they answer different
    questions: clean fraction says how much data survived, this says whether
    it survived in usable-length pieces. Two recordings can both be 60%
    clean while one has a single 100s stretch and the other has a hundred
    0.6s crumbs -- and short windows systematically inflate PLV (see point
    5), so the second is far worse than the fraction alone suggests.
    """
    good = np.asarray(good_mask, dtype=bool)
    if not good.any():
        return 0.0
    # run lengths via the positions where the mask changes value
    edges = np.flatnonzero(np.diff(np.concatenate(([0], good.view(np.int8), [0]))))
    return float((edges[1::2] - edges[0::2]).max()) / fs


def make_ocular_channels(raw, l_freq=OCULAR_L_FREQ, h_freq=OCULAR_H_FREQ,
                         subject_label="", bads=()):
    """
    Build bipolar ocular channels from the four Muse electrodes:

        saccade = AF7 - AF8                          (horizontal eye movement)
        blink   = mean(AF7, AF8) - mean(TP9, TP10)   (vertical / blink)

    See point 9 of the module docstring. Takes the RAW (unfiltered) signal,
    because it applies its own lower-frequency pass band -- do not hand it
    the output of preprocess(), which has already high-passed at 1 Hz and
    discarded most of the blink amplitude.

    bads is the list of channels flagged unusable (typically
    raw_pp.info["bads"]). A derived channel whose electrodes are bad is
    returned as None rather than as a plausible-looking trace built from a
    dead electrode.

    Returns an OcularChannels namedtuple; .saccade and .blink are arrays in
    microvolts, or None.
    """
    idx = {ch.split("_")[-1]: i for i, ch in enumerate(raw.ch_names)}
    bad_short = {ch.split("_")[-1] for ch in bads}
    sfreq = raw.info["sfreq"]

    # A 0.1 Hz high-pass needs an FIR filter roughly 33s long, so short
    # recordings cannot support it. MNE does not RAISE on this -- it emits a
    # RuntimeWarning ("filter_length is longer than the signal, distortion is
    # likely") and filters anyway, which would silently hand back a distorted
    # blink channel. So check the length up front and step the high-pass up
    # instead; a 0.5 Hz blink channel is still usable, and the caller is told
    # which band was actually used.
    filt, used_l = None, l_freq
    for candidate in (l_freq, 0.5, 1.0):
        if candidate is None or candidate <= 0:
            continue
        # MNE's 'auto' FIR length: 3.3 / transition-bandwidth seconds, with
        # l_trans_bandwidth = min(max(l_freq * 0.25, 2.0), l_freq)
        trans = min(max(candidate * 0.25, 2.0), candidate)
        if 3.3 / trans * sfreq > raw.n_times:
            continue
        try:
            filt = raw.copy().filter(l_freq=candidate, h_freq=h_freq,
                                     picks="eeg", verbose=False)
            used_l = candidate
            break
        except ValueError:
            continue
    if filt is None:
        print(f"     {subject_label}: ocular channels unavailable -- "
              f"recording too short to high-pass ({raw.n_times} samples)")
        return OcularChannels(None, None, raw.info["sfreq"], l_freq, h_freq,
                              subject_label)
    if used_l != l_freq:
        print(f"     {subject_label}: ocular high-pass raised "
              f"{l_freq} -> {used_l} Hz (recording too short for "
              f"{l_freq} Hz)")

    data = filt.get_data(picks="eeg") * 1e6

    def get(name):
        return data[idx[name]] if name in idx else None

    af7, af8 = get("AF7"), get("AF8")
    tp9, tp10 = get("TP9"), get("TP10")

    saccade = None
    if af7 is not None and af8 is not None and not ({"AF7", "AF8"} & bad_short):
        saccade = af7 - af8
    else:
        print(f"     {subject_label}: no saccade channel -- AF7/AF8 missing "
              f"or flagged bad")

    blink = None
    frontal = [x for x, n in ((af7, "AF7"), (af8, "AF8"))
               if x is not None and n not in bad_short]
    mastoid = [x for x, n in ((tp9, "TP9"), (tp10, "TP10"))
               if x is not None and n not in bad_short]
    if frontal and mastoid:
        # A single good mastoid still gives a usable (if laterally biased)
        # vertical channel, so fall back to it rather than dropping the
        # blink channel entirely.
        blink = np.mean(frontal, axis=0) - np.mean(mastoid, axis=0)
        if len(mastoid) == 1 or len(frontal) == 1:
            print(f"     {subject_label}: blink channel built from "
                  f"{len(frontal)} frontal / {len(mastoid)} mastoid "
                  "electrode(s) -- reduced quality")
    else:
        print(f"     {subject_label}: no blink channel -- needs at least one "
              "good frontal AND one good mastoid electrode")

    for name, trace in (("saccade", saccade), ("blink", blink)):
        if trace is not None:
            print(f"     {subject_label}: {name:8s} std={trace.std():6.1f} uV  "
                  f"p99|x|={np.percentile(np.abs(trace), 99):7.1f} uV  "
                  f"range={trace.min():.0f}..{trace.max():.0f} uV")

    return OcularChannels(saccade, blink, filt.info["sfreq"], used_l, h_freq,
                          subject_label)


def preprocess(raw, l_freq=1.0, h_freq=40.0, use_ica=False, subject_label="",
               reference="average"):
    """
    Bandpass + optional ICA blink removal + re-reference.

    reference="average" (default): common average over that subject's good
    channels. reference="mastoid": linked mastoid, i.e. every channel minus
    mean(TP9, TP10). See point 8 of the module docstring for why the choice
    matters -- in short, the Muse's online reference is FPZ, which sits over
    the eyes and injects ocular activity into all four channels as common
    mode. Both options cancel FPZ, but the mastoid option leaves the frontal
    channels far less mixed into each other than a 4-electrode average does.

    Bad channels (see detect_bad_channels) are excluded from the reference
    computation -- a chronically railed or dead channel would otherwise drag
    the shared reference around and contaminate the other, otherwise good,
    channels for that subject.
    """
    raw = raw.copy()
    strict = mastoid_channels(raw) if reference == "mastoid" else ()
    bad_chs = detect_bad_channels(raw, strict_channels=strict)
    raw.info["bads"] = bad_chs
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    if use_ica:
        raw = remove_blink_component(raw, subject_label)

    if reference == "mastoid":
        mastoids = mastoid_channels(raw)
        ref_chs = [ch for ch in mastoids if ch not in raw.info["bads"]]
        dropped = [ch for ch in mastoids if ch in raw.info["bads"]]
        if len(ref_chs) == 2:
            print(f"     {subject_label}: linked-mastoid reference {ref_chs}")
            # X - mean(TP9, TP10) makes TP9 and TP10 into +(TP9-TP10)/2 and
            # -(TP9-TP10)/2: exact mirror images. Their cross-brain metrics
            # are therefore redundant by construction (identical up to sign),
            # so only AF7/AF8 carry independent information here.
            print(f"     {subject_label}: note -- TP9/TP10 are now mirror "
                  "images of each other (an artefact of being the "
                  "reference); only AF7/AF8 carry independent information "
                  "under this reference")
        elif len(ref_chs) == 1:
            print(f"     {subject_label}: WARNING only one usable mastoid "
                  f"({dropped} flagged bad) -- referencing to {ref_chs[0]} "
                  "alone. A single-mastoid reference is laterally biased; "
                  "treat this subject's values as suspect.")
        else:
            print(f"     {subject_label}: WARNING both mastoids {mastoids} "
                  "flagged bad -- mastoid reference unavailable, falling "
                  "back to an average reference over good channels")

        # MNE applies the reference to the GOOD channels only, and raises if
        # there are none -- which happens when every channel is already
        # flagged bad. Check before asking, so a hopeless subject warns and
        # returns instead of killing the run.
        receivers = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
        if ref_chs and receivers:
            raw.set_eeg_reference(ref_channels=ref_chs, projection=False,
                                  verbose=False)
            if len(ref_chs) == 1:
                # X - X is identically zero, so a single reference channel
                # carries no signal after referencing. Flag it AFTER the
                # reference is applied (flagging it first would remove it
                # from the receiver set and can leave MNE nothing to
                # reference) so nothing downstream mistakes a flat trace for
                # real data.
                raw.info["bads"] = raw.info["bads"] + [ref_chs[0]]
            return raw
        if ref_chs and not receivers:
            print(f"     {subject_label}: WARNING all channels flagged bad "
                  f"({raw.info['bads']}) -- leaving data unreferenced; this "
                  "subject has no usable data")
            return raw
        # else (no usable mastoid): fall through to the average branch below

    good_chs = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    if not good_chs:
        # MNE raises ("No channels supplied to apply the reference to") if
        # every channel is bad, so don't ask it to. Nothing usable survives
        # this subject anyway -- leave the data unreferenced and let the
        # downstream artifact/epoch stage report the empty result, rather
        # than crashing the whole run here.
        print(f"     {subject_label}: WARNING all channels flagged bad "
              f"({bad_chs}) -- leaving data unreferenced; this subject has "
              "no usable data")
    else:
        if bad_chs:
            print(f"     {subject_label}: referencing to good channels only "
                  f"{good_chs} (excluding {bad_chs})")
        # average reference computed from good channels only, applied to
        # that subject's channels (MNE leaves bad channels' own data
        # unreferenced, which is fine -- their values are already flagged
        # unreliable downstream)
        raw.set_eeg_reference(ref_channels=good_chs, projection=False, verbose=False)
    return raw


def gratton_regress(raw, oc, regressors=("blink",), l_freq=1.0, h_freq=40.0,
                    subject_label="", min_variance_kept=0.10, quiet=False,
                    fit_on="blinks", fit_mask=None, k=10.0):
    """
    Regression-based ocular correction, continuous (non-epoched) form of
    Gratton, Coles & Donchin (1983). See point 12 of the module docstring,
    including the caveat that on this montage the regressor is derived from
    the same electrodes it corrects.

    raw must already be preprocessed (referenced and band-passed); oc is the
    OcularChannels for the same subject. The regressors are re-filtered to
    the raw's own band before fitting, so the subtraction cannot introduce
    out-of-band content that the analysis signal does not contain.

    fit_on selects WHERE beta is estimated, which matters far more than it
    might seem. Blinks occupy only ~2.5% of a typical recording here, so a
    fit over the whole recording is dominated by the 97.5% of the time when
    the "regressor" is simply a linear combination of ongoing brain signal.
    The resulting beta then measures the STRUCTURAL projection of one linear
    combination of the electrodes onto another, and subtracting it strips
    out that structural component everywhere -- measured on a real subject,
    a whole-recording fit removed 29-63% of the variance OUTSIDE any blink.

    fit_on="blinks" (default) estimates beta only on samples where the
    velocity detector (point 10) found a blink, so the fit is dominated by
    the artifact rather than by ongoing EEG, and applies that beta to the
    whole recording. This is the intent of the original Gratton procedure,
    where the EOG channel is a separate electrode and its variance is
    artifact-dominated by construction. fit_on="all" restores the naive
    whole-recording fit for comparison. fit_mask overrides the automatic
    blink detection with an explicit boolean mask.

    Refuses the correction (returns the input unchanged, with ok=False) if
    any channel would be left with less than min_variance_kept of its
    original variance -- the degeneracy described in point 12. The default
    is 10%, not something nearer zero: in the fully degenerate case
    (mastoid reference, both regressors) the residual does not land at
    exactly 0% because the regressors are re-filtered to the analysis band
    before fitting, so a little out-of-band variance survives the
    projection. Measured on synthetic data that case leaves about 6%, which
    a 2% bar would wave through.

    Returns (raw_corrected, info) where info carries the per-channel betas,
    the variance retained per channel, and whether the correction was applied.
    """
    available = {"blink": oc.blink, "saccade": oc.saccade}
    names = [n for n in regressors if available.get(n) is not None]
    missing = [n for n in regressors if available.get(n) is None]
    info = {"applied": False, "regressors": names, "missing": missing,
            "betas": {}, "variance_kept": {}}
    if not names:
        if not quiet:
            print(f"     {subject_label}: no ocular regressor available "
                  f"({missing} unavailable) -- regression skipped")
        return raw, info

    n = raw.n_times
    sfreq = raw.info["sfreq"]
    # regressors in volts, truncated/padded to the EEG length
    X = np.zeros((n, len(names)))
    for j, name in enumerate(names):
        trace = np.asarray(available[name], dtype=float) * 1e-6
        m = min(n, len(trace))
        X[:m, j] = trace[:m]
    # match the analysis band: the ocular channels were built in 0.1-15 Hz
    # (point 9), and subtracting a component the EEG cannot contain would
    # inject out-of-band signal rather than remove artifact
    X = mne.filter.filter_data(X.T.copy(), sfreq, l_freq, h_freq,
                               verbose=False).T

    Y = raw.get_data().T                      # (n_samples, n_channels)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # where to ESTIMATE beta (it is APPLIED everywhere either way)
    if fit_mask is not None:
        fit = np.asarray(fit_mask, dtype=bool)[:n]
    elif fit_on == "blinks" and oc.blink is not None:
        thr = ocular_thresholds(oc, k=k)["blink_velocity"]
        fit, n_blinks, _ = velocity_event_mask(oc.blink, oc.fs, thr)
        fit = fit[:n] if len(fit) >= n else np.pad(fit, (0, n - len(fit)))
        info["n_blinks_fit"] = n_blinks
    else:
        fit = np.ones(n, dtype=bool)
    if fit.sum() < 10 * max(1, len(names)):
        if not quiet:
            print(f"     {subject_label}: only {int(fit.sum())} samples to fit "
                  "beta on -- falling back to a whole-recording fit")
        fit = np.ones(n, dtype=bool)
    info["fit_on"] = fit_on
    info["fit_samples"] = int(fit.sum())
    info["fit_fraction"] = float(fit.mean())

    beta, *_ = np.linalg.lstsq(Xc[fit], Yc[fit], rcond=None)  # (n_reg, n_chan)
    residual = Yc - Xc @ beta

    var_before = Yc.var(axis=0)
    var_after = residual.var(axis=0)
    kept = np.divide(var_after, var_before,
                     out=np.ones_like(var_after), where=var_before > 0)

    for i, ch in enumerate(raw.ch_names):
        info["betas"][ch] = {names[j]: float(beta[j, i]) for j in range(len(names))}
        info["variance_kept"][ch] = float(kept[i])

    annihilated = [ch for ch, k in info["variance_kept"].items()
                   if k < min_variance_kept]
    if annihilated:
        if not quiet:
            print(f"     {subject_label}: REFUSING regression -- "
                  f"{annihilated} would lose >"
                  f"{100 * (1 - min_variance_kept):.0f}% of their variance. "
                  f"The regressors {names} span the channel space (point 12); "
                  "use --regress-channels blink, or keep all four channels.")
        return raw, info

    out = raw.copy()
    out._data = (residual + Y.mean(axis=0, keepdims=True)).T
    info["applied"] = True
    if not quiet:
        where = (f"beta fit on {info['fit_samples']} blink samples "
                 f"({100 * info['fit_fraction']:.1f}% of the recording)"
                 if fit_on == "blinks" and fit.mean() < 1.0
                 else "beta fit on the whole recording")
        print(f"     {subject_label}: regressed out {names}, {where}")
        print(f"     {subject_label}: variance kept "
              + "  ".join(f"{ch.split('_')[-1]}={100 * info['variance_kept'][ch]:.0f}%"
                          for ch in raw.ch_names))
        if missing:
            print(f"     {subject_label}: note -- {missing} unavailable, "
                  "not regressed")
    return out, info


def gratton_blink_mask(trace, sfreq, criterion=14.0, wind_variance=2.0):
    """
    Gratton's own blink detector, ported from gratton_emcp.m.

    A matched filter rather than an amplitude or velocity threshold: the
    vertical-EOG trace is correlated with a three-part step template
    [-1 x (third+1), +2 x third, -1 x (third-1)] spanning 210ms, where
    `third` is 70ms worth of samples forced odd. The "slope" statistic is
    that correlation divided by wind_variance**2, and any centre point whose
    |slope| exceeds `criterion` marks +/- (middle-1) samples around it as
    blink.

    The shape is what makes it selective: it responds to a sustained
    deflection flanked by baseline on BOTH sides, so a step or a drift
    scores near zero even when large, and only a bump of roughly blink
    duration scores highly. Note the template is deliberately asymmetric
    (third+1 vs third-1) -- that is how the original is written, and it is
    kept here so the port stays faithful.

    criterion=14.0 and wind_variance=2.0 are the original's constants, which
    assume data in microvolts.

    Returns (mask, n_events).
    """
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    ms_ten = int(round(10.0 / (1000.0 / sfreq) + 0.5))
    third = ms_ten * 7
    if third % 2 == 0:
        third += 1
    middle = (third + 1) // 2
    length = third * 3
    mark = np.zeros(n, dtype=bool)
    if n < length + 2 * middle:
        return mark, 0

    template = np.concatenate([-np.ones(third + 1),
                               2.0 * np.ones(third),
                               -np.ones(third - 1)])
    # out[s] = sum(trace[s:s+length] * template); the window centred on that
    # correlation sits at s + third + middle - 1
    out = np.correlate(trace, template, mode="valid") / float(length)
    slope = out / (wind_variance ** 2)
    centres = np.flatnonzero(np.abs(slope) > criterion) + third + middle - 1

    n_events = 0
    prev_end = -1
    for c in centres:
        lo, hi = max(0, c - middle + 1), min(n, c + middle)
        mark[lo:hi] = True
        if lo > prev_end:          # count contiguous groups, not samples
            n_events += 1
        prev_end = max(prev_end, hi)
    return mark, n_events


def gratton_emcp(raw, oc, l_freq=1.0, h_freq=40.0, subject_label="",
                 criterion=14.0, min_variance_kept=0.10, quiet=False,
                 channels="blink"):
    """
    Gratton, Coles & Donchin (1983) eye-movement correction, ported from
    gratton_emcp.m (Gehring's EEGLAB implementation) to continuous data.

    Differences from gratton_regress() -- which is the simplified version --
    are the two that matter in the original:

      1. TWO sets of propagation factors, not one. The recording is split
         into blink and non-blink ("saccade") samples by the template
         detector above, and a separate no-intercept regression is run on
         each. Correction is then applied PIECEWISE: blink samples get
         PropBlink, everything else gets PropSaccade. The rationale is that
         a blink and a saccade propagate to the scalp differently, so one
         coefficient cannot describe both.
      2. Means are removed WITHIN each regime before fitting, and the
         regime's own mean is added back into the subtraction, rather than
         one grand mean over the whole recording.

    Dropped from the original, deliberately: the selection-card machinery
    that subtracts each condition's average ERP before fitting. That step
    exists to stop stimulus-locked brain activity being absorbed into the
    propagation factors; a continuous connectivity analysis has no ERP to
    protect, so the whole-recording mean is removed instead. This is what
    makes the procedure work without epoching -- the epoching in the MATLAB
    version is a carrier for the ERP subtraction, not a requirement of the
    regression.

    channels="both" uses vertical and horizontal EOG, as the original's
    usual configuration does; channels="blink" uses the vertical channel
    alone, which the original also supports (its example 5: "correct for a
    single EOG channel ... corrects only blinks and vertical EOG"). On this
    montage the single-channel form is the safer default, because both
    derived regressors are linear combinations of the same four electrodes
    and using both together strips most of the frontal variance -- see the
    degeneracy discussion in point 12.

    Returns (raw, info) where info mirrors the MATLAB EEG.emcp structure.
    """
    info = {"applied": False, "blink_factors": {}, "saccade_factors": {},
            "variance_kept": {}, "n_blink_events": 0}
    if oc.blink is None:
        if not quiet:
            print(f"     {subject_label}: EMCP needs a vertical (blink) "
                  "channel -- unavailable, skipped")
        return raw, info

    n = raw.n_times
    sfreq = raw.info["sfreq"]
    names = ["blink"]
    if channels == "both" and oc.saccade is not None:
        names.append("saccade")
    X = np.zeros((n, len(names)))
    for j, nm in enumerate(names):
        tr = np.asarray(getattr(oc, nm), dtype=float)
        m = min(n, len(tr))
        X[:m, j] = tr[:m]
    X = mne.filter.filter_data(X.T.copy(), sfreq, l_freq, h_freq,
                               verbose=False).T

    # blink/saccade split, on the vertical channel, per the original
    is_blink, n_events = gratton_blink_mask(X[:, 0], sfreq, criterion=criterion)
    info["n_blink_events"] = n_events
    info["blink_fraction"] = float(is_blink.mean())
    n_b, n_s = int(is_blink.sum()), int((~is_blink).sum())
    if not quiet:
        print(f"     {subject_label}: EMCP template found {n_events} blinks "
              f"({100 * is_blink.mean():.1f}% of samples; "
              f"{60 * n_events / (n / sfreq):.1f}/min)")

    Y = raw.get_data().T * 1e6                       # uV, as the original
    # whole-recording mean removal stands in for the epoch-mean step
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    def fit(mask):
        """No-intercept least squares on one regime, means removed within it."""
        if mask.sum() <= len(names):
            return None
        xb = Xc[mask] - Xc[mask].mean(axis=0, keepdims=True)
        yb = Yc[mask] - Yc[mask].mean(axis=0, keepdims=True)
        beta, *_ = np.linalg.lstsq(xb, yb, rcond=None)
        return beta

    prop_blink = fit(is_blink) if n_b > 0 else None
    prop_sacc = fit(~is_blink)
    if prop_sacc is None:
        if not quiet:
            print(f"     {subject_label}: EMCP has too few non-blink samples "
                  "to fit -- skipped")
        return raw, info
    if prop_blink is None and not quiet:
        print(f"     {subject_label}: EMCP found no blinks -- all data "
              "treated as saccade, as the original does")

    corrected = Yc.copy()
    for mask, prop in ((is_blink, prop_blink), (~is_blink, prop_sacc)):
        if prop is None or not mask.any():
            continue
        # subtract the propagated DEVIATION from this regime's mean, then the
        # regime's own channel mean -- the original's
        # EEG - (Prop*(EOG - EOGmean) + EEGmean)
        x_mean = Xc[mask].mean(axis=0, keepdims=True)
        y_mean = Yc[mask].mean(axis=0, keepdims=True)
        corrected[mask] -= (Xc[mask] - x_mean) @ prop + y_mean

    var_before, var_after = Yc.var(axis=0), corrected.var(axis=0)
    kept = np.divide(var_after, var_before,
                     out=np.ones_like(var_after), where=var_before > 0)
    for i, ch in enumerate(raw.ch_names):
        info["variance_kept"][ch] = float(kept[i])
        if prop_blink is not None:
            info["blink_factors"][ch] = {names[j]: float(prop_blink[j, i])
                                         for j in range(len(names))}
        info["saccade_factors"][ch] = {names[j]: float(prop_sacc[j, i])
                                       for j in range(len(names))}

    annihilated = [ch for ch, k in info["variance_kept"].items()
                   if k < min_variance_kept]
    if annihilated:
        if not quiet:
            print(f"     {subject_label}: REFUSING EMCP -- {annihilated} "
                  f"would lose >{100 * (1 - min_variance_kept):.0f}% of their "
                  "variance (see point 12 on regressor degeneracy)")
        return raw, info

    out = raw.copy()
    out._data = (corrected + Y.mean(axis=0, keepdims=True)).T * 1e-6
    info["applied"] = True
    if not quiet:
        print(f"     {subject_label}: EMCP corrected using {names}, "
              "variance kept "
              + "  ".join(f"{ch.split('_')[-1]}={100 * info['variance_kept'][ch]:.0f}%"
                          for ch in raw.ch_names))
    return out, info


def epoch_with_gap_rejection(raw, epoch_len_s, overlap_s, amplitude_uv=150.0):
    """
    Fixed-length epochs, rejecting any epoch that:
      - overlaps a BAD_gap annotation (lost BLE packets), OR
      - has peak-to-peak amplitude > amplitude_uv on any channel (muscle artifact)

    150 uV is a standard threshold for consumer EEG. Lower it (e.g. 100) for
    stricter rejection; raise it (e.g. 200) if too many epochs are lost.
    """
    events = mne.make_fixed_length_events(
        raw, duration=epoch_len_s, overlap=overlap_s
    )
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=epoch_len_s - 1.0 / raw.info["sfreq"],
        baseline=None, preload=True, reject_by_annotation=True,
        reject={"eeg": amplitude_uv * 1e-6},
        verbose=False,
    )
    return epochs


def load_and_epoch_subject(csv_path, subject_label, epoch_len_s, overlap_s,
                            h_freq=40.0, amplitude_uv=150.0, use_ica=False,
                            align_onset=True, quiet=False, reference="average"):
    """
    Full load -> preprocess -> epoch chain for ONE subject's CSV.

    Factored out so the same steps can be reused for the two main subjects
    AND for any --pool-dir subjects used in pseudo-pair validation (they all
    need to go through identical preprocessing to be a fair comparison).

    Legacy path (--legacy-epochs) only -- see load_and_preprocess_continuous
    for the default, continuous artifact-rejection path.

    Returns (epochs, fs) or (None, None) if the file couldn't be used.
    """
    try:
        onset = load_stimulus_onset(csv_path) if align_onset else None
        raw, fs = load_csv_to_raw(csv_path, subject_label, onset_s=onset)
    except Exception as e:
        if not quiet:
            print(f"  WARNING: could not load {csv_path}: {e}")
        return None, None

    nyq = fs / 2
    h_freq_eff = min(h_freq, nyq * 0.95)
    raw_pp = preprocess(raw, h_freq=h_freq_eff, use_ica=use_ica,
                        subject_label=subject_label, reference=reference)
    epochs = epoch_with_gap_rejection(raw_pp, epoch_len_s, overlap_s,
                                       amplitude_uv=amplitude_uv)
    if len(epochs) == 0:
        if not quiet:
            print(f"  WARNING: {csv_path} produced 0 usable epochs -- skipping")
        return None, fs
    return epochs, fs


# ============================================================
# 2b. CONTINUOUS ARTIFACT DETECTION (default path, no fixed epochs)
# ============================================================
#
# Suggested by Evan (project supervisor, 2026-07-29): fixed-length epochs
# (even the 30s ones from the previous fix) get thrown out ENTIRELY if even
# a small fraction is bad -- e.g. 1s of blink inside a 30s window discards
# 29s of perfectly good data. A short sliding window instead marks and
# removes only the actual bad stretch, in BOTH subjects at the matching
# timepoints (since PLV/circ-corr need paired samples from both brains at
# the same instant), and the connectivity stats run on whatever continuous
# good data is left -- no arbitrary epoch boundaries at all.

def mask_runs(mask):
    """[(start, end_exclusive), ...] for each run of True in a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def robust_threshold(values, k=5.0):
    """median + k * MAD, with MAD scaled to standard-deviation units.

    MAD rather than standard deviation because the artifacts we are trying
    to detect are themselves part of `values`, and would inflate an sd
    estimate enough to push the threshold above the events it is supposed to
    catch. The median/MAD pair is dominated by the quiet majority of the
    recording instead, which is what we want the threshold measured against.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.inf
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + k * 1.4826 * mad


def window_ptp(trace, sfreq, window_s=0.5, step_s=0.1):
    """Sliding-window peak-to-peak. Returns (starts, ends, values)."""
    n = len(trace)
    win_n = max(1, int(round(window_s * sfreq)))
    step_n = max(1, int(round(step_s * sfreq)))
    starts = list(range(0, max(1, n - win_n + 1), step_n))
    last_start = max(0, n - win_n)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    ends, values = [], []
    for s in starts:
        e = min(s + win_n, n)
        seg = trace[s:e]
        ends.append(e)
        values.append(float(seg.max() - seg.min()) if e > s else 0.0)
    return np.array(starts), np.array(ends), np.array(values)


def ptp_event_mask(trace, sfreq, threshold, window_s=0.5, step_s=0.1):
    """Samples covered by any window whose peak-to-peak exceeds threshold."""
    starts, ends, values = window_ptp(trace, sfreq, window_s, step_s)
    bad = np.zeros(len(trace), dtype=bool)
    for s, e, v in zip(starts, ends, values):
        if v > threshold:
            bad[s:e] = True
    return bad


def velocity_event_mask(trace, sfreq, threshold, min_dur_s=0.05,
                        max_dur_s=0.6, merge_gap_s=0.15):
    """
    Blink detector: fast deflection of a plausible duration.

    Thresholds |d/dt| rather than amplitude, then keeps only events lasting
    between min_dur_s and max_dur_s. A blink produces two velocity
    excursions close together (the fast closing and the slower reopening),
    so runs separated by less than merge_gap_s are merged before the
    duration test.

    This is what makes it a BLINK detector rather than a large-thing
    detector: slow drift never exceeds the velocity threshold, and sustained
    muscle tone fails the duration test. Returns (mask, n_kept, n_candidates).
    """
    if len(trace) < 2:
        return np.zeros(len(trace), dtype=bool), 0, 0
    velocity = np.abs(np.diff(trace, prepend=trace[0])) * sfreq  # uV/s
    hot = velocity > threshold
    gap_n = int(round(merge_gap_s * sfreq))

    merged = []
    for s, e in mask_runs(hot):
        if merged and s - merged[-1][1] <= gap_n:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = np.zeros(len(trace), dtype=bool)
    kept = 0
    for s, e in merged:
        if min_dur_s <= (e - s) / sfreq <= max_dur_s:
            out[s:e] = True
            kept += 1
    return out, kept, len(merged)


def ocular_thresholds(oc, window_s=0.5, step_s=0.1, k=5.0,
                      blink_override=None, saccade_override=None):
    """
    Per-participant detection thresholds derived from that subject's own
    ocular channels (point 10). Returns a dict with 'blink'/'saccade'
    amplitude thresholds in uV and 'blink_velocity' in uV/s; entries are
    None where the channel is unavailable. Overrides bypass the estimate.
    """
    out = {"blink": None, "saccade": None, "blink_velocity": None}
    for name, trace, override in (("blink", oc.blink, blink_override),
                                  ("saccade", oc.saccade, saccade_override)):
        if trace is None:
            continue
        if override is not None:
            out[name] = float(override)
        else:
            _, _, values = window_ptp(trace, oc.fs, window_s, step_s)
            out[name] = robust_threshold(values, k)
    if oc.blink is not None:
        velocity = np.abs(np.diff(oc.blink, prepend=oc.blink[0])) * oc.fs
        out["blink_velocity"] = robust_threshold(velocity, k)
    return out


def ocular_bad_mask(oc, thresholds, n_samples, window_s=0.5, step_s=0.1,
                    detector="both", min_blink_s=0.05, max_blink_s=0.6,
                    merge_gap_s=0.15, subject_label="", quiet=False):
    """
    Boolean bad-sample mask built from the derived ocular channels.

    The ptp and velocity criteria are OR-ed: velocity adds blink
    specificity, ptp keeps coverage of the non-blink junk (jaw clench, cable
    tug) that a blink detector would ignore by design. Returns (mask, stats).
    """
    bad = np.zeros(n_samples, dtype=bool)
    stats = {}
    use_ptp = detector in ("ptp", "both")
    use_velocity = detector in ("velocity", "both")

    if use_ptp:
        for name, trace in (("blink", oc.blink), ("saccade", oc.saccade)):
            thr = thresholds.get(name)
            if trace is None or thr is None:
                continue
            m = ptp_event_mask(trace[:n_samples], oc.fs, thr, window_s, step_s)
            bad |= m[:n_samples]
            stats[f"{name}_ptp_pct"] = 100.0 * m.mean()
            if not quiet:
                print(f"     {subject_label}: {name} p2p > {thr:6.1f} uV "
                      f"-> {100 * m.mean():5.1f}% of samples")

    if use_velocity and oc.blink is not None and thresholds.get("blink_velocity"):
        thr = thresholds["blink_velocity"]
        m, kept, cand = velocity_event_mask(
            oc.blink[:n_samples], oc.fs, thr, min_blink_s, max_blink_s,
            merge_gap_s)
        bad |= m[:n_samples]
        stats["blink_velocity_pct"] = 100.0 * m.mean()
        stats["blink_events"] = kept
        if not quiet:
            print(f"     {subject_label}: blink velocity > {thr:7.0f} uV/s "
                  f"-> {kept} blinks kept of {cand} candidates "
                  f"({100 * m.mean():5.1f}% of samples)")

    stats["total_pct"] = 100.0 * bad.mean()
    return bad, stats


def continuous_bad_mask(raw, window_s=0.5, step_s=0.1, threshold_uv=500.0,
                        pad_s=0.3, extra_bad=None, use_eeg_amplitude=True):
    """
    Slide a window across a subject's continuous signal; mark every sample
    covered by a window as bad if any channel's peak-to-peak amplitude in
    that window exceeds threshold_uv. Also folds in existing BAD_gap
    annotations (lost BLE packets), so both artifact types end up in one
    boolean mask over samples.

    use_eeg_amplitude=False skips the EEG peak-to-peak criterion entirely
    (--artifact-source ocular), leaving only extra_bad plus the gap
    annotations. extra_bad is an optional precomputed mask -- in practice
    the ocular-channel detections from ocular_bad_mask() (point 10) -- which
    is OR-ed in before padding, so ocular events get the same filter-ringing
    pad as everything else.

    Each resulting bad run is then padded by pad_s on both sides. This
    matters because the band-pass filter applied to the continuous signal
    (see prefilter_raw_for_band) can ring for a short distance around a
    sharp artifact edge (e.g. a railed/saturated segment) -- without
    padding, that ringing can leak into samples just outside the detected
    bad run and get silently counted as clean data.
    """
    data_uv = raw.get_data(picks="eeg") * 1e6
    sfreq = raw.info["sfreq"]
    n_samples = data_uv.shape[1]
    window_n = max(1, int(round(window_s * sfreq)))
    step_n = max(1, int(round(step_s * sfreq)))

    bad = np.zeros(n_samples, dtype=bool)
    if use_eeg_amplitude:
        starts = list(range(0, max(1, n_samples - window_n + 1), step_n))
        last_start = max(0, n_samples - window_n)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        for start in starts:
            end = min(start + window_n, n_samples)
            seg = data_uv[:, start:end]
            ptp = seg.max(axis=1) - seg.min(axis=1)
            if np.any(ptp > threshold_uv):
                bad[start:end] = True

    # ocular-channel detections (point 10), computed by the caller
    if extra_bad is not None:
        bad[:min(n_samples, len(extra_bad))] |= \
            np.asarray(extra_bad, dtype=bool)[:n_samples]

    for ann in raw.annotations:
        if "BAD" in ann["description"]:
            onset_samp = int(round(ann["onset"] * sfreq))
            dur_samp = int(round(ann["duration"] * sfreq))
            bad[max(0, onset_samp):max(0, onset_samp) + dur_samp] = True

    pad_n = int(round(pad_s * sfreq))
    if pad_n > 0 and bad.any():
        kernel = np.ones(2 * pad_n + 1, dtype=np.uint8)
        bad = np.convolve(bad.astype(np.uint8), kernel, mode="same") > 0

    return bad


def load_and_preprocess_continuous(csv_path, subject_label, h_freq=40.0,
                                    use_ica=False, align_onset=True, quiet=False,
                                    artifact_window=0.5, artifact_step=0.1,
                                    artifact_threshold=500.0, artifact_pad=0.3,
                                    reference="average"):
    """
    Full load -> preprocess chain for ONE subject's CSV, for the default
    continuous (non-epoched) connectivity path.

    Returns (raw_pp, bad_mask, fs) or (None, None, None) if the file
    couldn't be used.
    """
    try:
        onset = load_stimulus_onset(csv_path) if align_onset else None
        raw, fs = load_csv_to_raw(csv_path, subject_label, onset_s=onset)
    except Exception as e:
        if not quiet:
            print(f"  WARNING: could not load {csv_path}: {e}")
        return None, None, None

    nyq = fs / 2
    h_freq_eff = min(h_freq, nyq * 0.95)
    raw_pp = preprocess(raw, h_freq=h_freq_eff, use_ica=use_ica,
                        subject_label=subject_label, reference=reference)
    bad_mask = continuous_bad_mask(raw_pp, window_s=artifact_window,
                                    step_s=artifact_step,
                                    threshold_uv=artifact_threshold,
                                    pad_s=artifact_pad)
    return raw_pp, bad_mask, fs


# ============================================================
# 3. CONNECTIVITY (PLV / circular correlation)
# ============================================================

def prefilter_raw_for_band(raw, band, order=4):
    """
    Band-pass the CONTINUOUS raw signal into `band` BEFORE epoching.

    Filtering the full-length continuous signal means any filter edge/
    transient effects only happen once (at the very start/end of the whole
    recording), instead of once per 2-second epoch. This matters most for
    narrow bands (e.g. a 1 Hz-wide SSVEP band) where a short epoch may be
    only a few filter time-constants long, and per-epoch filtering can
    inflate phase consistency across all epochs independent of any real
    coupling.
    """
    raw_band = raw.copy()
    raw_band.filter(l_freq=band[0], h_freq=band[1], picks="eeg",
                     method="iir",
                     iir_params={"order": order, "ftype": "butter"},
                     verbose=False)
    return raw_band


# ============================================================
# 2c. CROSS-DEVICE TIMING SYNC-CORRECTION (optional, --sync-signal-hz)
# ============================================================
#
# Two independent Muses stream over BLE, and LSL dejitters each stream onto
# a nominal 256 Hz grid -- which hides any small difference between the two
# devices' crystal-clock rates. That difference shows up as a slow, HIDDEN
# drift in the sample-by-sample alignment between the two recordings (tens
# of ms over a several-minute session), which erodes inter-brain phase
# coupling, worst in high-frequency bands (beta). It can only pull real
# coupling DOWN toward the null -- it never manufactures coupling -- so it
# is a sensitivity ceiling, not a false-positive risk.
#
# If BOTH subjects were exposed to the SAME external periodic driver (e.g. a
# shared flicker at --sync-signal-hz, picked up as an entrained SSVEP, or a
# photodiode/trigger channel), that shared signal is a timing ruler: both
# brains are driven by one clock, so their phase difference AT THE DRIVER
# FREQUENCY should be constant if the devices are aligned. A linear ramp in
# that phase difference over time IS the clock drift, and its slope gives
# the exact rate correction.
#
# IMPORTANT LIMITATION: at a SINGLE frequency, a constant time offset is
# indistinguishable from a constant NEURAL phase lag between the two
# subjects' responses (and wraps every 1/f seconds). So the DRIFT (slope)
# is unambiguous and safe to correct; the constant OFFSET (intercept) is
# confounded and is only corrected when explicitly requested (not
# --sync-drift-only), and even then is best sanity-checked against a
# positive control. To disambiguate the constant offset properly you need
# two driver frequencies (frequency-tagging) or a broadband shared trigger.

def _sync_reference_phase(raw, band):
    """Instantaneous phase of the strongest shared-driver component: pick
    the channel with the most power in `band`, band-pass, Hilbert."""
    from scipy.signal import hilbert
    raw_band = prefilter_raw_for_band(raw, band)
    data = raw_band.get_data(picks="eeg")
    idx = int(np.argmax(data.std(axis=1)))
    return np.angle(hilbert(data[idx]))


def estimate_sync_offset_drift(raw_a, raw_b, sync_hz, bandwidth=1.0, drift_only=False):
    """
    Estimate the cross-device timing offset (s) and drift rate (dimensionless
    s/s) between two recordings, from a shared periodic driver at sync_hz.

    Fits phase_a - phase_b = -2*pi*sync_hz*(offset + drift*t) over time:
      intercept -> constant offset,  slope -> drift rate.
    If drift_only, the constant offset is forced to 0 (only drift, the
    unambiguous part, is returned -- see the section note above).

    Returns (offset_s, drift_rate).
    """
    band = (sync_hz - bandwidth / 2.0, sync_hz + bandwidth / 2.0)
    fs = raw_a.info["sfreq"]
    pa = _sync_reference_phase(raw_a, band)
    pb = _sync_reference_phase(raw_b, band)
    n = min(len(pa), len(pb))
    t = np.arange(n) / fs
    dphi = np.unwrap(pa[:n] - pb[:n])
    w = 2.0 * np.pi * sync_hz
    if drift_only:
        slope = np.polyfit(t, dphi, 1)[0]
        return 0.0, -slope / w
    slope, intercept = np.polyfit(t, dphi, 1)
    return -intercept / w, -slope / w


def apply_sync_correction(raw_b, offset_s, drift_rate):
    """
    Resample raw_b onto raw_a's timeline given an estimated constant offset
    and linear drift (see estimate_sync_offset_drift). Sample i of the
    corrected signal is raw_b interpolated at grid time
    (t_i - offset)/(1 + drift). Returns a corrected copy; length unchanged.
    """
    fs = raw_b.info["sfreq"]
    data = raw_b.get_data()
    n = data.shape[1]
    grid = np.arange(n)
    src_idx = ((grid / fs - offset_s) / (1.0 + drift_rate)) * fs
    corrected = np.vstack([np.interp(src_idx, grid, data[ch]) for ch in range(data.shape[0])])
    raw_out = raw_b.copy()
    raw_out._data[:] = corrected
    return raw_out


def _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode, sfreq,
                               already_filtered=False):
    """
    Compute an inter-brain connectivity block with HyPyP.

    If already_filtered=True, the epochs' data is assumed to already be
    band-passed (via prefilter_raw_for_band on the continuous raw signal
    before epoching), and HyPyP is told not to filter again
    (filter_signal=False) -- it will just compute the analytic (Hilbert)
    signal and the sync measure. If already_filtered=False (old default
    behaviour), HyPyP band-passes each short epoch independently.
    """
    n_ep = min(len(epochs_a), len(epochs_b))
    if n_ep == 0:
        return None

    data = np.array([
        epochs_a.get_data()[:n_ep],
        epochs_b.get_data()[:n_ep],
    ])
    freq_bands = {"band": list(band)}
    complex_signal = analyses.compute_freq_bands(
        data=data,
        sampling_rate=int(round(float(sfreq))),
        freq_bands=freq_bands,
        filter_signal=not already_filtered,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
    )
    con = analyses.compute_sync(
        complex_signal,
        mode=mode,
        epochs_average=True,
    )
    # HyPyP returns (n_freq, 2*n_channels, 2*n_channels) when epochs_average=True.
    # Derive the block size from the data rather than from CH_NAMES, so the
    # slice follows --analysis-channels (point 11) instead of always assuming
    # the full 4-electrode montage.
    con = np.asarray(con)[0]
    n_chan = data.shape[2]
    return con[:n_chan, n_chan:2 * n_chan]


def plv_hypyp(epochs_a, epochs_b, band, sfreq, already_filtered=False):
    """
    Compute PLV between every channel pair (one from A, one from B) using HyPyP.

    Returns: (n_chan_a, n_chan_b) matrix of PLVs averaged across epochs.
    """
    return _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode="plv",
                                      sfreq=sfreq, already_filtered=already_filtered)


def circular_corr_hypyp(epochs_a, epochs_b, band, sfreq, already_filtered=False):
    """
    Adjusted circular correlation (ACCorr) between every channel pair, via HyPyP.

    Unlike PLV (which collapses phase difference to a magnitude), ACCorr
    preserves sign: positive = in-phase-leaning, negative = anti-phase-leaning,
    0 = no consistent relationship. Per-pair phase centering (rather than one
    global circular mean) gives a more accurate estimate than plain 'ccorr',
    which HyPyP computes by averaging abs(r) per epoch and therefore can never
    return a negative value.

    Zimmermann et al. (2024), Imaging Neuroscience, 2.

    Returns: (n_chan_a, n_chan_b) matrix, signed, or None if no epochs.
    """
    return _hyyp_connectivity_matrix(epochs_a, epochs_b, band, mode="accorr",
                                      sfreq=sfreq, already_filtered=already_filtered)


def surrogate_distribution(epochs_a, epochs_b, band, n_surrogates, metric_fn, sfreq,
                            already_filtered=False, seed=0):
    """
    WITHIN-DYAD null: shuffle epoch order between subjects A and B.

    If the real value falls outside this distribution, the coupling is unlikely
    to be due to chance ALONE -- but note the important caveat: this only tests
    whether coupling varies meaningfully from epoch to epoch. For a continuous,
    non-varying stimulus (e.g. a flicker running the whole recording), shuffling
    epoch order barely changes anything, and this test will under-detect real
    stimulus-locked signal. Use pseudo_pair_distribution() (cross-dyad) for a
    more appropriate test of stimulus-locked / stationary coupling.
    """
    rng = np.random.default_rng(seed)
    n_ep = min(len(epochs_a), len(epochs_b))
    nulls = []
    for k in range(n_surrogates):
        perm = rng.permutation(n_ep)
        # shuffle B's epochs
        epochs_b_shuffled = epochs_b.copy()
        epochs_b_shuffled._data = epochs_b_shuffled._data[perm]
        nulls.append(metric_fn(epochs_a, epochs_b_shuffled, band, sfreq,
                                already_filtered=already_filtered))
    return np.array(nulls)  # shape (n_surrogates, n_chan_a, n_chan_b)


def pseudo_pair_distribution(target_epochs, pool_epochs_list, band, metric_fn, sfreq,
                              already_filtered=False, shuffles_per_pool_member=1,
                              seed=0):
    """
    CROSS-DYAD null: compare `target_epochs` (one real subject) against
    epochs from OTHER people who were never in the same session with them
    (the --pool-dir recordings).

    This is the standard hyperscanning "pseudo-pair" validity check: if a
    real dyad's coupling is no higher than what you get pairing that person
    with random strangers who happened to be exposed to a similar stimulus,
    the "coupling" is not evidence of a genuine dyad-specific effect -- it's
    just shared, stimulus-driven, but independent, brain activity.

    Especially important for stationary/continuous stimuli (e.g. SSVEP),
    where within-dyad epoch shuffling (see surrogate_distribution) can't
    tell real coupling apart from shared independent entrainment, because
    shuffling doesn't change the stimulus each epoch is locked to.

    Returns: array shape (n_pool * shuffles_per_pool_member, n_chan, n_chan)
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for pool_epochs in pool_epochs_list:
        n_ep = min(len(target_epochs), len(pool_epochs))
        if n_ep == 0:
            continue
        for _ in range(shuffles_per_pool_member):
            perm = rng.permutation(len(pool_epochs))[:n_ep]
            pool_shuffled = pool_epochs.copy()
            pool_shuffled._data = pool_shuffled._data[perm]
            val = metric_fn(target_epochs, pool_shuffled, band, sfreq,
                             already_filtered=already_filtered)
            if val is not None:
                nulls.append(val)
    if not nulls:
        return None
    return np.array(nulls)


# ============================================================
# 3a2. CONTINUOUS (MASKED) CONNECTIVITY -- default path, no fixed epochs
# ============================================================
#
# Companion to section 2b: instead of chopping the recording into fixed
# epochs and rejecting whole ones, the Hilbert transform is computed ONCE
# on the FULL continuous band-passed recording (so there's no splicing
# discontinuity anywhere), and only afterward do we select just the
# samples that are clean in BOTH subjects for the actual PLV / circular
# correlation sum. This bypasses HyPyP's epoch-based API (which has no way
# to mask out individual timepoints within an epoch), so PLV and the
# circular correlation coefficient are computed directly here instead.
#
# TWO circular correlation variants are provided:
#   circ_corr_masked()           classic Fisher & Lee (1983); assumes a
#                                 well-defined circular mean per channel.
#   circ_corr_adjusted_masked()  bias-adjusted version for arbitrary/not-
#                                 well-defined mean directions (Jammalamadaka
#                                 & Sengupta, 2001, p.177), the same
#                                 correction Zimmermann et al. (2024) use and
#                                 recommend for continuous EEG, and the same
#                                 one HyPyP's 'accorr' mode applies in the
#                                 --legacy-epochs path. This is now the
#                                 default (see --circ-corr-method).

def analytic_signal(raw_band, n_samples=None):
    """Hilbert transform of a band-passed CONTINUOUS Raw -> complex analytic
    signal per channel, shape (n_channels, n_times)."""
    from scipy.signal import hilbert
    data = raw_band.get_data(picks="eeg")
    if n_samples is not None:
        data = data[:, :n_samples]
    return hilbert(data, axis=-1)


def plv_masked(analytic_a, analytic_b, good_mask):
    """PLV between every channel pair, summed over good_mask samples only.
    Returns (n_chan_a, n_chan_b)."""
    phase_a = np.angle(analytic_a[:, good_mask])
    phase_b = np.angle(analytic_b[:, good_mask])
    diff = phase_a[:, None, :] - phase_b[None, :, :]
    return np.abs(np.mean(np.exp(1j * diff), axis=-1))


def circ_corr_masked(analytic_a, analytic_b, good_mask):
    """Classic Fisher & Lee (1983) circular correlation coefficient between
    every channel pair, summed over good_mask samples only. Signed:
    positive = in-phase-leaning, negative = anti-phase-leaning. Assumes each
    channel has a well-defined circular mean -- for continuous EEG this is
    often NOT the case (see circ_corr_adjusted_masked below and Zimmermann
    et al., 2024). Returns (n_chan_a, n_chan_b)."""
    phase_a = np.angle(analytic_a[:, good_mask])
    phase_b = np.angle(analytic_b[:, good_mask])
    mean_a = np.angle(np.mean(np.exp(1j * phase_a), axis=-1))
    mean_b = np.angle(np.mean(np.exp(1j * phase_b), axis=-1))
    sin_a = np.sin(phase_a - mean_a[:, None])
    sin_b = np.sin(phase_b - mean_b[:, None])
    num = np.einsum("it,jt->ij", sin_a, sin_b)
    den = np.sqrt(np.sum(sin_a ** 2, axis=-1)[:, None] * np.sum(sin_b ** 2, axis=-1)[None, :])
    return num / den


def circ_corr_adjusted_masked(analytic_a, analytic_b, good_mask):
    """
    Bias-adjusted circular correlation coefficient between every channel
    pair, summed over good_mask samples only. Signed, same sign convention
    as circ_corr_masked.

    Implements Jammalamadaka & Sengupta (2001, p.177), the formula
    Zimmermann et al. (2024) recommend for continuous EEG data (whose
    circular mean, unlike directional data such as wind bearings, is not
    well-defined over an arbitrary analysis window). This is the exact
    formula used by pingouin.circ_corrcc(correction_uniform=True) and by
    HyPyP's 'accorr' mode (used in the --legacy-epochs path), so this
    continuous-path implementation is now consistent with both:

        r_minus = |sum_t exp(i*(phase_a_t - phase_b_t))|
        r_plus  = |sum_t exp(i*(phase_a_t + phase_b_t))|
        denom   = 2 * sqrt(sum(sin(phase_a - mean_a)^2)
                            * sum(sin(phase_b - mean_b)^2))
        r_adj   = (r_minus - r_plus) / denom

    Verified against pingouin's scalar implementation to float precision
    before being vectorized here for the (n_chan_a, n_chan_b) pairwise case.

    Returns (n_chan_a, n_chan_b).
    """
    phase_a = np.angle(analytic_a[:, good_mask])  # (n_chan_a, n_t)
    phase_b = np.angle(analytic_b[:, good_mask])  # (n_chan_b, n_t)

    mean_a = np.angle(np.sum(np.exp(1j * phase_a), axis=-1))
    mean_b = np.angle(np.sum(np.exp(1j * phase_b), axis=-1))
    sin_a = np.sin(phase_a - mean_a[:, None])
    sin_b = np.sin(phase_b - mean_b[:, None])

    ea = np.exp(1j * phase_a)  # (n_chan_a, n_t)
    eb = np.exp(1j * phase_b)  # (n_chan_b, n_t)
    r_minus = np.abs(np.einsum("it,jt->ij", ea, np.conj(eb)))
    r_plus = np.abs(np.einsum("it,jt->ij", ea, eb))

    denom = 2 * np.sqrt(np.sum(sin_a ** 2, axis=-1)[:, None]
                         * np.sum(sin_b ** 2, axis=-1)[None, :])
    return (r_minus - r_plus) / denom


def subsample_good_mask(good, target_n, rng):
    """
    Randomly select exactly target_n True positions from a boolean mask,
    returning a new boolean mask of the same length with only those
    positions set. If good.sum() <= target_n, returns good unchanged
    (nothing to trim).

    Used to length-match the real dyad's (often much larger) good_mask, or
    an individual null draw's good_mask, down to a common sample count so
    PLV/circular-correlation comparisons aren't confounded by different
    N's -- short windows are known to inflate these metrics for weak/no
    coupling (Zimmermann et al., 2024), so comparing a long real estimate
    against a null built from shorter draws (or vice versa) is not a fair
    like-for-like test. Order doesn't matter for PLV/circular correlation
    (both are order-independent sums over samples), so random subsampling
    is valid here -- no need to preserve contiguous runs.
    """
    n_good = int(good.sum())
    if n_good <= target_n:
        return good
    idx = np.where(good)[0]
    chosen = rng.choice(idx, size=target_n, replace=False)
    out = np.zeros_like(good)
    out[chosen] = True
    return out


def circular_shift_surrogates_continuous(analytic_a, bad_a, analytic_b, bad_b,
                                          n_surrogates, metric_fn, seed=0,
                                          target_n=None, rng=None):
    """
    WITHIN-DYAD null for the continuous path: circularly time-shift subject
    B's ENTIRE analytic signal (and its own bad-mask) by a random amount,
    wrapping at the recording boundary, then recompute the joint good_mask
    against A's (unshifted) bad-mask. This is the standard continuous-
    signal surrogate method -- it preserves each subject's own temporal/
    spectral structure while destroying true moment-to-moment alignment
    between the two brains, playing the same role the old epoch-shuffle
    surrogate did before fixed epochs were removed.

    If target_n is given, only draws with at least target_n jointly-clean
    samples are kept, and each surviving draw's good_mask is randomly
    subsampled down to exactly target_n samples (see subsample_good_mask)
    before computing the metric. That makes the null comparison exact: every
    included draw is evaluated at the same N.

    Returns (nulls, ns): nulls is an array of per-draw metric matrices (or
    None if every draw had zero usable samples), ns is an array of the
    actual per-draw sample count used (before any target_n floor is hit,
    this equals the natural overlap size; useful for diagnosing N
    mismatches even when target_n/matching is off).
    """
    rng = rng or np.random.default_rng(seed)
    n = analytic_b.shape[-1]
    nulls = []
    ns = []
    for _ in range(n_surrogates):
        shift = int(rng.integers(1, n))
        b_shifted = np.roll(analytic_b, shift, axis=-1)
        bad_b_shifted = np.roll(bad_b, shift)
        good = ~(bad_a | bad_b_shifted)
        ns.append(int(good.sum()))
        if target_n is not None:
            if good.sum() < target_n:
                continue
            good = subsample_good_mask(good, target_n, rng)
        if good.sum() == 0:
            continue
        nulls.append(metric_fn(analytic_a, b_shifted, good))
    return (np.array(nulls) if nulls else None), np.array(ns)


def pseudo_pair_continuous(target_analytic, target_bad, pool_analytic_list, pool_bad_list,
                            metric_fn, shuffles_per_pool_member=3, seed=0,
                            target_n=None, rng=None):
    """
    CROSS-DYAD null for the continuous path: compare the target subject's
    continuous analytic signal against OTHER subjects' (--pool-dir)
    continuous analytic signal, at randomly shifted alignments, keeping
    only jointly-clean samples. See pseudo_pair_distribution() (legacy
    epoch-based version) for the rationale.

    Each pool member has their OWN bad-mask from their OWN recording
    session, so the size of the jointly-clean overlap after intersecting
    with the target's bad-mask varies draw to draw, and can be much smaller
    than the target's own (often much longer) good_mask. If target_n is
    given, only draws with at least target_n jointly-clean samples are kept,
    and each surviving draw is randomly subsampled down to exactly target_n
    samples before computing the metric (see subsample_good_mask), so the
    resulting null is length-matched and not biased by short-window PLV/
    circ-corr inflation relative to a longer real-dyad estimate.

    Returns (nulls, ns): nulls has shape (n_pool * shuffles_per_pool_member,
    n_chan, n_chan) or None if no pool member overlapped with clean data;
    ns is the array of actual per-draw sample counts (pre-target_n, so it
    reflects the natural overlap size for diagnostics even with matching on).
    """
    rng = rng or np.random.default_rng(seed)
    nulls = []
    ns = []
    for pool_analytic, pool_bad in zip(pool_analytic_list, pool_bad_list):
        n = min(target_analytic.shape[-1], pool_analytic.shape[-1])
        if n == 0:
            continue
        t_analytic = target_analytic[:, :n]
        t_bad = target_bad[:n]
        for _ in range(shuffles_per_pool_member):
            shift = int(rng.integers(0, pool_analytic.shape[-1]))
            p_analytic = np.roll(pool_analytic, shift, axis=-1)[:, :n]
            p_bad = np.roll(pool_bad, shift)[:n]
            good = ~(t_bad | p_bad)
            ns.append(int(good.sum()))
            if target_n is not None:
                if good.sum() < target_n:
                    continue
                good = subsample_good_mask(good, target_n, rng)
            if good.sum() == 0:
                continue
            nulls.append(metric_fn(t_analytic, p_analytic, good))
    if not nulls:
        return None, np.array(ns)
    return np.array(nulls), np.array(ns)


def matched_observed_value(analytic_a, analytic_b, good_mask, target_n, metric_fn,
                            n_draws=5, seed=0):
    """
    Recompute the observed PLV/circ-corr value on the real dyad, averaged
    over n_draws random subsamples of good_mask down to exactly target_n
    samples each (see subsample_good_mask).

    This is used ONLY for the null-comparison/p-value, so that the real
    value being compared against the null is at the SAME sample count as
    the null draws -- not for the headline PLV/circ-r number, which should
    still use ALL available clean data (lower variance, better point
    estimate). Averaging over several subsamples reduces the extra
    sampling noise introduced by only using target_n < the full available N.

    Returns the mean matrix over the n_draws subsamples.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        sub_mask = subsample_good_mask(good_mask, target_n, rng)
        if sub_mask.sum() == 0:
            continue
        draws.append(metric_fn(analytic_a, analytic_b, sub_mask))
    if not draws:
        return None
    return np.mean(draws, axis=0)


# ============================================================
# 3b. MULTIPLE-COMPARISONS CORRECTION
# ============================================================

def fdr_bh(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Takes a flat array of p-values, returns (reject_mask, corrected_pvals)
    both the same shape as the input. Used instead of raw per-pair p<0.05
    counting: with up to 16 channel-pair tests per band at uncorrected p<0.05,
    ~0.8 false positives are expected per band by chance alone, so raw
    counts of "1/16" or "2/16" significant pairs are not meaningfully
    different from noise. FDR correction controls the expected proportion
    of false discoveries among the pairs called significant.
    """
    pvals = np.asarray(pvals, dtype=float)
    shape = pvals.shape
    flat = pvals.ravel()
    n = len(flat)
    order = np.argsort(flat)
    ranked = flat[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    if below.any():
        max_i = np.max(np.where(below)[0])
        cutoff = ranked[max_i]
    else:
        cutoff = -1.0  # nothing survives
    reject_flat = flat <= cutoff
    # corrected p-values (BH step-up), monotone
    corrected = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        corrected[i] = val
        prev = val
    corrected_full = np.empty(n)
    corrected_full[order] = corrected
    reject_full = np.empty(n, dtype=bool)
    reject_full[order] = reject_flat
    return reject_full.reshape(shape), corrected_full.reshape(shape)


def summarize_positive_control(observed, null, pvalues, sig_mask=None, alpha=0.05):
    """
    Turn a stimulus-band pseudo-pair comparison into a plain-language verdict.

    A positive control is considered a pass when the observed stimulus-band
    effect exceeds the pool baseline and at least one pair is significant
    after the chosen multiple-comparisons correction.
    """
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    if sig_mask is None:
        sig_mask = pvalues < alpha
    n_sig = int(np.count_nonzero(sig_mask))
    obs_mean = float(observed.mean())
    null_mean = float(null.mean())
    above_null = obs_mean > null_mean
    passed = above_null and n_sig >= 1

    if passed:
        status = "PASS"
        reason = (
            f"stimulus-band PLV {obs_mean:.3f} exceeded the pool baseline "
            f"{null_mean:.3f} and {n_sig} pair(s) were significant"
        )
    elif above_null:
        status = "WEAK"
        reason = (
            f"stimulus-band PLV {obs_mean:.3f} exceeded the pool baseline "
            f"{null_mean:.3f}, but no pairs reached the significance threshold"
        )
    else:
        status = "FAIL"
        reason = (
            f"stimulus-band PLV {obs_mean:.3f} did not exceed the pool baseline "
            f"{null_mean:.3f}"
        )
    return {
        "status": status,
        "passed": passed,
        "observed_mean": obs_mean,
        "null_mean": null_mean,
        "n_sig_pairs": n_sig,
        "reason": reason,
    }


def resolve_pool_csvs(pool_dir, csv_a, csv_b):
    """
    Resolve a pool of CSV recordings for pseudo-pair validation.

    Prefer the explicitly supplied pool directory. If that directory is empty
    or missing, fall back to the parent directory of the two real recordings,
    which is often the most practical place for a pilot dataset.
    """
    candidates = []
    if pool_dir:
        candidates.append(pool_dir)
    for path in (csv_a, csv_b):
        parent = os.path.dirname(os.path.abspath(path))
        if parent not in candidates:
            candidates.append(parent)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        abs_candidate = os.path.abspath(candidate)
        if not os.path.isdir(abs_candidate):
            continue
        files = sorted(glob.glob(os.path.join(abs_candidate, "*.csv")))
        if files:
            return files
    return []


# ============================================================
# 4. PLOTS
# ============================================================

def plot_raw_with_gaps(raw_a, raw_b, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, raw, label in zip(axes, [raw_a, raw_b], ["A", "B"]):
        data = raw.get_data() * 1e6  # back to microvolts
        t = np.arange(data.shape[1]) / raw.info["sfreq"]
        for i, ch in enumerate(raw.ch_names):
            ax.plot(t, data[i] + i * 100, lw=0.5, label=ch.split("_")[-1])
        # shade gap annotations
        for ann in raw.annotations:
            if "BAD" in ann["description"]:
                ax.axvspan(ann["onset"], ann["onset"] + ann["duration"],
                           color="red", alpha=0.15, lw=0)
        ax.set_ylabel(f"Subject {label}\n(uV, offset)")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw signal with BAD_gap segments highlighted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_psd(raw_a, raw_b, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, raw, label in zip(axes, [raw_a, raw_b], ["A", "B"]):
        fmax = min(40.0, raw.info["sfreq"] / 2 * 0.99)
        psd = raw.compute_psd(fmin=1, fmax=fmax, verbose=False)
        psd.plot(axes=ax, show=False)
        ax.set_title(f"Subject {label}")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ocular_channels(oc_a, oc_b, out_path):
    """
    QC figure for the derived ocular channels (point 9), one row per subject.

    What to look for: blinks should appear on the BLINK trace as large,
    same-shape deflections a few hundred ms wide, and should be much smaller
    on the SACCADE trace; horizontal eye movements should do the opposite.
    If both traces look identical, the frontal electrodes are probably not
    picking up lateral differences and the saccade channel is not
    trustworthy for this subject.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 6.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    colors = {"blink": "#0072B2", "saccade": "#D55E00"}
    zoom_s = 8.0
    for (ax_full, ax_zoom), oc in zip(axes, (oc_a, oc_b)):
        traces = [(n, t) for n, t in (("blink", oc.blink),
                                      ("saccade", oc.saccade)) if t is not None]
        if not traces:
            ax_full.text(0.5, 0.5, f"Subject {oc.label}: no ocular channels "
                                   "available (electrodes flagged bad)",
                         ha="center", va="center", transform=ax_full.transAxes,
                         fontsize=11, color="#888")
            for ax in (ax_full, ax_zoom):
                ax.set_yticks([])
                ax.set_xticks([])
            continue
        # stack the traces with a shared, robust offset so a single huge
        # artifact cannot squash the other trace flat
        scale = max(np.percentile(np.abs(t), 99) for _, t in traces) or 1.0
        step = 4.0 * scale
        time = np.arange(len(traces[0][1])) / oc.fs

        # Zoom on the single largest excursion of the primary trace. At full
        # length a 300ms blink is about one pixel wide, so the overview
        # cannot show morphology -- and morphology is the whole point of
        # eyeballing these channels.
        primary = dict(traces).get("blink", traces[0][1])
        half_n = int(0.5 * zoom_s * oc.fs)
        peak = int(np.argmax(np.abs(primary)))
        z0 = max(0, min(peak - half_n, len(primary) - 2 * half_n))
        z1 = min(len(primary), z0 + 2 * half_n)

        for ax, (lo, hi) in ((ax_full, (0, len(time))), (ax_zoom, (z0, z1))):
            for i, (name, trace) in enumerate(traces):
                ax.plot(time[lo:hi], trace[lo:hi] + i * step, lw=0.6,
                        color=colors[name])
            ax.set_yticks([i * step for i in range(len(traces))])
            ax.set_yticklabels([n for n, _ in traces])
            ax.margins(x=0)
        ax_full.axvspan(time[z0], time[max(z0, z1 - 1)], color="#000",
                        alpha=0.08, lw=0)
        ax_full.set_ylabel(f"Subject {oc.label}")
        ax_full.set_title(
            f"Subject {oc.label}  ({oc.l_freq:g}-{oc.h_freq:g} Hz)   "
            + "   ".join(f"{n}: std={t.std():.0f} uV" for n, t in traces),
            fontsize=9, loc="left")
        ax_zoom.set_title(f"zoom: {zoom_s:g}s around the largest excursion "
                          f"(t={time[peak]:.1f}s)", fontsize=9, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Derived ocular channels   "
                 "saccade = AF7 - AF8,   blink = mean(AF7, AF8) - mean(TP9, TP10)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_plv_matrix(plv, band_name, out_path, surrogate_p=None, sig_mask=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(plv, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ANALYSIS_CH)))
    ax.set_yticks(range(len(ANALYSIS_CH)))
    ax.set_xticklabels([f"B:{c}" for c in ANALYSIS_CH])
    ax.set_yticklabels([f"A:{c}" for c in ANALYSIS_CH])
    ax.set_title(f"Inter-brain PLV -- {band_name}")
    plt.colorbar(im, ax=ax, label="PLV")
    for i in range(plv.shape[0]):
        for j in range(plv.shape[1]):
            val = plv[i, j]
            star = ""
            if sig_mask is not None and sig_mask[i, j]:
                star = "*"
            elif surrogate_p is not None and surrogate_p[i, j] < 0.05:
                star = "(*)"  # uncorrected only
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_circ_corr_matrix(cc, band_name, out_path, surrogate_p=None, sig_mask=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ANALYSIS_CH)))
    ax.set_yticks(range(len(ANALYSIS_CH)))
    ax.set_xticklabels([f"B:{c}" for c in ANALYSIS_CH])
    ax.set_yticklabels([f"A:{c}" for c in ANALYSIS_CH])
    ax.set_title(f"Inter-brain Circular Corr -- {band_name}")
    plt.colorbar(im, ax=ax, label="r (circ)")
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            val = cc[i, j]
            star = ""
            if sig_mask is not None and sig_mask[i, j]:
                star = "*"
            elif surrogate_p is not None and surrogate_p[i, j] < 0.05:
                star = "(*)"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_circ_corr_comparison(ccs_by_band, out_path):
    fig, axes = plt.subplots(1, len(ccs_by_band), figsize=(4 * len(ccs_by_band), 4.5))
    if len(ccs_by_band) == 1:
        axes = [axes]
    for ax, (band, cc) in zip(axes, ccs_by_band.items()):
        im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(ANALYSIS_CH)))
        ax.set_yticks(range(len(ANALYSIS_CH)))
        ax.set_xticklabels([f"B:{c}" for c in ANALYSIS_CH], rotation=45, ha="right")
        ax.set_yticklabels([f"A:{c}" for c in ANALYSIS_CH])
        ax.set_title(f"{band}  (mean={cc.mean():.2f})")
    fig.suptitle("Inter-brain Circular Correlation across frequency bands")
    plt.colorbar(im, ax=axes, shrink=0.8, label="r (circ)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_plv_comparison(plvs_by_band, out_path):
    fig, axes = plt.subplots(1, len(plvs_by_band), figsize=(4 * len(plvs_by_band), 4.5))
    if len(plvs_by_band) == 1:
        axes = [axes]
    for ax, (band, plv) in zip(axes, plvs_by_band.items()):
        im = ax.imshow(plv, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ANALYSIS_CH)))
        ax.set_yticks(range(len(ANALYSIS_CH)))
        ax.set_xticklabels([f"B:{c}" for c in ANALYSIS_CH], rotation=45, ha="right")
        ax.set_yticklabels([f"A:{c}" for c in ANALYSIS_CH])
        ax.set_title(f"{band}  (mean={plv.mean():.2f})")
    fig.suptitle("Inter-brain PLV across frequency bands")
    plt.colorbar(im, ax=axes, shrink=0.8, label="PLV")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    p.add_argument("--legacy-epochs", action="store_true",
                   help="use the OLD fixed-length-epoch pipeline (see "
                        "--epoch-len/--epoch-overlap/--amplitude-threshold "
                        "below) instead of the default continuous sliding-"
                        "window artifact rejection (see --artifact-window/"
                        "--artifact-step/--artifact-threshold). Fixed epochs "
                        "throw out the WHOLE epoch if even a small fraction "
                        "is bad (e.g. 1s of blink inside a 30s window wastes "
                        "29s of good data) -- kept only for reproducing/"
                        "comparing against older runs.")
    p.add_argument("--artifact-window", type=float, default=0.5,
                   help="(continuous/default path only) sliding window "
                        "length in seconds for artifact detection (default "
                        "0.5). A window is marked bad if any channel's "
                        "peak-to-peak amplitude inside it exceeds "
                        "--artifact-threshold.")
    p.add_argument("--artifact-step", type=float, default=0.1,
                   help="(continuous/default path only) step size in "
                        "seconds between sliding artifact-detection windows "
                        "(default 0.1).")
    p.add_argument("--artifact-threshold", type=float, default=500.0,
                   help="(continuous/default path only) peak-to-peak "
                        "amplitude threshold in uV for the sliding-window "
                        "artifact detector (default 500).")
    p.add_argument("--artifact-pad", type=float, default=0.3,
                   help="(continuous/default path only) seconds of margin "
                        "added on both sides of every detected bad run "
                        "before it's excluded (default 0.3), to absorb "
                        "band-pass filter ringing around artifact edges "
                        "(e.g. a railed/saturated segment) that would "
                        "otherwise leak into neighboring samples still "
                        "counted as clean. 0 = disabled (old behaviour).")
    p.add_argument("--sync-signal-hz", type=float, default=None,
                   help="(continuous/default path only) frequency (Hz) of a "
                        "shared external driver both subjects were exposed to "
                        "(e.g. a 6 Hz flicker seen as SSVEP, or a photodiode/"
                        "trigger channel). Used as a cross-device timing "
                        "ruler: the two Muses' clocks drift relative to each "
                        "other (hidden by LSL's nominal-rate dejittering), and "
                        "that drift erodes inter-brain phase coupling. The "
                        "shared driver's phase difference over time measures "
                        "and corrects it. Off by default. See --sync-drift-only.")
    p.add_argument("--sync-bandwidth", type=float, default=1.0,
                   help="full width in Hz of the band around --sync-signal-hz "
                        "used to extract the shared driver (default 1.0).")
    p.add_argument("--sync-drift-only", action="store_true",
                   help="correct only the cross-device DRIFT (slope), not the "
                        "constant offset. At a single driver frequency a "
                        "constant time offset is confounded with a genuine "
                        "neural phase lag (and wraps every 1/f s), so the "
                        "drift is the unambiguous, always-safe part to "
                        "correct; the constant offset is applied only when "
                        "this flag is absent.")
    p.add_argument("--epoch-len", type=float, default=30.0,
                   help="(--legacy-epochs only) epoch length in seconds "
                        "(default 30.0). Short epochs (e.g. 2s) inflate "
                        "PLV/circ-corr -- Zimmermann et al. (2024) and "
                        "Cassioli et al. (2025) report values stabilize "
                        "only past ~5-6s, with ~55s (+20%% for artifact "
                        "loss) needed for adequately powered detection of a "
                        "moderate effect at 256 Hz. Use --epoch-len 2.0 to "
                        "reproduce the old (inflated) behaviour for "
                        "comparison.")
    p.add_argument("--epoch-overlap", type=float, default=0.0,
                   help="(--legacy-epochs only) epoch overlap in seconds "
                        "(default 0.0, i.e. non-overlapping). Overlapping "
                        "long epochs share most of their data and are not "
                        "independent samples for epoch-averaging or "
                        "surrogate shuffling.")
    p.add_argument("--bands", nargs="+", default=list(FREQ_BANDS.keys()),
                   help="which bands to compute (theta alpha beta)")
    p.add_argument("--stim-hz", type=float, default=None,
                   help="SSVEP reversal frequency (Hz), e.g. 6.0 for a 6 Hz "
                        "checkerboard/flicker. Adds a narrow band centered "
                        "here (see --stim-bandwidth) on top of --bands. NOTE: "
                        "for a continuous, non-varying flicker, prefer "
                        "--pool-dir (pseudo-pair) over --surrogate (within-"
                        "dyad shuffle) to validate this band -- see module "
                        "docstring.")
    p.add_argument("--stim-bandwidth", type=float, default=1.0,
                   help="full width in Hz of the narrow band around --stim-hz "
                        "/ --tag-hz-a / --tag-hz-b (default 1.0, i.e. +/-0.5 Hz)")
    p.add_argument("--tag-hz-a", type=float, default=None,
                   help="FREQUENCY-TAGGING design: subject A's own SSVEP "
                        "reversal rate, when A and B watched DIFFERENT "
                        "flicker rates (needs two monitors -- see --pos in "
                        "make_checkerboard.py). Adds a narrow inter-brain band "
                        "at this frequency. Because B was never driven at "
                        "A's rate, any inter-brain PLV/circ-corr here can't "
                        "be explained by both brains locking onto one shared "
                        "external clock -- unlike --stim-hz, where both "
                        "subjects see the same flicker and elevated coupling "
                        "is ambiguous between 'real interpersonal effect' and "
                        "'two brains independently entrained to the same "
                        "signal'. Pass --tag-hz-b too for the full design.")
    p.add_argument("--tag-hz-b", type=float, default=None,
                   help="FREQUENCY-TAGGING design: subject B's own SSVEP "
                        "reversal rate. See --tag-hz-a.")
    p.add_argument("--surrogate", type=int, default=0,
                   help="number of within-dyad epoch-shuffle surrogate "
                        "permutations (0=off). Weak for continuous/stationary "
                        "stimuli -- see --pool-dir.")
    p.add_argument("--pool-dir", default=None,
                   help="directory of OTHER subjects' single-person CSVs "
                        "(same column format) to build a cross-dyad "
                        "pseudo-pair null distribution from. Preferred over "
                        "--surrogate for stimulus-locked bands like --stim-hz.")
    p.add_argument("--pool-shuffles", type=int, default=3,
                   help="random epoch-order draws per pool member when "
                        "building the pseudo-pair null (default 3)")
    p.add_argument("--match-null-length", dest="match_null_length",
                   action="store_true", default=True,
                   help="(continuous/default path only) length-match the "
                        "null distribution (--surrogate and/or --pool-dir) "
                        "against the real dyad's observed value before "
                        "computing p-values (default: on). Pool draws in "
                        "particular can end up with much less usable "
                        "overlap than the real dyad's own (often much "
                        "longer) joint-clean window, and short windows "
                        "inflate PLV/circ-corr for weak/no coupling -- "
                        "comparing a long real estimate to a null built "
                        "from short draws is not apples-to-apples. When on, "
                        "a common target sample count is chosen (see "
                        "--min-null-seconds) and both the null draws and a "
                        "matched copy of the observed value (averaged over "
                        "several random subsamples) are computed at that "
                        "N for the p-value comparison. The full-length "
                        "observed PLV/circ-r reported as the headline "
                        "number is unaffected -- only the null comparison "
                        "changes. Per-draw sample counts (Ns) are always "
                        "logged regardless of this flag.")
    p.add_argument("--no-match-null-length", dest="match_null_length",
                   action="store_false",
                   help="disable --match-null-length (restore old, "
                        "potentially N-mismatched null comparison).")
    p.add_argument("--min-null-seconds", type=float, default=10.0,
                   help="(continuous/default path only, --match-null-length) "
                        "floor, in seconds, on the length-matching target "
                        "sample count (default 10.0). Prevents matching down "
                        "to a degenerately short window if some pool draws "
                        "have very little usable overlap; draws shorter than "
                        "this are excluded from the target-size calculation "
                        "(they still appear in the logged N distribution).")
    p.add_argument("--pool-amplitude-threshold", type=float, default=None,
                   help="(--legacy-epochs only) peak-to-peak amplitude "
                        "threshold (uV) applied when epoching --pool-dir "
                        "recordings, separate from --amplitude-threshold. "
                        "Defaults to whatever --amplitude-threshold is. Pool "
                        "recordings often come from unrelated sessions with "
                        "different noise levels (e.g. a hardware test "
                        "recording) -- loosen this (or pass 0 to disable) "
                        "if they get rejected wholesale under the main "
                        "dyad's threshold.")
    p.add_argument("--correction", choices=["fdr", "none"], default="fdr",
                   help="multiple-comparisons correction across the 16 "
                        "channel pairs per band (default fdr). 'none' "
                        "restores the old raw p<0.05 counting.")
    p.add_argument("--amplitude-threshold", type=float, default=150.0,
                   help="(--legacy-epochs only) peak-to-peak amplitude "
                        "threshold in uV for epoch rejection (default 150). "
                        "Lower = stricter. 0 = disabled.")
    p.add_argument("--ica", action="store_true",
                   help="remove the single most blink-correlated ICA component per "
                        "subject before referencing (experimental -- only 4 channels "
                        "available, so separation is weak)")
    p.add_argument("--ocular-correction",
                   choices=["none", "regress", "emcp"], default="none",
                   help="ocular artifact handling (default none). 'regress' "
                        "applies the continuous Gratton regression: subtract "
                        "beta * ocular(t) per channel and KEEP the samples, "
                        "instead of masking them. Masking cannot give long "
                        "windows in a dyad (see analyze_blink_ceiling.py); "
                        "this is the route that can. 'emcp' is the faithful "
                        "Gratton/Coles/Donchin procedure: its own template "
                        "blink detector, SEPARATE propagation factors for "
                        "blink and non-blink samples, applied piecewise. "
                        "'regress' is the one-coefficient simplification. "
                        "See point 12.")
    p.add_argument("--emcp-channels", choices=["blink", "both"],
                   default="blink",
                   help="(--ocular-correction emcp) which derived EOG "
                        "channels enter the regression. 'blink' = vertical "
                        "only, which the original supports and which is the "
                        "safer choice here; 'both' adds the horizontal "
                        "channel as the original usually does, but on four "
                        "electrodes the two regressors together remove most "
                        "of the frontal variance (point 12).")
    p.add_argument("--regress-fit", choices=["blinks", "all"],
                   default="blinks",
                   help="where the regression coefficient is estimated "
                        "(default blinks). Blinks are ~2.5%% of a recording, "
                        "so fitting over the whole recording measures the "
                        "structural covariance between the regressor and the "
                        "channels it is built from, and subtracting that "
                        "strips real signal everywhere. Fitting on detected "
                        "blink periods only keeps the fit "
                        "artifact-dominated. 'all' restores the naive fit "
                        "for comparison.")
    p.add_argument("--regress-channels", choices=["blink", "saccade", "both"],
                   default="blink",
                   help="which ocular channels to regress out (default "
                        "blink). 'both' is refused when the analysis set is "
                        "frontal-only under a mastoid reference, where the "
                        "two regressors span the whole channel space and "
                        "would annihilate the data -- see point 12.")
    p.add_argument("--analysis-channels", choices=["auto", "all", "frontal"],
                   default="auto",
                   help="which channels the connectivity metrics run on "
                        "(default auto). 'auto' = all four under --reference "
                        "average, AF7/AF8 under --reference mastoid, because "
                        "a linked-mastoid reference makes TP9/TP10 exact "
                        "mirror images whose cross-brain metrics are "
                        "redundant. 'all'/'frontal' force it. Referencing, "
                        "artifact detection and the ocular channels always "
                        "use the full montage. See point 11.")
    p.add_argument("--artifact-source", choices=["eeg", "ocular", "both"],
                   default="eeg",
                   help="what artifact detection keys off (default eeg = the "
                        "original behaviour, so results stay comparable). "
                        "'ocular' detects on the derived blink/saccade "
                        "channels with per-participant thresholds instead of "
                        "a fixed EEG amplitude; 'both' ORs the two. See "
                        "point 10 of the module docstring.")
    p.add_argument("--ocular-k", type=float, default=5.0,
                   help="threshold = median + k*MAD of that participant's own "
                        "windowed peak-to-peak distribution (default 5). "
                        "Lower = stricter. MAD not sd, so the artifacts "
                        "cannot inflate the threshold meant to catch them.")
    p.add_argument("--ocular-detector", choices=["ptp", "velocity", "both"],
                   default="both",
                   help="ocular detector (default both). 'ptp' = "
                        "sliding-window peak-to-peak, catches anything large; "
                        "'velocity' = |d/dt| plus a duration test, which is "
                        "specifically a blink detector; 'both' ORs them.")
    p.add_argument("--blink-min-dur", type=float, default=0.05,
                   help="shortest accepted blink for the velocity detector, "
                        "seconds (default 0.05)")
    p.add_argument("--blink-max-dur", type=float, default=0.6,
                   help="longest accepted blink for the velocity detector, "
                        "seconds (default 0.6). Longer events are sustained "
                        "muscle/movement, not blinks, and are left to the "
                        "peak-to-peak criterion.")
    p.add_argument("--blink-threshold-a", type=float, default=None,
                   help="override subject A's automatic blink-channel "
                        "threshold (uV)")
    p.add_argument("--blink-threshold-b", type=float, default=None,
                   help="override subject B's automatic blink-channel "
                        "threshold (uV)")
    p.add_argument("--saccade-threshold-a", type=float, default=None,
                   help="override subject A's automatic saccade-channel "
                        "threshold (uV)")
    p.add_argument("--saccade-threshold-b", type=float, default=None,
                   help="override subject B's automatic saccade-channel "
                        "threshold (uV)")
    p.add_argument("--ocular-band", type=float, nargs=2,
                   default=[OCULAR_L_FREQ, OCULAR_H_FREQ],
                   metavar=("LOW", "HIGH"),
                   help="pass band (Hz) for the derived ocular channels "
                        f"(default {OCULAR_L_FREQ} {OCULAR_H_FREQ}). Kept "
                        "separate from the 1-40 Hz analysis band on purpose: "
                        "blink energy is mostly below 3 Hz, so the analysis "
                        "high-pass would remove most of what a blink detector "
                        "needs. Raised automatically if the recording is too "
                        "short to support the requested high-pass.")
    p.add_argument("--reference", choices=["average", "mastoid"], default="average",
                   help="re-reference scheme (default average). 'mastoid' uses a "
                        "linked-mastoid reference, i.e. every channel minus "
                        "mean(TP9, TP10). Both cancel the Muse's FPZ online "
                        "reference (which sits over the eyes and injects ocular "
                        "activity into all 4 channels as common mode); the mastoid "
                        "option additionally avoids mixing all 4 electrodes into "
                        "each other, at the cost of spending TP9/TP10 -- under it, "
                        "only AF7/AF8 carry independent information. See point 8 "
                        "of the module docstring.")
    p.add_argument("--prefilter", dest="prefilter", action="store_true", default=True,
                   help="(--legacy-epochs only; the continuous/default path "
                        "always band-passes the continuous signal before the "
                        "Hilbert transform, since that step is required, not "
                        "optional, there) band-pass the CONTINUOUS raw "
                        "signal into each band before epoching, instead of "
                        "filtering each short epoch independently inside "
                        "HyPyP (default: on). Reduces filter edge/transient "
                        "bias for narrow bands.")
    p.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                   help="(--legacy-epochs only) disable --prefilter and restore old per-epoch "
                        "narrowband filtering inside HyPyP (for comparison).")
    p.add_argument("--circ-corr-method", choices=["adjusted", "classic"], default="adjusted",
                   help="(continuous/default path only) which circular "
                        "correlation formula to use. 'adjusted' (default) "
                        "is the bias-adjusted Jammalamadaka & Sengupta (2001) "
                        "formula for arbitrary/not-well-defined circular "
                        "means, matching Zimmermann et al. (2024)'s "
                        "recommendation for continuous EEG and consistent "
                        "with HyPyP's 'accorr' mode used in --legacy-epochs. "
                        "'classic' restores the plain Fisher & Lee (1983) "
                        "formula (circ_corr_masked), which can be biased/"
                        "unstable for continuous EEG whose per-window "
                        "circular mean is not well defined -- kept for "
                        "comparison only. The --legacy-epochs path is "
                        "unaffected by this flag; it always uses HyPyP's "
                        "'accorr' (equivalent to 'adjusted').")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        "out", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    circ_corr_fn_continuous = (circ_corr_adjusted_masked if args.circ_corr_method == "adjusted"
                                else circ_corr_masked)

    freq_bands = dict(FREQ_BANDS)
    if args.stim_hz is not None:
        half = args.stim_bandwidth / 2
        stim_band_name = f"stim_{args.stim_hz:g}hz"
        freq_bands[stim_band_name] = (args.stim_hz - half, args.stim_hz + half)
        if stim_band_name not in args.bands:
            args.bands = list(args.bands) + [stim_band_name]
        print(f"  positive-control band added: {stim_band_name} "
              f"({freq_bands[stim_band_name][0]:.2f}-{freq_bands[stim_band_name][1]:.2f} Hz)")
        if args.pool_dir is None and args.surrogate > 0:
            print("  NOTE: validating a stimulus-locked band with within-dyad "
                  "--surrogate only. For a continuous/stationary flicker this "
                  "test is weak (see module docstring) -- consider also "
                  "passing --pool-dir with other subjects' recordings.")

    tag_band_names = []
    half = args.stim_bandwidth / 2
    if args.tag_hz_a is not None:
        tag_a_name = f"tagA_{args.tag_hz_a:g}hz"
        freq_bands[tag_a_name] = (args.tag_hz_a - half, args.tag_hz_a + half)
        if tag_a_name not in args.bands:
            args.bands = list(args.bands) + [tag_a_name]
        tag_band_names.append(tag_a_name)
        print(f"  frequency-tagging band added: {tag_a_name} "
              f"({freq_bands[tag_a_name][0]:.2f}-{freq_bands[tag_a_name][1]:.2f} Hz, "
              f"subject A's own reversal rate)")
    if args.tag_hz_b is not None:
        tag_b_name = f"tagB_{args.tag_hz_b:g}hz"
        freq_bands[tag_b_name] = (args.tag_hz_b - half, args.tag_hz_b + half)
        if tag_b_name not in args.bands:
            args.bands = list(args.bands) + [tag_b_name]
        tag_band_names.append(tag_b_name)
        print(f"  frequency-tagging band added: {tag_b_name} "
              f"({freq_bands[tag_b_name][0]:.2f}-{freq_bands[tag_b_name][1]:.2f} Hz, "
              f"subject B's own reversal rate)")
    if tag_band_names:
        print("  NOTE: only ONE subject was actually driven at each tag "
              "frequency (the other's activity there is incidental), so "
              "elevated inter-brain coupling in a tag_* band -- if it also "
              "clears --surrogate/--pool-dir -- is not explainable by both "
              "brains sharing one external clock, unlike --stim-hz.")
    if args.tag_hz_a is not None and args.tag_hz_b is not None and \
       abs(args.tag_hz_a - args.tag_hz_b) < args.stim_bandwidth:
        print(f"  WARNING: --tag-hz-a ({args.tag_hz_a}) and --tag-hz-b "
              f"({args.tag_hz_b}) are closer than --stim-bandwidth "
              f"({args.stim_bandwidth}) -- their bands overlap, defeating "
              "the point of tagging each subject at a different rate.")

    print("="*60)
    print("LOADING")
    print("="*60)
    onset_a = load_stimulus_onset(args.csv_a)
    onset_b = load_stimulus_onset(args.csv_b)
    if onset_a is not None and onset_b is not None:
        print(f"  IBS mode: stimulus markers found "
              f"(A={onset_a:.3f}s, B={onset_b:.3f}s)")
    elif onset_a is not None or onset_b is not None:
        print("  WARNING: only one recording has a stimulus marker -- "
              "alignment skipped")
        onset_a = onset_b = None
    else:
        print("  No stimulus markers found -- using full recordings")
    raw_a, fs_a = load_csv_to_raw(args.csv_a, "A", onset_s=onset_a)
    raw_b, fs_b = load_csv_to_raw(args.csv_b, "B", onset_s=onset_b)
    if fs_a != fs_b:
        print(f"  WARNING: sampling rates differ ({fs_a} vs {fs_b}). "
              "record_both.py should produce matched 64 Hz.")

    print()
    print("="*60)
    print("PREPROCESSING")
    print("="*60)
    # cap high-pass cutoff below Nyquist; for 64 Hz that's <32 Hz
    nyq = fs_a / 2
    h_freq = min(40.0, nyq * 0.95)
    ica_str = " + ICA blink removal" if args.ica else ""
    ref_str = ("linked-mastoid reference (TP9/TP10)" if args.reference == "mastoid"
               else "average reference")
    print(f"  bandpass 1-{h_freq:.0f} Hz{ica_str}, {ref_str}")
    raw_a_pp = preprocess(raw_a, h_freq=h_freq, use_ica=args.ica,
                          subject_label="A", reference=args.reference)
    raw_b_pp = preprocess(raw_b, h_freq=h_freq, use_ica=args.ica,
                          subject_label="B", reference=args.reference)

    # Derived ocular channels (point 9), built from the UNFILTERED raw so
    # they can use a lower high-pass than the analysis band. Inspection only
    # at this stage -- nothing downstream keys off them yet.
    print(f"  derived ocular channels "
          f"({args.ocular_band[0]:g}-{args.ocular_band[1]:g} Hz):")
    oc_a = make_ocular_channels(raw_a, l_freq=args.ocular_band[0],
                                h_freq=args.ocular_band[1],
                                subject_label="A", bads=raw_a_pp.info["bads"])
    oc_b = make_ocular_channels(raw_b, l_freq=args.ocular_band[0],
                                h_freq=args.ocular_band[1],
                                subject_label="B", bads=raw_b_pp.info["bads"])

    # Ocular correction by regression (point 12). Applied AFTER the ocular
    # channels are built (they are the regressors) and BEFORE artifact
    # detection, so detection sees the corrected signal and only has to catch
    # what the regression could not remove. oc_a/oc_b are deliberately left
    # uncorrected -- they are the record of where the blinks were, and the
    # ocular_channels.png figure should show the regressor, not the residual.
    regress_info = {"A": None, "B": None}
    if args.ocular_correction == "emcp":
        print("  ocular correction: Gratton EMCP (template blink detector, "
              "separate blink/saccade propagation factors)")
        raw_a_pp, regress_info["A"] = gratton_emcp(
            raw_a_pp, oc_a, l_freq=1.0, h_freq=h_freq, subject_label="A",
            channels=args.emcp_channels)
        raw_b_pp, regress_info["B"] = gratton_emcp(
            raw_b_pp, oc_b, l_freq=1.0, h_freq=h_freq, subject_label="B",
            channels=args.emcp_channels)
        if not (regress_info["A"]["applied"] and regress_info["B"]["applied"]):
            print("  WARNING: EMCP was not applied for at least one subject "
                  "(see above); that subject's data below is uncorrected")
    elif args.ocular_correction == "regress":
        regressors = (("blink", "saccade") if args.regress_channels == "both"
                      else (args.regress_channels,))
        print(f"  ocular correction: Gratton regression, regressors="
              f"{list(regressors)}")
        raw_a_pp, regress_info["A"] = gratton_regress(
            raw_a_pp, oc_a, regressors=regressors, l_freq=1.0, h_freq=h_freq,
            subject_label="A", fit_on=args.regress_fit, k=args.ocular_k)
        raw_b_pp, regress_info["B"] = gratton_regress(
            raw_b_pp, oc_b, regressors=regressors, l_freq=1.0, h_freq=h_freq,
            subject_label="B", fit_on=args.regress_fit, k=args.ocular_k)
        if not (regress_info["A"]["applied"] and regress_info["B"]["applied"]):
            print("  WARNING: regression was not applied for at least one "
                  "subject (see above); results below are uncorrected for "
                  "that subject")

    if args.sync_signal_hz is not None and not args.legacy_epochs:
        offset_s, drift_rate = estimate_sync_offset_drift(
            raw_a_pp, raw_b_pp, args.sync_signal_hz,
            bandwidth=args.sync_bandwidth, drift_only=args.sync_drift_only)
        dur = raw_b_pp.n_times / fs_a
        print(f"  cross-device sync ({args.sync_signal_hz:g} Hz driver): "
              f"offset={offset_s * 1000:.1f}ms  drift={drift_rate * 1e6:.1f}ppm "
              f"(~{drift_rate * dur * 1000:.1f}ms over the recording)"
              f"{'  [drift only]' if args.sync_drift_only else ''}")
        raw_b_pp = apply_sync_correction(raw_b_pp, offset_s, drift_rate)

    plot_raw_with_gaps(raw_a_pp, raw_b_pp, os.path.join(out_dir, "raw_with_gaps.png"))
    plot_psd(raw_a_pp, raw_b_pp, os.path.join(out_dir, "psd.png"))
    plot_ocular_channels(oc_a, oc_b, os.path.join(out_dir, "ocular_channels.png"))

    # Which channels the METRICS run on (point 11). Everything above this
    # line -- referencing, bad-channel detection, ocular channels -- and the
    # artifact detection below it still use the full four-electrode montage;
    # only the connectivity step is restricted.
    analysis_ch = resolve_analysis_channels(args.analysis_channels,
                                            args.reference)
    set_analysis_channels(analysis_ch)
    n_pairs = len(analysis_ch) ** 2
    print(f"  analysis channels: {analysis_ch} "
          f"({n_pairs} cross-brain pairs per band"
          + (", auto-selected for the mastoid reference"
             if args.analysis_channels == "auto" and args.reference == "mastoid"
             else "") + ")")

    epochs_a = epochs_b = None
    n_ep = None
    bad_a = bad_b = good_mask = None
    n_common = None

    if args.legacy_epochs:
        print()
        print("="*60)
        print("EPOCHING (legacy fixed-length epochs, --legacy-epochs)")
        print("="*60)
        amp_thresh = args.amplitude_threshold if args.amplitude_threshold > 0 else None
        thresh_str = f"{amp_thresh} uV" if amp_thresh else "disabled"
        print(f"  epoch_len={args.epoch_len}s  overlap={args.epoch_overlap}s  "
              f"amplitude_threshold={thresh_str}")
        # epoch rejection keys off amplitude, so restrict AFTER it would be
        # wrong for the dropped channels but right for the kept ones; restrict
        # first so the epochs that survive are the ones the metrics will use
        raw_a_ana = restrict_to_analysis(raw_a_pp, analysis_ch, "A")
        raw_b_ana = restrict_to_analysis(raw_b_pp, analysis_ch, "B")
        epochs_a = epoch_with_gap_rejection(raw_a_ana, args.epoch_len, args.epoch_overlap,
                                            amplitude_uv=amp_thresh or 1e9)
        epochs_b = epoch_with_gap_rejection(raw_b_ana, args.epoch_len, args.epoch_overlap,
                                            amplitude_uv=amp_thresh or 1e9)
        print(f"  Subject A: {len(epochs_a)} epochs survived (out of "
              f"{len(epochs_a.drop_log)} attempted)")
        print(f"  Subject B: {len(epochs_b)} epochs survived (out of "
              f"{len(epochs_b.drop_log)} attempted)")

        if len(epochs_a) == 0 or len(epochs_b) == 0:
            print()
            print("  No surviving epochs. The recording is too gappy for this "
                  "epoch length.")
            print("  Try:  --epoch-len 1.0 --epoch-overlap 0.5")
            print("  Or: get cleaner data (single-BT-adapter problem).")
            sys.exit(0)

        # Match epochs by original time-slot index, not by position in each
        # subject's own surviving list. A and B reject different epochs (whichever
        # overlap THEIR OWN gaps/artifacts), so truncating by position pairs
        # epochs from different real-world moments -- e.g. A's epoch #5 (its 5th
        # survivor) might be B's epoch #9 in time, growing worse the more the two
        # subjects' rejections diverge. Keep only slots that survived in both.
        common = np.intersect1d(epochs_a.selection, epochs_b.selection)
        epochs_a = epochs_a[np.isin(epochs_a.selection, common)]
        epochs_b = epochs_b[np.isin(epochs_b.selection, common)]
        n_ep = len(common)
        print(f"  Using {n_ep} time-aligned matched epochs for connectivity "
              f"(kept only time slots that survived rejection in both subjects).")
    else:
        print()
        print("="*60)
        print("CONTINUOUS ARTIFACT DETECTION (default; see --legacy-epochs "
              "for the old fixed-epoch path)")
        print("="*60)
        print(f"  sliding window={args.artifact_window}s  step={args.artifact_step}s  "
              f"threshold={args.artifact_threshold} uV  pad={args.artifact_pad}s")
        print(f"  source={args.artifact_source}"
              + (f"  detector={args.ocular_detector}  k={args.ocular_k:g}"
                 if args.artifact_source != "eeg" else ""))
        print(f"  circular correlation method: {args.circ_corr_method}")

        use_eeg = args.artifact_source in ("eeg", "both")
        use_ocular = args.artifact_source in ("ocular", "both")
        ocular_thr = {"A": None, "B": None}
        extra = {"A": None, "B": None}
        if use_ocular:
            for lbl, oc, raw_pp, b_ovr, s_ovr in (
                    ("A", oc_a, raw_a_pp, args.blink_threshold_a,
                     args.saccade_threshold_a),
                    ("B", oc_b, raw_b_pp, args.blink_threshold_b,
                     args.saccade_threshold_b)):
                thr = ocular_thresholds(oc, window_s=args.artifact_window,
                                        step_s=args.artifact_step,
                                        k=args.ocular_k,
                                        blink_override=b_ovr,
                                        saccade_override=s_ovr)
                ocular_thr[lbl] = thr
                extra[lbl], _ = ocular_bad_mask(
                    oc, thr, raw_pp.n_times,
                    window_s=args.artifact_window, step_s=args.artifact_step,
                    detector=args.ocular_detector,
                    min_blink_s=args.blink_min_dur,
                    max_blink_s=args.blink_max_dur, subject_label=lbl)
                # --artifact-source ocular REPLACES the EEG amplitude
                # criterion. If this subject has no usable ocular channel
                # (every frontal or every mastoid electrode flagged bad),
                # that leaves no amplitude rejection at all, and the run
                # will cheerfully report ~100% "clean" on data that is
                # entirely railed. Say so rather than let the number stand.
                if not use_eeg and oc.blink is None and oc.saccade is None:
                    print(f"     {lbl}: WARNING no ocular channels AND "
                          "--artifact-source ocular -- NO amplitude "
                          "rejection is being applied to this subject. "
                          "Any 'clean' fraction below is meaningless; use "
                          "--artifact-source both.")

        bad_a = continuous_bad_mask(raw_a_pp, window_s=args.artifact_window,
                                     step_s=args.artifact_step,
                                     threshold_uv=args.artifact_threshold,
                                     pad_s=args.artifact_pad,
                                     extra_bad=extra["A"],
                                     use_eeg_amplitude=use_eeg)
        bad_b = continuous_bad_mask(raw_b_pp, window_s=args.artifact_window,
                                     step_s=args.artifact_step,
                                     threshold_uv=args.artifact_threshold,
                                     pad_s=args.artifact_pad,
                                     extra_bad=extra["B"],
                                     use_eeg_amplitude=use_eeg)
        n_common = min(len(bad_a), len(bad_b))
        bad_a = bad_a[:n_common]
        bad_b = bad_b[:n_common]
        good_mask = ~(bad_a | bad_b)
        total_s = n_common / fs_a
        good_s = good_mask.sum() / fs_a
        longest_s = longest_clean_run_s(good_mask, fs_a)
        print(f"  Subject A: {100 * (~bad_a).mean():.1f}% clean "
              f"(longest run {longest_clean_run_s(~bad_a, fs_a):.1f}s)")
        print(f"  Subject B: {100 * (~bad_b).mean():.1f}% clean "
              f"(longest run {longest_clean_run_s(~bad_b, fs_a):.1f}s)")
        print(f"  Jointly clean (usable for connectivity): {good_s:.1f}s / "
              f"{total_s:.1f}s ({100 * good_mask.mean():.1f}%)")
        # The metric that decides whether a preprocessing change actually
        # helped: clean fraction can stay flat while the data goes from one
        # long stretch to many short crumbs, and short windows inflate PLV
        # (point 5). Compare this number across --reference settings.
        print(f"  Longest continuous jointly-clean run: {longest_s:.1f}s")

        if good_mask.sum() == 0:
            print()
            print("  No jointly-clean samples survive. Try loosening "
                  "--artifact-threshold, or check hardware/fit.")
            sys.exit(0)

        # masks are built; from here on only the analysis channels matter
        raw_a_pp = restrict_to_analysis(raw_a_pp, analysis_ch, "A")
        raw_b_pp = restrict_to_analysis(raw_b_pp, analysis_ch, "B")

    # ------------------------------------------------------------------
    # Optionally load a pool of OTHER subjects' recordings for pseudo-pair
    # (cross-dyad) validation. Each pool file goes through the SAME load ->
    # preprocess chain as the two main subjects, so the comparison is fair.
    # ------------------------------------------------------------------
    pool_epochs = []
    pool_data = []  # continuous mode: list of (raw_pp, bad_mask)
    pool_dir_used = args.pool_dir
    if args.pool_dir or args.stim_hz is not None:
        print()
        print("="*60)
        print("LOADING POOL (for pseudo-pair / cross-dyad validation)")
        print("="*60)
        pool_files = resolve_pool_csvs(args.pool_dir, args.csv_a, args.csv_b)
        pool_source = args.pool_dir or os.path.dirname(os.path.abspath(args.csv_a))
        if args.pool_dir and not pool_files:
            print(f"  WARNING: no CSVs found in {args.pool_dir}; falling back to "
                  f"{pool_source} for pool validation.")
        elif not args.pool_dir and not pool_files:
            print(f"  WARNING: no pool CSVs found in {pool_source}; "
                  "pseudo-pair validation will be skipped.")
        # never pool the two real subjects' own files against themselves
        pool_files = [f for f in pool_files
                      if os.path.abspath(f) not in
                      (os.path.abspath(args.csv_a), os.path.abspath(args.csv_b))]
        if not pool_files:
            print(f"  WARNING: no usable pool CSVs remained after excluding the "
                  "real subjects -- pseudo-pair validation will be skipped.")
        pool_amp_thresh_arg = (args.pool_amplitude_threshold
                               if args.pool_amplitude_threshold is not None
                               else args.amplitude_threshold)
        pool_amp_thresh = pool_amp_thresh_arg if pool_amp_thresh_arg > 0 else None

        if args.legacy_epochs:
            print(f"  pool amplitude_threshold={pool_amp_thresh or 'disabled'} uV "
                  f"{'(same as main dyad)' if args.pool_amplitude_threshold is None else '(override)'}")
            for f in pool_files:
                ep, fs_pool = load_and_epoch_subject(
                    f, subject_label=f"POOL_{os.path.basename(f)}",
                    epoch_len_s=args.epoch_len, overlap_s=args.epoch_overlap,
                    h_freq=h_freq, amplitude_uv=pool_amp_thresh or 1e9,
                    use_ica=args.ica, align_onset=True, quiet=False,
                    reference=args.reference,
                )
                if ep is not None and fs_pool == fs_a:
                    pool_epochs.append(restrict_to_analysis(
                        ep, analysis_ch, quiet=True))
                elif ep is not None:
                    print(f"  WARNING: {f} has fs={fs_pool} != {fs_a}, skipping "
                          "(sampling rate must match for pooling)")
            print(f"  Pool ready: {len(pool_epochs)} usable recordings "
                  f"(from {len(pool_files)} files found)")
        else:
            for f in pool_files:
                raw_pool, bad_pool, fs_pool = load_and_preprocess_continuous(
                    f, subject_label=f"POOL_{os.path.basename(f)}",
                    h_freq=h_freq, use_ica=args.ica, align_onset=True, quiet=False,
                    artifact_window=args.artifact_window,
                    artifact_step=args.artifact_step,
                    artifact_threshold=args.artifact_threshold,
                    artifact_pad=args.artifact_pad,
                    reference=args.reference,
                )
                if raw_pool is not None and fs_pool == fs_a:
                    pool_data.append((restrict_to_analysis(
                        raw_pool, analysis_ch, quiet=True), bad_pool))
                elif raw_pool is not None:
                    print(f"  WARNING: {f} has fs={fs_pool} != {fs_a}, skipping "
                          "(sampling rate must match for pooling)")
            print(f"  Pool ready: {len(pool_data)} usable recordings "
                  f"(from {len(pool_files)} files found)")

    print()
    print("="*60)
    print("PLV + CIRCULAR CORRELATION PER BAND")
    print("="*60)
    if args.legacy_epochs:
        if args.prefilter:
            print("  --prefilter is ON: continuous raw is band-passed BEFORE "
                  "epoching for each band (reduces per-epoch filter edge bias).")
        else:
            print("  --prefilter is OFF: each short epoch is narrowband-filtered "
                  "independently inside HyPyP (legacy behaviour).")
    else:
        print("  continuous path: band-passing + Hilbert transform run on the "
              "full recording per band, then only jointly-clean samples "
              "(good_mask) are used for the PLV/circ-corr sum "
              f"(circ-corr method: {args.circ_corr_method}).")

    plvs = {}
    ccs = {}
    p_values = {}
    cc_p_values = {}
    sig_masks = {}
    cc_sig_masks = {}
    summary_lines = [f"Hyperscanning summary  {datetime.now().isoformat()}"]
    summary_lines.append(f"  A: {args.csv_a}")
    summary_lines.append(f"  B: {args.csv_b}")
    if args.legacy_epochs:
        summary_lines.append(f"  fs={fs_a:.0f} Hz, epochs={n_ep}, "
                             f"epoch_len={args.epoch_len}s")
    else:
        summary_lines.append(
            f"  fs={fs_a:.0f} Hz, continuous artifact rejection "
            f"(window={args.artifact_window}s step={args.artifact_step}s "
            f"threshold={args.artifact_threshold}uV), "
            f"{good_mask.sum() / fs_a:.1f}s/{n_common / fs_a:.1f}s usable "
            f"({100 * good_mask.mean():.1f}%), circ-corr method={args.circ_corr_method}"
        )
    summary_lines.append(f"  reference={args.reference}  ica={args.ica}")
    ocular_bits = []
    for oc in (oc_a, oc_b):
        have = [n for n, t in (("blink", oc.blink), ("saccade", oc.saccade))
                if t is not None]
        ocular_bits.append(f"{oc.label}:{'+'.join(have) if have else 'none'}")
    summary_lines.append(
        f"  ocular channels ({oc_a.l_freq:g}-{oc_a.h_freq:g} Hz): "
        + "  ".join(ocular_bits))
    summary_lines.append(
        f"  artifact_source={args.artifact_source}  "
        f"analysis_channels={'+'.join(analysis_ch)} "
        f"({len(analysis_ch) ** 2} pairs/band)")
    summary_lines.append(f"  ocular_correction={args.ocular_correction}"
                         + (f" ({args.regress_channels})"
                            if args.ocular_correction == "regress" else ""))
    for lbl in ("A", "B"):
        ri = regress_info.get(lbl)
        if ri and ri["applied"]:
            summary_lines.append(
                f"    {lbl} variance kept: "
                + "  ".join(f"{ch.split('_')[-1]}={100 * v:.0f}%"
                            for ch, v in ri["variance_kept"].items()))
        elif ri:
            summary_lines.append(f"    {lbl} regression NOT applied")
    if not args.legacy_epochs and args.artifact_source != "eeg":
        summary_lines.append(
            f"  ocular detector={args.ocular_detector}  k={args.ocular_k:g}")
        for lbl in ("A", "B"):
            thr = ocular_thr.get(lbl)
            if not thr:
                continue
            bits = "  ".join(
                f"{n}={thr[n]:.1f}" for n in ("blink", "saccade")
                if thr.get(n) is not None)
            vel = (f"  velocity={thr['blink_velocity']:.0f} uV/s"
                   if thr.get("blink_velocity") else "")
            summary_lines.append(f"    {lbl} thresholds (uV): {bits}{vel}")
    if not args.legacy_epochs:
        summary_lines.append(
            f"  longest continuous jointly-clean run: "
            f"{longest_clean_run_s(good_mask, fs_a):.1f}s")
    summary_lines.append(f"  prefilter={args.prefilter}  "
                         f"correction={args.correction}  "
                         f"pool_dir={args.pool_dir or '(none)'}")
    summary_lines.append("")

    for band_name in args.bands:
        if band_name not in freq_bands:
            print(f"  skipping unknown band: {band_name}")
            continue
        band = freq_bands[band_name]
        if band[1] >= nyq:
            print(f"  skipping {band_name} ({band[0]}-{band[1]} Hz): "
                  f"above Nyquist ({nyq:.1f} Hz)")
            continue

        if args.legacy_epochs:
            # -- build (possibly pre-filtered) epochs for THIS band --------
            if args.prefilter:
                raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
                raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
                epochs_a_band = epoch_with_gap_rejection(
                    raw_a_band, args.epoch_len, args.epoch_overlap,
                    amplitude_uv=amp_thresh or 1e9)
                epochs_b_band = epoch_with_gap_rejection(
                    raw_b_band, args.epoch_len, args.epoch_overlap,
                    amplitude_uv=amp_thresh or 1e9)
                common_band = np.intersect1d(epochs_a_band.selection, epochs_b_band.selection)
                epochs_a_band = epochs_a_band[np.isin(epochs_a_band.selection, common_band)]
                epochs_b_band = epochs_b_band[np.isin(epochs_b_band.selection, common_band)]
                already_filtered = True
            else:
                epochs_a_band, epochs_b_band = epochs_a, epochs_b
                already_filtered = False

            if len(epochs_a_band) == 0 or len(epochs_b_band) == 0:
                print(f"  {band_name}: 0 epochs survive after band-specific "
                      "filtering/rejection -- skipping")
                continue

            plv = plv_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                             already_filtered=already_filtered)
            cc = circular_corr_hypyp(epochs_a_band, epochs_b_band, band, fs_a,
                                      already_filtered=already_filtered)

            def compute_within_null(n_surr, _epochs_a_band=epochs_a_band,
                                     _epochs_b_band=epochs_b_band, _band=band,
                                     _already_filtered=already_filtered):
                null_plv = surrogate_distribution(
                    _epochs_a_band, _epochs_b_band, _band, n_surr,
                    metric_fn=plv_hypyp, sfreq=fs_a, already_filtered=_already_filtered)
                null_cc = surrogate_distribution(
                    _epochs_a_band, _epochs_b_band, _band, n_surr,
                    metric_fn=circular_corr_hypyp, sfreq=fs_a, already_filtered=_already_filtered)
                return null_plv, null_cc

            def compute_pool_null(_epochs_a_band=epochs_a_band, _epochs_b_band=epochs_b_band,
                                   _band=band, _already_filtered=already_filtered):
                null_plv_pool = []
                for target in (_epochs_a_band, _epochs_b_band):
                    res = pseudo_pair_distribution(
                        target, pool_epochs, _band, plv_hypyp, sfreq=fs_a,
                        already_filtered=_already_filtered,
                        shuffles_per_pool_member=args.pool_shuffles)
                    if res is not None:
                        null_plv_pool.append(res)
                return np.concatenate(null_plv_pool, axis=0) if null_plv_pool else None

            have_pool = bool(pool_epochs)
        else:
            raw_a_band = prefilter_raw_for_band(raw_a_pp, band)
            raw_b_band = prefilter_raw_for_band(raw_b_pp, band)
            analytic_a = analytic_signal(raw_a_band, n_samples=n_common)
            analytic_b = analytic_signal(raw_b_band, n_samples=n_common)

            plv = plv_masked(analytic_a, analytic_b, good_mask)
            cc = circ_corr_fn_continuous(analytic_a, analytic_b, good_mask)

            def compute_within_null(n_surr, target_n=None, seed=0,
                                     _analytic_a=analytic_a, _analytic_b=analytic_b,
                                     _bad_a=bad_a, _bad_b=bad_b):
                null_plv, ns_plv = circular_shift_surrogates_continuous(
                    _analytic_a, _bad_a, _analytic_b, _bad_b, n_surr, plv_masked,
                    seed=seed, target_n=target_n)
                null_cc, ns_cc = circular_shift_surrogates_continuous(
                    _analytic_a, _bad_a, _analytic_b, _bad_b, n_surr, circ_corr_fn_continuous,
                    seed=seed, target_n=target_n)
                return null_plv, null_cc, ns_plv, ns_cc

            def compute_pool_null(target_n=None, seed=0,
                                   _analytic_a=analytic_a, _analytic_b=analytic_b,
                                   _bad_a=bad_a, _bad_b=bad_b, _band=band):
                if not pool_data:
                    return None, np.array([])
                pool_analytic_list = []
                pool_bad_list = []
                for raw_pool, bad_pool in pool_data:
                    raw_pool_band = prefilter_raw_for_band(raw_pool, _band)
                    pool_analytic_list.append(analytic_signal(raw_pool_band))
                    pool_bad_list.append(bad_pool)
                null_plv_pool = []
                ns_pool_all = []
                for target_analytic, target_bad in ((_analytic_a, _bad_a), (_analytic_b, _bad_b)):
                    res, ns = pseudo_pair_continuous(
                        target_analytic, target_bad, pool_analytic_list, pool_bad_list,
                        plv_masked, shuffles_per_pool_member=args.pool_shuffles,
                        seed=seed, target_n=target_n)
                    ns_pool_all.append(ns)
                    if res is not None:
                        null_plv_pool.append(res)
                ns_pool_all = np.concatenate(ns_pool_all) if ns_pool_all else np.array([])
                pooled = np.concatenate(null_plv_pool, axis=0) if null_plv_pool else None
                return pooled, ns_pool_all

            have_pool = bool(pool_data)

        plvs[band_name] = plv
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean PLV = {plv.mean():.3f}  max = {plv.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"plv_{band_name}.npy"), plv)

        ccs[band_name] = cc
        line = (f"  {band_name:6s} ({band[0]:4.1f}-{band[1]:4.1f} Hz): "
                f"mean circ-r = {cc.mean():.3f}  min = {cc.min():.3f}  max = {cc.max():.3f}")
        print(line)
        summary_lines.append(line)
        np.save(os.path.join(out_dir, f"circ_corr_{band_name}.npy"), cc)

        # ---------------- WITHIN-DYAD surrogate (optional) -------------
        if args.surrogate > 0:
            print(f"     running {args.surrogate} within-dyad surrogates "
                  f"(PLV + circ-corr)...")
            if args.legacy_epochs:
                null_plv, null_cc = compute_within_null(args.surrogate)
            else:
                null_plv, null_cc, ns_within_plv, ns_within_cc = compute_within_null(args.surrogate)
                real_n = int(good_mask.sum())
                line = (f"     within-dyad null sample sizes: "
                        f"min={ns_within_plv.min()/fs_a:.1f}s  "
                        f"median={np.median(ns_within_plv)/fs_a:.1f}s  "
                        f"max={ns_within_plv.max()/fs_a:.1f}s  "
                        f"(real dyad N={real_n/fs_a:.1f}s)")
                print(line)
                summary_lines.append(line)
            p_val = (null_plv >= plv[None, :, :]).mean(axis=0)
            p_values[band_name] = p_val

            cc_p_val = (np.abs(null_cc) >= np.abs(cc[None, :, :])).mean(axis=0)
            cc_p_values[band_name] = cc_p_val

            if args.correction == "fdr":
                sig_mask, p_corrected = fdr_bh(p_val)
                cc_sig_mask, cc_p_corrected = fdr_bh(cc_p_val)
                n_sig = int(sig_mask.sum())
                n_sig_cc = int(cc_sig_mask.sum())
                n_sig_raw = int((p_val < 0.05).sum())
                n_sig_cc_raw = int((cc_p_val < 0.05).sum())
                line = (f"     PLV significant pairs: {n_sig}/{plv.size} "
                        f"(FDR-corrected)   [{n_sig_raw}/{plv.size} raw p<0.05, uncorrected]")
                print(line)
                summary_lines.append(line)
                line = (f"     circ-corr significant pairs: {n_sig_cc}/{cc.size} "
                        f"(FDR-corrected)   [{n_sig_cc_raw}/{cc.size} raw p<0.05, uncorrected]")
                print(line)
                summary_lines.append(line)
                sig_masks[band_name] = sig_mask
                cc_sig_masks[band_name] = cc_sig_mask
            else:
                n_sig = int((p_val < 0.05).sum())
                n_sig_cc = int((cc_p_val < 0.05).sum())
                line = f"     PLV significant pairs (p<0.05, UNCORRECTED): {n_sig}/{plv.size}"
                print(line)
                summary_lines.append(line)
                line = f"     circ-corr significant pairs (p<0.05, UNCORRECTED): {n_sig_cc}/{cc.size}"
                print(line)
                summary_lines.append(line)

            np.save(os.path.join(out_dir, f"plv_p_within_{band_name}.npy"), p_val)
            np.save(os.path.join(out_dir, f"circ_corr_p_within_{band_name}.npy"), cc_p_val)

        # ---------------- CROSS-DYAD pseudo-pair null (preferred) ------
        if have_pool:
            n_pool = len(pool_epochs) if args.legacy_epochs else len(pool_data)
            print(f"     running pseudo-pair null against {n_pool} "
                  f"pool recordings x{args.pool_shuffles} draws (PLV)...")

            plv_for_pval = plv  # may be replaced by a length-matched value below

            if args.legacy_epochs:
                null_plv_pool = compute_pool_null()
            else:
                null_plv_pool, ns_pool = compute_pool_null()
                real_n = int(good_mask.sum())
                if ns_pool.size > 0:
                    line = (f"     pool null sample sizes: "
                            f"min={ns_pool.min()/fs_a:.1f}s  "
                            f"median={np.median(ns_pool)/fs_a:.1f}s  "
                            f"max={ns_pool.max()/fs_a:.1f}s  "
                            f"(real dyad N={real_n/fs_a:.1f}s)")
                    print(line)
                    summary_lines.append(line)

                    if args.match_null_length:
                        nonzero = ns_pool[ns_pool > 0]
                        floor_n = int(args.min_null_seconds * fs_a)
                        robust_pool_n = int(np.percentile(nonzero, 10)) if nonzero.size else 0
                        target_n = max(floor_n, robust_pool_n)
                        target_n = min(target_n, real_n) if real_n > 0 else target_n
                        target_n = max(target_n, 1)

                        if 0 < target_n < real_n:
                            line = (f"     length-matching pool-null comparison to "
                                    f"N={target_n/fs_a:.1f}s (10th pct of pool draw "
                                    f"sizes, floored at {args.min_null_seconds:.0f}s)")
                            print(line)
                            summary_lines.append(line)

                            null_plv_pool_matched, ns_pool_matched = compute_pool_null(
                                target_n=target_n, seed=1)
                            plv_matched_obs = matched_observed_value(
                                analytic_a, analytic_b, good_mask, target_n,
                                plv_masked, n_draws=5, seed=2)

                            if null_plv_pool_matched is not None and plv_matched_obs is not None:
                                null_plv_pool = null_plv_pool_matched
                                plv_for_pval = plv_matched_obs
                                line = (f"     length-matched observed PLV "
                                        f"(N={target_n/fs_a:.1f}s, avg of 5 subsamples) "
                                        f"= {plv_matched_obs.mean():.3f}  "
                                        f"[full-length headline PLV = {plv.mean():.3f}]")
                                print(line)
                                summary_lines.append(line)
                            else:
                                line = ("     length-matched recompute failed "
                                        "(insufficient samples) -- falling back to "
                                        "unmatched null comparison")
                                print(line)
                                summary_lines.append(line)

            if null_plv_pool is not None:
                p_val_pool = (null_plv_pool >= plv_for_pval[None, :, :]).mean(axis=0)
                matched_tag = " (length-matched)" if plv_for_pval is not plv else ""
                if args.correction == "fdr":
                    sig_mask_pool, _ = fdr_bh(p_val_pool)
                    n_sig_pool = int(sig_mask_pool.sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair{matched_tag}, "
                            f"FDR-corrected): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                else:
                    sig_mask_pool = p_val_pool < 0.05
                    n_sig_pool = int(sig_mask_pool.sum())
                    line = (f"     PLV significant pairs vs POOL (pseudo-pair{matched_tag}, "
                            f"UNCORRECTED): {n_sig_pool}/{plv.size}  "
                            f"[pool null mean={null_plv_pool.mean():.3f}]")
                print(line)
                summary_lines.append(line)
                np.save(os.path.join(out_dir, f"plv_p_pool_{band_name}.npy"), p_val_pool)
                line = (f"     Interpretation: real PLV={plv_for_pval.mean():.3f}{matched_tag} vs "
                        f"pool (independent, same-stimulus) PLV={null_plv_pool.mean():.3f} "
                        f"-> {'ABOVE pool baseline' if plv_for_pval.mean() > null_plv_pool.mean() else 'NOT above pool baseline'}")
                print(line)
                summary_lines.append(line)

                if args.stim_hz is not None and band_name == stim_band_name:
                    verdict = summarize_positive_control(
                        plv_for_pval, null_plv_pool, p_val_pool, sig_mask=sig_mask_pool)
                    verdict_line = (
                        f"     positive-control verdict: {verdict['status']} - "
                        f"{verdict['reason']}"
                    )
                    print(verdict_line)
                    summary_lines.append(verdict_line)
            else:
                if args.stim_hz is not None and band_name == stim_band_name:
                    verdict_line = (
                        f"     positive-control verdict: NOT EVALUATED - no pool "
                        "recordings were available for pseudo-pair comparison"
                    )
                    print(verdict_line)
                    summary_lines.append(verdict_line)

        plot_plv_matrix(
            plv, band_name,
            os.path.join(out_dir, f"plv_interbrain_{band_name}.png"),
            surrogate_p=p_values.get(band_name),
            sig_mask=sig_masks.get(band_name),
        )
        plot_circ_corr_matrix(
            cc, band_name,
            os.path.join(out_dir, f"circ_corr_{band_name}.png"),
            surrogate_p=cc_p_values.get(band_name),
            sig_mask=cc_sig_masks.get(band_name),
        )

    if plvs:
        plot_plv_comparison(plvs, os.path.join(out_dir, "plv_comparison.png"))
    if ccs:
        plot_circ_corr_comparison(ccs, os.path.join(out_dir, "circ_corr_comparison.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print()
    print(f"Wrote outputs to: {out_dir}")
    print("  plv_<band>.npy                - PLV matrices")
    print("  plv_interbrain_<band>.png     - PLV heatmaps (0 to 1); * = FDR-sig, (*) = uncorrected only")
    print("  plv_comparison.png            - PLV all bands side by side")
    print("  circ_corr_<band>.npy          - circular correlation matrices")
    print("  circ_corr_<band>.png          - circular corr heatmaps (-1 to 1)")
    print("  circ_corr_comparison.png      - circular corr all bands side by side")
    print("  plv_p_within_<band>.npy       - within-dyad surrogate p-values (if --surrogate)")
    print("  plv_p_pool_<band>.npy         - cross-dyad pseudo-pair p-values (if --pool-dir)")
    print("  raw_with_gaps.png             - signal + gap markers")
    print("  psd.png                       - power spectrum QC")
    print("  ocular_channels.png           - derived blink/saccade channels")
    print("  summary.txt                   - numerical summary")


if __name__ == "__main__":
    main()