# Findings

Everything here was found by measurement, not by reading code. Each item looked
plausible in its broken state, and several of the "improvements" listed under
dead ends sounded obviously right and were measurably wrong.

Read this before changing anything in `detection/` or `tracking/`.

---

## 1. Bugs found on synthetic footage

1. **MOG2 initialisation ghost.** A target present in the first frames gets
   baked into the background model, leaving a *stationary blob detected every
   frame* at the target's start position, which seeds a permanent phantom track.
   Fixed with a warm-up phase (fast adaptation, suppressed output):
   `detection.motion.warmup_frames`.

2. **Mode flapping with a stale background model** — severe: precision 4 %. In
   `camera_motion: auto`, a sinusoidally panning camera is momentarily still at
   each sweep end, flipping back to MOG2, whose model was assembled from frames
   at *different* pan positions and therefore fires almost everywhere. Fixed
   with switch hysteresis plus discarding and rebuilding the model whenever the
   camera comes to rest.

3. **Coasting tails.** `max_age` extrapolated departed objects for a full
   second, painting phantom contacts. Fixed by terminating tracks whose
   prediction leaves the frame (coasting is for *occlusion*, not departure) and
   by scaling the coasting budget with track maturity.

4. **Config writes silently discarded.** `Config.__getattr__` returned a fresh
   shallow copy, so `cfg.zoom["initial"] = 3.0` mutated a temporary — `--zoom`,
   `--theme` and `--backend` all did nothing, with no error. Nested dicts are
   now converted at insertion time. **If you touch `utils/config.py`, verify
   write-through still works.**

5. **Mahalanobis gating on size.** The association gate originally used all four
   measurement dimensions. For a faint target the apparent box size swings
   wildly frame to frame, so the size terms dominated and vetoed associations
   whose *position* agreed perfectly — the track died and the object was reborn
   under a new ID. Gating on position only (2 DoF) lifted hard-clip recall
   81.6 → 84.7 % and cut ID switches 22 → 15.

---

## 2. Bugs found only once real footage was used

These three did not show up in three scenarios of synthetic video and were
obvious within one run of real video. That is the lesson, more than any
individual fix.

### 2.1 The ego-motion estimate failed open

`_estimate_ego_motion` returned `0.0` from every failure path — no previous
frame, too few corners, optical flow not converging — and the mode selector
cannot tell that apart from a genuine measurement of "the camera is perfectly
still".

Over clear blue sky there is nothing to track, so on the chase-camera clip
**every single frame the system believed was static had an ego estimate of
exactly 0.00**; not one was a still camera. The cost was not merely a wrong
mode: each spurious switch to static discards the background model and
suppresses output through the rebuild, so the detector was **blind for 18 % of
the clip** while looking straight at the target.

It now returns `None` for "unmeasurable" and the mode is held. Mode switches on
that clip went 4 → 1, blindness 18 % → 7 % (the remainder being the legitimate
start-up warm-up).

> Absence of evidence is not evidence of stillness. If you add another regime
> cue, give it the same treatment.

`camera_mode`, `ego_measured` and `warming_up` are in the JSONL export and
`tools/eval_real.py` reports mode stability, so this cannot silently return.

### 2.2 `edge_sensitivity: 0.0` was overfitted to synthetic open sky

The gradient-adaptive threshold in `_mask_moving` was measured on the synthetic
pan clip — open sky, no edges — where it can only cost recall, and was therefore
defaulted off, with a note reserving it for "genuinely cluttered backgrounds".

A handheld urban pan is precisely that case, and with the flat threshold it
produced **161 false tracks per minute and a false lock on 96 % of frames**. The
scene is nothing but the hard edges this term exists to tolerate.

| `edge_sensitivity` | urban false tracks/min | fpv F1 | fpv recall | synthetic cost |
|---|---|---|---|---|
| 0.0 | 161 | 48 % | 93.4 % | — |
| **0.15** | **65** | **85 %** | **92.8 %** | **none** |
| 0.25 | 46 | 84 % | 80.2 % | pan recall −10 pts |
| 0.55 | — | 40 % | 25.8 % | severe |

0.15 is the setting that regresses nothing — the synthetic pan clip actually
improves. A missed target is gone; a false contact can still be filtered
downstream.

The *code* default said `0.55` while the config said `0.0`, so a missing key
changed behaviour drastically. They now agree.

### 2.3 The small-target path floods on real sensor noise

On the night clip this path produced 81 % of 28.5 detections per frame — about
23 phantom contacts every frame on a scene containing one real light.

The adaptive threshold was working correctly. The failure is that
`min_core_pixels` stops discriminating when noise is **spatially correlated**:
synthetic noise is independent per pixel and never clumps, while real sensor
noise is smeared by in-camera denoising and inter-frame compression into exactly
the connected cores this filter looks for.

Fixed with `max_candidates`, which keeps the strongest candidates by peak
residual rather than raising the threshold. Measured to be free on all three
synthetic scenarios.

### 2.4 `detection.interval > 1` detected nothing at all

Two features that are each correct in isolation and cannot both be true. The
pipeline skips detection on most frames and passes an empty list; `mark_missed`
then runs on every skipped frame; and its first rule is *"a tentative track that
misses even once is almost always noise"* — a good false-positive filter **when
the detector runs every frame**. With `interval: 3` every new track was deleted
before the next detection frame, so nothing could reach `min_hits`.

Measured: **0.0 % recall**, for a setting the docs advertised as "roughly
doubles throughput". The synthetic suite never covered it.

Fixed by telling the tracker the difference between *"the detector looked and
found nothing"* and *"the detector never ran"*
(`MultiObjectTracker.update(..., detections_valid=False)` →
`Track.predict(counts_as_miss=False)`). An empty list is not evidence of absence
unless somebody actually looked. fpv recall at interval 3: 0 → 80.6 %;
interval 1 unchanged to the decimal.

**The default stays `interval: 1`.** Now that it works, interval 2 is genuinely
better on the fpv clip (F1 84.6 → 86.4 %, ID switches 8 → 1) and measurably
worse on everything else — synthetic static F1 95.5 → 79.4 %, hard 87.7 → 74.3 %,
night 32.8 → 27.4 %. Tempting on one clip, wrong overall.

---

## 3. Measured dead ends — do not re-attempt without reading the numbers

* **Raising `mog2_learning_rate`** improves precision (77 → 88 %) but wrecks ID
  stability (1 → 15 switches) because slow-moving targets get absorbed into the
  background. ID stability matters more for cueing a gimbal.

* **Raising `small_target.threshold_scale` 1.0 → 2.5** fixes the night clip well
  (precision 11.4 → 40.5 % at unchanged recall) and costs **16 points of recall
  on the synthetic hard clip** (83.8 → 67.9 %), because that clip's hazy
  background has a non-zero residual percentile, so scaling it lifts the
  threshold above the 4–10 grey levels the faint target actually produces.
  Reverted. Handle noisy scenes with `max_candidates` instead.

* **Top-hat saliency on the raw frame** contributed *exactly nothing* (identical
  metrics on and off). A top-hat isolates small structures, it cannot amplify
  them: the target's response was 2–6 grey levels — it cannot exceed the
  target's own contrast — while cloud texture in the same frame reached 105.
  Cloud texture is static, so the background model removes it completely; that
  is why the *residual* separates what the raw frame cannot.

* **Temporal accumulation** (`acc <- decay*dilate(acc) + residual`) raised recall
  84.7 → 88.3 % but collapsed precision 89.2 → 2.4 %. Isotropic dilation spreads
  evidence in every direction each frame, so any persistent residual blooms into
  an ever-growing disc. Proper track-before-detect needs a bank of accumulators
  shifted along hypothesised target *velocities*. Code remains,
  `temporal_decay: 0.0`.

* **Fragment merging** was built on the theory that faint targets split into
  several detections and spawn duplicate tracks. Measurement showed **never more
  than one detection per frame** near the target — the duplicates were stale
  coasting tracks. The code is harmless and correct, but solved a problem that
  did not exist here.

* **Raising `small_target.min_residual`** to buy back precision destroys the gain
  it exists for (6 → 66 % recall, 8 → 62 %). The threshold must stay low;
  precision is recovered by the core/displacement/`min_hits` filters instead.

* **Running the sensitive path in moving-camera mode** cost 18 points of
  precision (82.5 → 64.5 %) for no recall gain — its thresholds assume the clean
  residual a background model gives, not a registration-error-laden frame
  difference. Hence `small_target.static_only: true`.

---

## 4. Process note: ground truth nobody checked

The first ground truth produced by `tools/label_real.py` reported
**"600/600 frames labelled"** and was completely wrong — the correlation tracker
had drifted onto the horizon within 100 frames and confidently labelled a patch
of hillside as the aircraft.

It was caught only because the tool writes a contact sheet. Ground truth nobody
looked at is worse than no ground truth, because it gets believed and then
everything measured against it is fiction.

**Look at the check sheet, every time.**

---

## 5. Open issues

1. **ID stability regressed on real footage** (fpv 1 → 8 switches, night 9 → 17)
   as a side effect of the precision work. For a system whose purpose is cueing
   a gimbal, this is the weakest number in the system. The mechanism is
   understood — a tighter detector drops out more often, and a dropout past
   `coast_budget` ends the track — so the levers are `tracking.coast_floor`,
   `max_age` and the re-association gates, not the detector.

2. **Mean IoU on the fpv clip fell 0.37 → 0.29.** The gradient-adaptive
   threshold keeps only the part of a target sitting on low-gradient background,
   so boxes are tighter than the object. Harmless for cueing (the centre is
   right), bad for any future size- or range-estimation work.

3. **Get real drone footage.** None of the clips used so far contains an actual
   multirotor. Until one does, no claim about drone detection in this repository
   has been tested.
