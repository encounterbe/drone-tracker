# Architecture & Working Notes

Guidance for anyone working in this repository. Read §6 and
`FINDINGS.md` before changing anything in `detection/` or `tracking/`.

> The owner is a German speaker. The repository is English throughout — code,
> comments, `README.md` and this file. Comments explain *why*, and several of
> them record a specific measured failure mode; those are the expensive part of
> this repository, so keep them.

---

## 1. What this is

A modular computer-vision system that detects and tracks small airborne objects
(UAS/drones) in video, with an operator HUD: reticle, target brackets, digital
zoom, focus measurement, telemetry panels.

**Current stage: 1 of 3.** It runs on recorded MP4 files. Stage 2 (live camera)
already works via `--source 0` but is untested against real hardware. Stage 3
(gimbal actuation) now has firmware and a host-side controller, but has never
been run against hardware — see §8.

The system was built to eventually drive a physical pan/tilt gimbal (ESP32 +
2 servos), and the architecture reflects that: `tracker.primary` exposes exactly
the target-offset signal a gimbal controller consumes.

---

## 2. Running it

```bash
pip install -r requirements.txt

# Generate a synthetic test clip with ground truth (no drone footage needed)
python tools/make_test_video.py --out output/test_drone.mp4

# Interactive operator display
python -m src.main --source output/test_drone.mp4

# Headless: annotated video + per-frame track log
python -m src.main --source clip.mp4 --no-display \
    --record output/annotated.mp4 --export-tracks output/tracks.jsonl

# Score the result against ground truth
python tools/evaluate.py --tracks output/tracks.jsonl \
    --ground-truth output/test_drone_groundtruth.json

# Tune anything without editing config files
python -m src.main --source clip.mp4 --set detection.motion.min_area=30
```

Keyboard: `SPACE` pause · `n` step (while paused) · `+`/`-` zoom · arrows pan ·
`0` reset zoom · `t` cycle target · `y` release manual lock · `f` focus assist ·
`h` HUD · `m` motion mask · `g` grid · `d` raw detections · `a` auto-follow ·
`s` snapshot · `q` quit.

---

## 3. Architecture

```
src/
├── main.py              CLI entry point; flag→config resolution
├── pipeline.py          frame loop; owns ORDER of operations only
├── detection/
│   ├── base.py          Detection dataclass + Detector ABC  ← the data contract
│   ├── motion.py        MOG2 / ego-motion-compensated differencing
│   ├── small_target.py  sensitive path for small dim targets
│   ├── cnn.py           optional YOLO backend (graceful if absent)
│   ├── fusion.py        evidence combination across backends
│   └── __init__.py      build_detector(cfg) factory
├── tracking/
│   ├── kalman.py        8-state constant-velocity filter
│   ├── track.py         Track + lifecycle state machine
│   └── tracker.py       Hungarian association + primary-target selection
├── ui/
│   ├── hud.py           all overlay drawing
│   ├── viewport.py      digital zoom + source↔screen coordinate mapping
│   ├── focus.py         sharpness metric + focus assist
│   └── theme.py         colour themes + drawing primitives
└── utils/
    ├── config.py        YAML load + deep-merge + dotted overrides
    ├── geometry.py      box maths (IoU, NMS, conversions)
    ├── video.py         VideoSource / VideoRecorder
    └── perf.py          per-stage timing

tools/
├── make_test_video.py   synthetic clip generator + ground truth
│                        (--target-blur / --haze / --size-range build hard cases)
└── evaluate.py          recall / precision / IoU / ID-switch scoring
```

### Per-frame flow

```
capture → detect (every Nth frame) → track → select primary
        → focus measure → viewport update → HUD render → display/record/export
```

### Invariants — do not break these

1. **Detection and tracking consume the untouched source frame.** Zoom and focus
   assist are *display only*. An operator aid must never change what the system
   believes it sees.
2. **All detection/tracking geometry is in source-pixel coordinates.** The HUD
   converts to screen space via `viewport.map_point` / `map_box`. Never draw
   track geometry without mapping it, or overlays drift when zoomed.
3. **`pipeline.py` contains no algorithms.** It owns sequencing. New capability
   goes in `detection/`, `tracking/` or `ui/`.
4. **New detectors implement `Detector.detect()` and return `Detection`.**
   Nothing downstream should need changing.

---

## 4. Data contracts

`Detection` (`src/detection/base.py`)
```python
bbox: np.ndarray  # [x1,y1,x2,y2] float32, source pixels
score: float      # [0,1]
label: str        # "CONTACT" | "AIR-CONTACT" | model class
source: str       # "motion" | "cnn" | "fusion"
meta: dict        # area, contrast, aspect, mode, ...
```

`Track` (`src/tracking/track.py`) — states `TENTATIVE → CONFIRMED → COASTING → LOST`.
`to_dict()` defines the JSONL export schema. Useful members: `.center`, `.bbox`,
`.velocity` (px/frame), `.speed_px_per_s(fps)`, `.heading_deg()`, `.confidence`,
`.uncertainty`, `.track_quality`.

Track export is one JSON object per frame:
`{frame, timestamp, detections[], tracks[], primary_id, focus{}, zoom}`.

---

## 5. Measured baseline

Against `tools/make_test_video.py` ground truth, 35-frame warm-up excluded.
**Re-run all three after any detection or tracking change** — this is the
regression check, and several plausible-sounding "improvements" measurably made
things worse.

Reproduce:

```bash
python tools/make_test_video.py --out output/test_drone.mp4 --seconds 20
python tools/make_test_video.py --out output/test_pan.mp4 --seconds 12 --camera-pan 0.8
python tools/make_test_video.py --out output/test_hard.mp4 --seconds 16 \
    --target-blur 1.8 --haze 0.62 --size-range 8 14 --no-bird

python -m src.main --source output/test_X.mp4 --no-display --export-tracks /tmp/t.jsonl
python tools/evaluate.py --tracks /tmp/t.jsonl \
    --ground-truth output/test_X_groundtruth.json --warmup 35
```

Or in one command: `python tools/bench_synthetic.py`.

| Scenario | Recall | Precision | F1 | Mean IoU | ID sw |
|---|---|---|---|---|---|
| Static camera, 600 frames | 99.6 % | 91.6 % | 95.5 % | 0.520 | 2 |
| Panning camera, 360 frames | 100 % | 79.1 % | 88.3 % | 0.473 | 1 |
| **Hard**: 8–14 px, defocused, hazy | 80.4 % | 96.3 % | 87.7 % | 0.381 | 7 |

These are the current measured numbers and they differ slightly from the ones
this table carried before the real-footage work (hard recall 83.8 → 80.4 %).
That is not a code change: forcing `camera_motion=static`, which bypasses
everything that was touched, reproduces the new numbers exactly. The clips are
seeded and deterministic, so the remaining suspect is a numpy/OpenCV version
difference. Trust your own run and compare against *it*.

**A second regression suite exists for real footage** — `tools/bench_real.py`.
It needs clips under `video/`, which are not in the repository. See
`docs/FINDINGS.md` section 2 for what was measured there and what it changed.

The hard clip is the small-dim-target case: 8–14 px targets at roughly **4 grey
levels of contrast against a noise sigma of 3**. Before the small-target work it
scored 59.1 % recall with an acquisition latency of 135 frames; it is now 83.8 %
with latency 0. The static clip ended up *above* its own original baseline
(F1 92.9 → 95.1 %), so the sensitivity gain did not cost accuracy elsewhere.

Throughput on a 2-core sandbox at 1280×720: ~18 FPS (`detect` ≈ 33 ms,
`render` ≈ 17 ms). `detection.interval=3` roughly doubles it. A normal laptop is
considerably faster; this is not an optimised build.

Remaining precision loss is mostly **coasting tracks on the bird distractor** —
a real object held through detection dropouts, drifting past the evaluator's
tight match radius. That is arguably correct behaviour, not a bug.

---

## 6. Bugs that were found and fixed — read before "improving" things

> **`docs/FINDINGS.md` is the full record**, including three bugs found only
> once real footage was used (the ego-motion estimate failing open, an
> `edge_sensitivity` default overfitted to synthetic sky, and
> `detection.interval > 1` yielding 0 % recall) and every measured dead end.
> Read it before touching `detection/` or `tracking/`.

These were all found by measurement, not by reading the code. Each looked
plausible in its broken state.

1. **MOG2 initialisation ghost.** A target present in the first frames gets
   baked into the background model, leaving a *stationary blob that is detected
   every frame* at the target's start position, which seeds a permanent phantom
   track. Fixed with a warm-up phase (fast adaptation + suppressed output):
   `detection.motion.warmup_frames`.

2. **Mode flapping with a stale background model** (this one was severe:
   precision 4 %). In `camera_motion: auto`, a sinusoidally panning camera is
   momentarily still at each sweep end, flipping back to MOG2 — whose model was
   assembled from frames at *different* pan positions and therefore fires
   almost everywhere. Fixed with switch hysteresis plus discarding and
   rebuilding the model whenever the camera comes to rest.

3. **Coasting tails.** `max_age` extrapolated departed objects for a full
   second, painting phantom contacts. Fixed by terminating tracks whose
   prediction leaves the frame (coasting is for *occlusion*, not departure), and
   by scaling the coasting budget with track maturity (`Track.coast_budget`).

4. **Config writes silently discarded.** `Config.__getattr__` returned a fresh
   shallow copy, so `cfg.zoom["initial"] = 3.0` mutated a temporary — `--zoom`,
   `--theme` and `--backend` all did nothing, with no error. Nested dicts are
   now converted to `Config` at insertion time. **If you touch `utils/config.py`,
   verify write-through still works.**

5. **Mahalanobis gating on size.** The association gate originally used all four
   measurement dimensions (cx, cy, w, h). For a faint, barely-resolved target the
   apparent box size swings wildly frame to frame, so the size terms dominated the
   distance and vetoed associations whose *position* agreed perfectly — the track
   died and the same object was reborn under a new ID. Gating on position only
   (2 DoF) lifted hard-clip recall 81.6 → 84.7 % and cut ID switches 22 → 15.

### The small-target path (added after the first delivery)

Goal: detect ~10 px defocused, hazy targets without losing accuracy elsewhere.
What actually worked, in order of contribution:

1. **Re-threshold the MOG2 residual adaptively.** MOG2 exposes only a binary
   decision, and at ~4 grey levels of contrast it says "background". But the
   evidence exists — differencing against `getBackgroundImage()` recovers it, and
   there the target is the *frame-wide maximum* in 8 of 10 sampled frames
   (background p99.9 = 3, target 4–10). The target was never undetectable; it was
   below a fixed threshold. Recall 59.1 → 80.4 %, acquisition latency 135 → 0.
2. **Connected-core requirement** (`min_core_pixels`, before dilation). Lets the
   threshold stay low: noise crosses a permissive threshold as isolated pixels,
   a target as a connected core. Hard-clip precision 89.2 → 90.8 %.
3. **Moving-target indication** (`tracking.min_displacement`). Every false
   contact that survived everything else came from a *stationary* source —
   residual at a high-contrast cloud edge left by an imperfect background model —
   with under 3 px total displacement, against 151 px for real tracks. Gating
   confirmation on displacement removed 100 % of them and 0 % of real targets,
   and pushed the static clip *above* its original baseline (F1 92.9 → 95.1 %).
4. **`min_hits` 3 → 4.** Hard-clip precision 89 → 93 % for under a point of recall.
5. **Freezing size extrapolation while coasting.** The constant-velocity model
   propagates width/height velocity too, so through a long gap the box collapsed
   to the 1 px floor. A degenerate box has zero IoU with *every* detection, so
   the track could not re-acquire its own target when detections returned. Worth
   4.5 points of recall (79.3 → 83.8 %) and 3 fewer ID switches on its own.

Tuning note: `min_displacement: 6.0` is chosen so that **no scenario regresses**
below the original baseline. Lowering it to 3.0 buys ~4.5 points of faint-target
recall (79.3 → 83.8 %) at the cost of ~7 points of static-clip precision
(90.7 → 83.2 %). Pick per deployment; the default is the conservative one.

### Measured dead ends

Each of these sounded right and was measured to be wrong. Do not re-attempt
without reading the numbers.

* **Raising `mog2_learning_rate`** improves precision (77 → 88 %) but wrecks ID
  stability (1 → 15 switches) because slow-moving targets get absorbed into the
  background. ID stability matters more for cueing a gimbal.
* **`edge_sensitivity` > 0** (gradient-adaptive differencing threshold) costs
  recall fast on open sky (0.25 → 71 %, 0.55 → 39 %) for modest precision gain.
  Defaulted to 0; only useful for genuinely cluttered backgrounds.
* **Top-hat saliency on the raw frame** for small-target detection contributed
  *exactly nothing* (identical metrics on and off). A top-hat isolates small
  structures, it cannot amplify them: the target's response was 2–6 grey levels —
  it cannot exceed the target's own contrast — while cloud texture in the same
  frame reached 105. Cloud texture is static, so the background model removes it
  completely; that is why the *residual* separates what the raw frame cannot.
* **Temporal accumulation** (`acc <- decay*dilate(acc) + residual`) raised recall
  84.7 → 88.3 % but collapsed precision 89.2 → 2.4 %. Isotropic dilation spreads
  accumulated evidence in every direction each frame, so any persistent residual
  blooms into an ever-growing disc. Proper track-before-detect needs a bank of
  accumulators shifted along hypothesised target *velocities*. Code is left in
  place, `temporal_decay: 0.0`.
* **Fragment merging** (`fragment_merge_margin`) was built on the theory that
  faint targets split into several detections and spawn duplicate tracks.
  Measurement showed **never more than one detection per frame** near the
  target — the duplicates were stale coasting tracks instead. The code is
  harmless and correct, but it solved a problem that did not exist here. It may
  earn its keep on real footage; verify before relying on it.
* **Raising `small_target.min_residual`** to buy back precision destroys the
  gain it exists for (6 → 66 % recall, 8 → 62 %). The threshold must stay low;
  precision is recovered by the core/displacement/`min_hits` filters instead.
* **Running the sensitive path in moving-camera mode** cost 18 points of
  precision (82.5 → 64.5 %) for no recall gain — its thresholds assume the clean
  residual a background model gives, not a registration-error-laden frame
  difference. Hence `small_target.static_only: true`.

---

## 7. Known limitations — be honest about these

* **The CNN backend is not a drone detector.** Stock `yolov8n.pt` is COCO, which
  has no UAS class. It currently maps `airplane`/`bird`/`kite` to `AIR-CONTACT`
  as a stand-in so the plumbing is testable. It will miss real drones and fire
  on real birds. See §8 for training a real one.
* **No drone/bird discrimination.** The motion detector reports "something is
  moving up there". Telling a quadcopter from a bird needs either a trained
  classifier or micro-Doppler/rotor-frequency analysis (see §8).
* **"Focus" is measurement + display assist, not autofocus.** There is no lens
  motor to drive. `FocusAnalyzer.trend` is the gradient signal a hill-climbing
  AF loop would need if a motorised lens is ever attached.
* **Ranges, altitudes and real-world speeds are not computed.** Everything is in
  pixels. Real units need camera intrinsics plus either known target size or a
  second sensor.
* **Validated only on synthetic footage.** The numbers in §5 come from
  `make_test_video.py`. Real sensor noise, rolling shutter, haze, exposure
  changes and JPEG artefacts are not represented. **Get real footage before
  trusting any of it.**

---

## 8. Roadmap / next tasks

### Stage 1 follow-ups (highest value first)

1. **Validate on real footage.** Everything above is synthetic. Record or source
   real drone clips; expect to retune `detection.motion.*`.
2. **Train a real UAS detector.** Datasets: Roboflow Universe drone datasets,
   the Drone-vs-Bird Detection Challenge set, Anti-UAV. Train YOLOv8s at
   `imgsz=1280` (small objects need the resolution), point
   `detection.cnn.weights` at the result, and run `--backend fusion`. The
   fusion path and the class-remap logic are already in place.
3. **Track-level classification.** Drone vs bird is much easier from a *track*
   than a frame: birds flap (periodic size/aspect oscillation) and turn
   erratically; multirotors hold straight legs and hover. The features
   (`trail`, `velocity`, `heading_deg`, aspect history) are already on `Track`.
4. **Tune assignment for crossing targets.** ID switches are 0 in the static
   case but 7 when panning — investigate with `tools/evaluate.py`.

### Stage 2 — live camera

`--source 0 --realtime` already runs. Needs: camera-specific tuning, exposure
lock (auto-exposure hunting looks like global motion), and a check that
`process_width` keeps throughput above the servo control rate.

### Stage 3 — gimbal actuation

The earlier ESP32 firmware in this project's history uses a line protocol
`"<pan>,<tilt>,<laser>\n"` at 115200 baud with a 1 s failsafe. To integrate:

* `src/control/gimbal.py` exists: a `GimbalController` that opens the serial
  port and converts `primary.center - frame_centre` (shown as `OFFSET` in the
  HUD) into pan/tilt demands via a PID loop. It needs `pyserial`, which is
  optional, and it deliberately does **not** follow a coasting track — chasing
  an extrapolated position moves the camera away from where the target was last
  actually seen.
* `firmware/gimbal/gimbal.ino` exists: ESP32 + ESP32Servo, the line protocol
  above, a 1 s on-device failsafe and software tilt limits. **It is a
  reimplementation from the protocol spec and has never been run against
  hardware.** Bench-test with the servos unloaded before mounting anything.
* Still to do: subscribe the controller in `pipeline.py` after `select primary`
  — no other file changes — and add a `control.gimbal` block to
  `config/default.yaml`. It is deliberately not wired in yet, so the default
  configuration cannot move hardware by accident.

**Note on the laser.** A 5 mW pointer cannot damage anything; the issue is
legal and safety-related, not technical. Directing a laser at aircraft is a
criminal offence in Germany (§ 315 StGB and Luftverkehrsgesetz provisions), and
a tracker that is slightly off, or that locks onto a manned aircraft instead of
the intended drone, points it somewhere nobody chose. If Stage 3 gets built, the
defensible version drives **the camera** — keeping the target framed, logging
the track — and leaves the laser output unwired. Raise this with the owner
rather than quietly wiring the laser to the lock signal.

---

## 9. Conventions

* Python ≥ 3.10, type hints throughout, `from __future__ import annotations`.
* NumPy for array maths; avoid per-pixel Python loops in the frame path.
* Every tunable belongs in `config/default.yaml` with a comment explaining
  *what it trades off* — never a bare magic number in code.
* Comments explain **why**, not what. Several in this codebase document a
  specific failure mode; keep those, they are the expensive knowledge.
* New algorithm parameters must be measurable via `tools/evaluate.py`.
* No new hard dependencies without discussion. Optional backends must degrade
  gracefully (`DetectorUnavailable` → log → fall back), never crash at frame 1.
