"""Motion-based airborne object detection.

Why motion first: a small UAS at range occupies a handful of pixels and carries
almost no texture, which is exactly the regime where appearance-based CNNs are
weakest and where "it moves differently from the sky behind it" is the strongest
available cue. This detector needs no weights and no GPU, so the system is
useful the moment it is installed.

Two regimes are supported and can be switched automatically per frame:

``static``
    The camera does not move (tripod, fixed mast). A MOG2 mixture-of-Gaussians
    background model adapts to the sky and flags anything that deviates.

``moving``
    The camera pans/tilts (gimbal, handheld). A background model is meaningless
    here because the whole scene translates, so instead the previous frame is
    warped into the current frame's coordinate system using a RANSAC-estimated
    similarity transform from sparse optical flow, and the residual after
    warping is thresholded. Global camera motion cancels; independently moving
    objects survive.

Deliberately *not* filtered here: small blobs. A distant quadcopter can be
12 px. Rejecting small candidates per-frame would throw the target away with the
noise, so this stage stays permissive and false positives are killed temporally
by the tracker's ``min_hits`` promotion gate instead.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Detection, Detector
from .small_target import SmallTargetEnhancer

__all__ = ["MotionDetector"]


class MotionDetector(Detector):
    """Background/ego-motion based blob detector."""

    name = "motion"

    def __init__(self, cfg) -> None:
        """Args:
        cfg: The ``detection.motion`` config node (plus frame-level context).
        """
        self.cfg = cfg
        self.camera_motion = str(cfg.get("camera_motion", "auto")).lower()
        self.ego_threshold = float(cfg.get("ego_motion_threshold", 1.5))

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.get("mog2_history", 320)),
            varThreshold=float(cfg.get("mog2_var_threshold", 28.0)),
            detectShadows=False,
        )
        self._learning_rate = float(cfg.get("mog2_learning_rate", 0.006))

        # Sensor settling. A background model built at the steady-state learning
        # rate bakes whatever was in the first frames into the background - so a
        # target present at startup leaves a permanent "ghost" at its initial
        # position, which is detected every frame until the model finally adapts
        # (tens of seconds at a slow rate) and which seeds a stationary phantom
        # track. During warm-up the model therefore adapts fast and emits nothing.
        self._warmup_frames = int(cfg.get("warmup_frames", 30))
        self._warmup_learning_rate = float(cfg.get("mog2_warmup_learning_rate", 0.12))
        self._frames_seen = 0

        # Mode-switch hysteresis (see _select_mode).
        self._ego_leave_ratio = float(cfg.get("ego_leave_ratio", 0.4))
        self._settle_frames = int(cfg.get("camera_settle_frames", 8))
        self._still_frames = 0

        k_open = int(cfg.get("open_kernel", 3))
        k_dil = int(cfg.get("dilate_kernel", 5))
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        self._k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))
        self._dilate_iters = int(cfg.get("dilate_iterations", 2))

        # Sensitive path for small, low-contrast targets (see small_target.py).
        _st_cfg = cfg.get("small_target", {}) or {}
        self._small_target = SmallTargetEnhancer(_st_cfg)
        self._small_target_static_only = bool(_st_cfg.get("static_only", True))

        self._prev_gray: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._sensitive_mask: np.ndarray | None = None
        self.last_small_target_count: int = 0
        # Sparse correspondences from the current frame's flow estimate, reused
        # by the warp step so optical flow is computed at most once per frame.
        self._flow_cache: tuple[np.ndarray, np.ndarray] | None = None
        self.last_ego_motion: float = 0.0
        # False when this frame's ego motion could not be measured at all; the
        # value above is then stale. See _estimate_ego_motion.
        self.last_ego_measured: bool = True
        self.last_mode: str = "static"

    # ------------------------------------------------------------------ API
    def _reset_background(self) -> None:
        """Discard and rebuild the background model."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(self.cfg.get("mog2_history", 320)),
            varThreshold=float(self.cfg.get("mog2_var_threshold", 28.0)),
            detectShadows=False,
        )

    def reset(self) -> None:
        self._reset_background()
        self._prev_gray = None
        self._mask = None
        self._flow_cache = None
        self._frames_seen = 0

    @property
    def debug_mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def warming_up(self) -> bool:
        """True while the background model is still settling."""
        return self._frames_seen < self._warmup_frames

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Light blur suppresses sensor noise without erasing a 12 px target.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        self._frames_seen += 1

        ego = self._estimate_ego_motion(gray)
        # Keep the last measured value on the HUD when this frame could not be
        # measured, but flag it, so "0.0 px" never means "we have no idea".
        self.last_ego_measured = ego is not None
        if ego is not None:
            self.last_ego_motion = ego
        mode = self._select_mode(ego)
        self.last_mode = mode

        if mode == "moving":
            strong, residual = self._mask_moving(gray)
        else:
            strong, residual = self._mask_static(frame, gray)

        mask = self._clean(strong)

        # Sensitive path for small, low-contrast targets. Strictly additive:
        # the primary mask above is never weakened, so this cannot remove a
        # detection the detector would previously have made.
        # The sensitive path assumes a *clean* residual. A background model
        # delivers one (measured p99.9 of 3 grey levels); a motion-compensated
        # frame difference does not - registration error leaves a heavy tail
        # that the same thresholds read as targets. Measured on the pan clip,
        # running it in moving mode cost 18 points of precision for no recall
        # gain, so by default it is restricted to static mode.
        self.last_small_target_count = 0
        sensitive_allowed = not (self._small_target_static_only and mode == "moving")
        if not self.warming_up and sensitive_allowed:
            sensitive = self._small_target.compute(gray, residual)
            if sensitive.any():
                self._sensitive_mask = sensitive
                self.last_small_target_count = self._small_target.last_candidate_count
                mask = cv2.bitwise_or(mask, sensitive)
            else:
                self._sensitive_mask = None
        else:
            self._sensitive_mask = None

        self._mask = mask
        self._prev_gray = gray

        # Emit nothing until the model has settled, so start-up artefacts never
        # reach the tracker and seed phantom tracks.
        if self.warming_up:
            return []

        return self._contours_to_detections(mask, gray, frame)

    # -------------------------------------------------------------- internals
    def _select_mode(self, ego_motion: float | None) -> str:
        """Pick the detection regime, with hysteresis and model invalidation.

        Two failure modes have to be avoided here, and both are subtle.

        *Mode flapping.* A gimbal slewing sinusoidally is momentarily stationary
        at each end of its sweep. A single threshold would flip modes twice per
        sweep, so entering and leaving "moving" use different thresholds and
        leaving additionally requires the camera to stay still for several
        frames.

        *Stale background model.* MOG2 models a scene at a fixed alignment. Once
        the camera has moved, every statistic it holds refers to a different
        part of the world; resuming static mode with that model produces
        foreground almost everywhere. So when the camera comes to rest the model
        is discarded and rebuilt from scratch, with output suppressed until it
        has settled - exactly as at start-up.
        """
        if self.camera_motion in ("static", "moving"):
            return self.camera_motion

        # Unmeasurable ego motion: hold the current regime and, above all, do
        # not treat it as "still". Switching to static here is what discarded
        # the background model over featureless sky. Holding is also the safe
        # option for the mask: in moving mode a failed flow estimate yields an
        # empty mask (no detections this frame) rather than a garbage one.
        if ego_motion is None:
            self._still_frames = 0
            return self.last_mode

        enter = self.ego_threshold
        leave = self.ego_threshold * self._ego_leave_ratio

        if self.last_mode == "moving":
            if ego_motion < leave:
                self._still_frames += 1
            else:
                self._still_frames = 0
            if self._still_frames < self._settle_frames:
                return "moving"
            # Camera has come to rest: the old background model is meaningless.
            self._reset_background()
            self._frames_seen = 0
            self._still_frames = 0
            return "static"

        if ego_motion > enter:
            self._still_frames = 0
            return "moving"
        return "static"

    def _estimate_ego_motion(self, gray: np.ndarray) -> float | None:
        """Median displacement of static scene features between frames, in px.

        Returns ``None`` when the displacement could not be *measured* - no
        previous frame, too few trackable corners, or optical flow failing.

        That distinction is not pedantry, it was a real and expensive bug. This
        function used to return ``0.0`` on every one of those paths, which the
        mode selector reads as "the camera is perfectly still". Measured on
        test3 (a chase camera that never stops moving) *every single frame the
        system believed was static had an ego estimate of exactly 0.00* - not
        one of them was a still camera, they were all measurement failures over
        clear blue sky, where there is nothing to track. The consequence was not
        just a wrong mode: each spurious switch to static discarded the
        background model and suppressed output through the rebuild, so the
        detector was blind for 18 % of the clip while looking straight at the
        target.

        Absence of evidence is not evidence of stillness. The caller must decide
        what to do when the answer is unknown, and it must not be "assume the
        best case".
        """
        self._flow_cache = None
        if self._prev_gray is None:
            return None

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=int(self.cfg.get("stab_max_features", 900)),
            qualityLevel=float(self.cfg.get("stab_quality", 0.012)),
            minDistance=int(self.cfg.get("stab_min_distance", 8)),
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) < 12:
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            return None

        ok = status.reshape(-1) == 1
        if ok.sum() < 12:
            return None

        p0 = prev_pts.reshape(-1, 2)[ok]
        p1 = next_pts.reshape(-1, 2)[ok]
        # Cache for the warp step so flow is computed only once per frame.
        self._flow_cache = (p0, p1)
        return float(np.median(np.linalg.norm(p1 - p0, axis=1)))

    def _mask_static(
        self, frame: np.ndarray, gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MOG2 foreground mask for a fixed camera, plus a continuous residual.

        MOG2 only exposes a binary decision, but the sensitive small-target path
        needs *graded* motion evidence - "did this pixel change a little" rather
        than "did it change enough". That is recovered by differencing against
        the model's own background image.
        """
        rate = self._warmup_learning_rate if self.warming_up else self._learning_rate
        strong = self._bg.apply(frame, learningRate=rate)

        residual = np.zeros_like(gray)
        background = self._bg.getBackgroundImage()
        if background is not None and background.size:
            bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            # Match the pre-processing applied to `gray` so the difference
            # reflects scene change rather than differing blur.
            bg_gray = cv2.GaussianBlur(bg_gray, (3, 3), 0)
            residual = cv2.absdiff(gray, bg_gray)
        return strong, residual

    def _mask_moving(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Residual after cancelling global camera motion.

        The transform is estimated as a partial affine (rotation + uniform scale
        + translation) with RANSAC, which is the right model for a pan/tilt head
        and is far more stable to estimate from few points than a full homography.
        """
        empty = np.zeros(gray.shape, dtype=np.uint8)
        prev = self._prev_gray
        flow = self._flow_cache
        if prev is None or flow is None:
            return empty, empty

        p0, p1 = flow
        matrix, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.5,
            maxIters=2000, confidence=0.995,
        )
        if matrix is None:
            return empty, empty

        h, w = gray.shape
        warped = cv2.warpAffine(
            prev, matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        diff = cv2.absdiff(gray, warped)

        # Warp artefacts concentrate at the frame border; suppress a thin rim.
        rim = max(2, int(0.01 * max(w, h)))
        diff[:rim, :] = 0
        diff[-rim:, :] = 0
        diff[:, :rim] = 0
        diff[:, -rim:] = 0

        # --- gradient-adaptive threshold -------------------------------------
        # A global threshold is unusable here. No estimated transform is exact,
        # and a sub-pixel registration error on a hard edge (a tree line, a roof)
        # produces a residual as large as a real target's, so a threshold low
        # enough to see a drone against sky lights up every edge in the scene.
        #
        # The discriminator is *where the structure lives*. Misregistration
        # residual requires structure in the previous frame at that location;
        # an object that has just arrived does not have any, because the warped
        # previous frame shows only the background it is now covering. Basing
        # the local threshold on the warped previous frame's gradient therefore
        # suppresses edge artefacts while leaving new arrivals fully sensitive -
        # and as a bonus it also suppresses the trailing "ghost" at the object's
        # previous position, where the old frame does have strong structure.
        base = float(self.cfg.get("diff_threshold", 16))
        # Keep this in step with config/default.yaml - the two disagreeing (the
        # code said 0.55 while the config said 0.0) means a missing key silently
        # changes behaviour far more than anyone would expect from a default.
        sensitivity = float(self.cfg.get("edge_sensitivity", 0.15))

        gx = cv2.Sobel(warped, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(warped, cv2.CV_16S, 0, 1, ksize=3)
        grad = cv2.addWeighted(
            cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0
        )
        # Dilate so the tolerance covers the whole edge neighbourhood, not just
        # the single pixel where the gradient peaks.
        grad = cv2.dilate(grad, self._k_dilate, iterations=1)

        threshold_map = np.clip(
            base + sensitivity * grad.astype(np.float32), 0, 255
        ).astype(np.uint8)
        mask = np.where(diff > threshold_map, np.uint8(255), np.uint8(0))
        return mask, diff

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup: kill speckle, then re-join blob fragments.

        Order matters. Opening first removes isolated noise pixels; dilating
        afterwards merges the rotor-arm fragments of one airframe into a single
        contour instead of reporting four contacts for one drone.
        """
        if mask is None or mask.size == 0:
            return mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
        mask = cv2.dilate(mask, self._k_dilate, iterations=self._dilate_iters)
        return mask

    def _contours_to_detections(
        self, mask: np.ndarray, gray: np.ndarray, frame: np.ndarray
    ) -> list[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        h, w = mask.shape
        frame_area = float(h * w)
        min_area = float(self.cfg.get("min_area", 12))
        max_area = float(self.cfg.get("max_area_ratio", 0.08)) * frame_area
        min_aspect = float(self.cfg.get("min_aspect", 0.22))
        max_aspect = float(self.cfg.get("max_aspect", 4.5))
        min_fill = float(self.cfg.get("min_fill_ratio", 0.18))
        border = int(self.cfg.get("border_margin", 2))

        candidates: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / float(bh)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            # Fill ratio rejects thin diagonal streaks (wires, contrails, flare)
            # that pass the aspect test only because their bbox is square.
            if area / float(bw * bh) < min_fill:
                continue

            if border > 0 and (
                x <= border or y <= border or x + bw >= w - border or y + bh >= h - border
            ):
                continue

            bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
            contrast = self._local_contrast(gray, x, y, bw, bh)
            score = self._score(area, contrast)

            # Record which path found this contact: useful when tuning, and it
            # lets the HUD mark faint contacts that rest on weaker evidence.
            via_sensitive = False
            if self._sensitive_mask is not None:
                cy_i = min(max(y + bh // 2, 0), self._sensitive_mask.shape[0] - 1)
                cx_i = min(max(x + bw // 2, 0), self._sensitive_mask.shape[1] - 1)
                via_sensitive = bool(self._sensitive_mask[cy_i, cx_i])

            candidates.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    label="CONTACT",
                    source=self.name,
                    meta={
                        "area": area,
                        "contrast": contrast,
                        "aspect": aspect,
                        "mode": self.last_mode,
                        "small_target": via_sensitive,
                    },
                )
            )

        candidates = self._merge_fragments(candidates)

        # Cap per-frame output: under a pathological mask (e.g. sudden exposure
        # change) thousands of contours would otherwise stall the tracker.
        max_det = int(self.cfg.get("max_detections", 60))
        if len(candidates) > max_det:
            candidates.sort(key=lambda d: d.score, reverse=True)
            candidates = candidates[:max_det]
        return candidates

    def _merge_fragments(self, detections: list[Detection]) -> list[Detection]:
        """Merge detections that are almost certainly one broken-up object.

        A faint target does not threshold into one clean blob - it breaks into
        two or three pieces that drift in and out between frames. Each piece
        becomes its own detection, then its own track, and the display ends up
        with two or three contacts sitting on one drone while the target lock
        alternates between them.

        Morphological dilation is supposed to prevent this, but it cannot be
        made large enough to bridge the gaps without also merging genuinely
        separate contacts. Grouping at box level is safer: boxes are expanded by
        a fraction of their own size, anything still overlapping after that is
        treated as one object, and the group is replaced by its union box. The
        margin scales with object size, so distant fragments merge while two
        separate contacts far apart never do.
        """
        if len(detections) < 2:
            return detections

        margin = float(self.cfg.get("fragment_merge_margin", 0.6))
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        pad = np.maximum(2.0, sizes * margin)

        expanded = boxes.copy()
        expanded[:, 0] -= pad
        expanded[:, 1] -= pad
        expanded[:, 2] += pad
        expanded[:, 3] += pad

        n = len(detections)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i in range(n):
            for j in range(i + 1, n):
                if (
                    expanded[i, 0] < expanded[j, 2]
                    and expanded[j, 0] < expanded[i, 2]
                    and expanded[i, 1] < expanded[j, 3]
                    and expanded[j, 1] < expanded[i, 3]
                ):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        if len(groups) == n:
            return detections

        merged: list[Detection] = []
        for members in groups.values():
            if len(members) == 1:
                merged.append(detections[members[0]])
                continue
            group_boxes = boxes[members]
            union_box = np.array(
                [
                    group_boxes[:, 0].min(),
                    group_boxes[:, 1].min(),
                    group_boxes[:, 2].max(),
                    group_boxes[:, 3].max(),
                ],
                dtype=np.float32,
            )
            parts = [detections[i] for i in members]
            best = max(parts, key=lambda d: d.score)
            meta = dict(best.meta)
            meta["fragments"] = len(parts)
            meta["area"] = float(sum(p.meta.get("area", 0.0) for p in parts))
            merged.append(
                Detection(
                    bbox=union_box,
                    score=best.score,
                    label=best.label,
                    source=self.name,
                    meta=meta,
                )
            )
        return merged

    @staticmethod
    def _local_contrast(gray: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
        """Mean absolute intensity difference between the blob and its surround.

        A real object against sky has high local contrast; a background-model
        artefact usually does not. Used to weight detection confidence.
        """
        h, w = gray.shape
        pad = max(4, int(0.75 * max(bw, bh)))
        ox1, oy1 = max(0, x - pad), max(0, y - pad)
        ox2, oy2 = min(w, x + bw + pad), min(h, y + bh + pad)

        inner = gray[y : y + bh, x : x + bw]
        outer = gray[oy1:oy2, ox1:ox2]
        if inner.size == 0 or outer.size <= inner.size:
            return 0.0

        inner_mean = float(inner.mean())
        # Surround mean excluding the object itself.
        outer_sum = float(outer.sum()) - float(inner.sum())
        outer_count = outer.size - inner.size
        surround_mean = outer_sum / max(1, outer_count)
        return abs(inner_mean - surround_mean)

    def _score(self, area: float, contrast: float) -> float:
        """Map blob statistics to a confidence in [0, 1].

        Intentionally conservative: motion evidence alone rarely exceeds ~0.8,
        leaving headroom for fusion with a CNN detector to push a contact higher.
        """
        base = float(self.cfg.get("conf_base", 0.30))
        w_area = float(self.cfg.get("conf_area_weight", 0.25))
        w_contrast = float(self.cfg.get("conf_contrast_weight", 0.45))
        ref_contrast = max(1.0, float(self.cfg.get("contrast_reference", 45.0)))

        # Saturating response: past a few hundred px more area adds no certainty.
        area_term = float(np.tanh(area / 900.0))
        contrast_term = float(np.clip(contrast / ref_contrast, 0.0, 1.0))
        return float(np.clip(base + w_area * area_term + w_contrast * contrast_term, 0.0, 1.0))
