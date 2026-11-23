"""Small dim target enhancement.

The problem this solves
-----------------------
A UAS at range is a handful of pixels with very little contrast. Atmospheric
haze scatters airlight into the line of sight, which washes the target toward
the sky's own brightness. Measured on the project's hard test clip, a 10 px
target sits at roughly 4 grey levels of contrast against a sensor noise sigma
of ~3 - close to the single-frame noise floor.

MOG2 decides foreground with a per-pixel Mahalanobis-style criterion tuned to be
conservative, and at that contrast it simply says "background". But the
*evidence* is not gone: it is thrown away at the thresholding step. Differencing
against the model's own background image recovers it as a continuous residual,
and there the picture is very different. Measured on the hard clip:

===================  =========================
Background residual  median 0, p99 2, p99.9 3
Target residual      4-10, and the frame-wide
                     maximum in 8 of 10 frames
===================  =========================

The target is the brightest thing in the residual image. It was never
undetectable - it was below a fixed threshold. So this path re-thresholds the
residual *adaptively*, against a per-frame estimate of the background's own
residual level, instead of against a constant.

Why percentiles rather than median/MAD
--------------------------------------
The obvious robust estimator, ``median + k * MAD``, collapses here: the residual
is exactly zero across almost the whole frame, so both median and MAD are 0 and
the scale estimate carries no information. A high percentile (p99.5) of the
residual describes the tail that actually matters - "how large does this frame's
background residual get" - and adapts to noise level, exposure changes and
background-model quality on its own.

Measured dead end: top-hat saliency
-----------------------------------
The first implementation gated on bidirectional morphological top-hat saliency
computed on the greyscale frame, on the theory that a small compact blob is
locally distinctive. It was measured to contribute *nothing* (identical metrics
with the path on and off) and the reason is worth recording: a top-hat filter
isolates small structures, it cannot amplify them. The target's top-hat response
was 2-6 grey levels - it cannot exceed the target's own contrast - while cloud
texture elsewhere in the same frame reached 105. Ranking by saliency put the
target far down the list. Cloud texture is static, though, so the background
model removes it completely; that is why the residual separates what the raw
frame cannot. Do not reintroduce saliency gating on the raw frame.

What protects accuracy
----------------------
This module only ever *adds* candidates: the primary mask is untouched and is
OR-ed with the result, so no previously-made detection can be lost. Added
candidates then pass the same geometry filters as any other detection and, above
all, the tracker's ``min_hits`` gate. That division of labour is the whole
design - be permissive per frame, strict across time. Noise passing an
aggressive threshold lands at random positions and cannot be associated across
several consecutive frames; a real target can.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["SmallTargetEnhancer"]


class SmallTargetEnhancer:
    """Produces an extra detection mask for small, low-contrast targets."""

    def __init__(self, cfg) -> None:
        """Args:
        cfg: The ``detection.motion.small_target`` config node.
        """
        self.enabled = bool(cfg.get("enabled", True))
        self.background_percentile = float(cfg.get("background_percentile", 99.5))
        self.threshold_scale = float(cfg.get("threshold_scale", 1.0))
        self.threshold_margin = float(cfg.get("threshold_margin", 2.0))
        self.min_residual = float(cfg.get("min_residual", 4.0))
        self.min_core_pixels = int(cfg.get("min_core_pixels", 4))
        self.max_area = int(cfg.get("max_area", 900))
        self.max_candidates = int(cfg.get("max_candidates", 12))
        self.dilate_size = int(cfg.get("dilate_size", 3))
        self.sample_stride = max(1, int(cfg.get("sample_stride", 4)))

        d = max(1, self.dilate_size)
        self._se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

        # Temporal integration (see _accumulate).
        self.temporal_decay = float(cfg.get("temporal_decay", 0.0))
        m = max(1, int(cfg.get("motion_radius", 5)) | 1)
        self._se_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
        self._accumulator: np.ndarray | None = None

        # Diagnostics, surfaced on the HUD and useful when tuning.
        self.last_threshold: float = 0.0
        self.last_candidate_count: int = 0
        self.last_background_level: float = 0.0
        # Non-zero when candidates were dropped because the frame produced more
        # of them than a real scene plausibly contains.
        self.last_rejected_count: int = 0

    # ------------------------------------------------------------------ API
    def compute(self, gray: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Build the sensitive-path mask.

        Args:
            gray: Current frame, single channel. Unused by the current method,
                kept so alternative strategies can be tried without changing
                every call site.
            residual: Continuous motion evidence in grey levels - for a static
                camera the difference from the background model, for a moving
                one the motion-compensated frame difference.

        Returns:
            A uint8 binary mask of additional candidate regions.
        """
        if not self.enabled or residual is None or residual.size == 0:
            self.last_candidate_count = 0
            return np.zeros(gray.shape, dtype=np.uint8)

        evidence = self._accumulate(residual)

        # Subsampled percentile: a noise statistic does not need every pixel,
        # and the full-frame version costs about an order of magnitude more time
        # than the rest of this function put together.
        sample = evidence[:: self.sample_stride, :: self.sample_stride]
        background_level = float(np.percentile(sample, self.background_percentile))
        self.last_background_level = background_level

        threshold = max(
            self.min_residual,
            background_level * self.threshold_scale + self.threshold_margin,
        )
        self.last_threshold = threshold

        mask = (evidence > threshold).astype(np.uint8) * 255

        # Filter on the *raw* thresholded components, before any dilation. This
        # is the discriminator that lets the threshold stay low: at a threshold
        # this permissive, sensor noise does cross it, but it crosses as isolated
        # single pixels, whereas even a barely-resolved target puts several
        # adjacent pixels over the line. Requiring a connected core separates the
        # two without raising the threshold - which was measured to be the only
        # lever that does not immediately cost recall on faint targets.
        mask = self._filter_components(mask, evidence)

        # Only now join surviving pixels into coherent blobs. Deliberately no
        # opening: at this scale an opening erases the target along with the
        # noise, which is exactly the failure this path exists to avoid.
        return cv2.dilate(mask, self._se_dilate, iterations=1)

    # -------------------------------------------------------------- internals
    def _accumulate(self, residual: np.ndarray) -> np.ndarray:
        """Integrate residual evidence over time (track-before-detect, cheaply).

        In the worst frames the target's residual genuinely sits at or below the
        noise floor, so no single-frame threshold can recover it - the
        information is not there. Across frames it is: the target contributes
        residual at (nearly) the same place every frame, while sensor noise is
        independent from frame to frame and averages away.

        The accumulator is

            acc <- decay * dilate(acc) + residual

        The dilation is what makes this work for a *moving* target: it lets
        yesterday's evidence count for anywhere within one frame's travel, so a
        drone crossing the sky keeps accumulating instead of smearing into
        nothing. A persistent target converges to roughly ``1/(1-decay)`` times
        its per-frame residual - a gain of about 3x at decay 0.7 - while a noise
        spike contributes once and then decays.

        Set ``temporal_decay: 0`` to disable and threshold the raw residual.
        """
        if self.temporal_decay <= 0.0:
            self._accumulator = None
            return residual.astype(np.float32)

        current = residual.astype(np.float32)
        if self._accumulator is None or self._accumulator.shape != current.shape:
            self._accumulator = current.copy()
            return self._accumulator

        spread = cv2.dilate(self._accumulator, self._se_motion, iterations=1)
        self._accumulator = self.temporal_decay * spread + current
        return self._accumulator

    def reset(self) -> None:
        """Drop temporal state (called on seek/loop)."""
        self._accumulator = None

    def _filter_components(self, mask: np.ndarray, evidence: np.ndarray) -> np.ndarray:
        """Keep connected components that are plausible small targets.

        Size bounds
        -----------
        Lower (``min_core_pixels``): isolated noise pixels are discarded while a
        genuine target's connected core survives.

        Upper (``max_area``): a large region here means the threshold was
        defeated by something global - a cloud front moving through, an exposure
        shift, a gust rocking the mast - not by a distant aircraft. The primary
        detector already handles anything big, so discarding it costs nothing
        and keeps this path from flooding the tracker when conditions change.

        Count bound (``max_candidates``)
        --------------------------------
        The size bounds alone were measured to be insufficient on real footage.
        On a night CCTV clip this path produced 81 % of 28.5 detections per
        frame - about 23 phantom contacts every frame on a scene containing one
        real light. The adaptive threshold had done its job and calibrated to
        the clip's noise level; the problem is that at ~2.5 sigma over a
        megapixel frame, enough noise crosses the line *in connected groups of
        four* that ``min_core_pixels`` stops discriminating. Synthetic noise is
        independent per pixel and never did this; real sensor noise is spatially
        correlated by in-camera denoising and inter-frame compression, so it
        clumps into exactly the shape this filter is looking for.

        Raising the threshold is not the answer - that has been measured several
        times over to destroy the faint-target recall this path exists for. What
        works is to stop treating the candidate list as unordered. This module's
        own founding measurement is that *the target is the frame-wide maximum*
        of the residual, so when more candidates appear than a real scene can
        plausibly contain, the weakest are the ones to drop.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            self.last_candidate_count = 0
            self.last_rejected_count = 0
            return mask

        viable = [
            label
            for label in range(1, count)
            if self.min_core_pixels <= stats[label, cv2.CC_STAT_AREA] <= self.max_area
        ]

        self.last_rejected_count = 0
        if self.max_candidates > 0 and len(viable) > self.max_candidates:
            # Rank by the strongest single pixel in each component. Peak rather
            # than mean: a barely-resolved target is a few bright pixels inside
            # a dim skirt, and averaging over the skirt ranks it below a larger,
            # flatter clump of noise.
            peaks = {label: float(evidence[labels == label].max()) for label in viable}
            ranked = sorted(viable, key=lambda lab: peaks[lab], reverse=True)
            self.last_rejected_count = len(viable) - self.max_candidates
            viable = ranked[: self.max_candidates]

        out = np.zeros_like(mask)
        for label in viable:
            out[labels == label] = 255
        self.last_candidate_count = len(viable)
        return out
