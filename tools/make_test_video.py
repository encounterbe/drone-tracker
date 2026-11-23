#!/usr/bin/env python3
"""Synthesise a UAS test clip with ground truth.

Purpose: make the whole system runnable and *measurable* the moment it is
installed, without needing a real drone recording. The generated clip contains
the things that actually break naive detectors:

* a small target whose apparent size changes as it approaches and recedes,
* a bird distractor with a different motion signature (flapping, erratic),
* drifting cloud texture, so "anything that changes is a target" fails,
* a tree line at the bottom, giving real clutter and occlusion,
* sensor noise and vignetting,
* optionally a slowly panning camera, to exercise ego-motion compensation.

A JSON sidecar with per-frame ground-truth boxes is written next to the video so
detection performance can be scored rather than eyeballed
(see ``tools/evaluate.py``).

Usage::

    python tools/make_test_video.py --out output/test_drone.mp4
    python tools/make_test_video.py --out output/pan.mp4 --camera-pan 0.6 --seconds 15
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

SUPERSAMPLE = 4  # drone/bird are rendered at this scale then downsampled


# --------------------------------------------------------------------- scene
def make_sky(width: int, height: int) -> np.ndarray:
    """Vertical gradient sky, deeper blue at zenith."""
    top = np.array([150, 95, 45], dtype=np.float32)      # BGR, deep blue
    bottom = np.array([225, 200, 175], dtype=np.float32)  # pale near horizon
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    sky = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
    return np.repeat(sky, width, axis=1)


def make_cloud_texture(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Band-limited noise upscaled into soft cloud alpha in [0, 1].

    Built at several octaves so clouds have both large shapes and fine edges -
    single-octave noise looks like blur and does not challenge the detector.
    """
    acc = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    for octave in range(4):
        small_h = max(2, height // (64 >> octave))
        small_w = max(2, width // (64 >> octave))
        noise = rng.random((small_h, small_w), dtype=np.float32)
        upscaled = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        acc += amplitude * upscaled
        total += amplitude
        amplitude *= 0.5
    acc /= max(total, 1e-6)

    # Push the histogram so most of the sky is clear and clouds are distinct.
    acc = np.clip((acc - 0.52) * 2.6, 0.0, 1.0)
    return cv2.GaussianBlur(acc, (0, 0), sigmaX=height * 0.006)


def make_treeline(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Dark, irregular horizon clutter occupying the bottom of the frame."""
    mask = np.zeros((height, width), dtype=np.float32)
    base_y = int(height * 0.88)
    x = 0
    while x < width:
        tree_w = int(rng.integers(18, 70))
        tree_h = int(rng.integers(int(height * 0.03), int(height * 0.13)))
        cx = x + tree_w // 2
        cv2.ellipse(
            mask, (cx, base_y), (tree_w // 2, tree_h), 0, 0, 360, 1.0, thickness=-1
        )
        cv2.rectangle(mask, (cx - 3, base_y), (cx + 3, height), 1.0, thickness=-1)
        x += int(tree_w * rng.uniform(0.55, 0.95))
    mask[base_y + int(height * 0.05):, :] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2)


# -------------------------------------------------------------------- sprites
def render_quadcopter(size: int, roll_deg: float, rotor_phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Render a quadcopter silhouette. Returns ``(bgr_patch, alpha)``.

    Below roughly 10 px the airframe structure is not resolvable in a real
    sensor either, so the sprite degenerates to a blob - which is exactly the
    hard case the detector has to handle.
    """
    size = max(4, int(size))
    s = size * SUPERSAMPLE
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    alpha = np.zeros((s, s), dtype=np.float32)

    body_color = (48, 46, 44)
    c = s // 2

    if size < 10:
        cv2.circle(alpha, (c, c), max(1, int(s * 0.30)), 1.0, -1)
        canvas[:] = body_color
    else:
        arm_len = int(s * 0.40)
        arm_thick = max(1, int(s * 0.055))
        rotor_r = int(s * 0.20)

        for angle in (45, 135, 225, 315):
            rad = np.radians(angle + roll_deg)
            ex = int(c + arm_len * np.cos(rad))
            ey = int(c + arm_len * np.sin(rad))
            cv2.line(alpha, (c, c), (ex, ey), 1.0, arm_thick, cv2.LINE_AA)
            # Spinning rotor disc: a faint ring, brightness modulated by phase.
            disc = 0.30 + 0.18 * np.sin(rotor_phase + np.radians(angle))
            cv2.circle(alpha, (ex, ey), rotor_r, float(disc), -1, cv2.LINE_AA)
            cv2.circle(alpha, (ex, ey), rotor_r, 0.75, max(1, arm_thick // 2), cv2.LINE_AA)

        body_w = int(s * 0.22)
        body_h = int(s * 0.15)
        cv2.ellipse(alpha, (c, c), (body_w, body_h), roll_deg, 0, 360, 1.0, -1, cv2.LINE_AA)
        # Payload/gimbal bulge under the hull - a real recognition cue.
        cv2.circle(alpha, (c, c + int(s * 0.10)), int(s * 0.075), 1.0, -1, cv2.LINE_AA)
        canvas[:] = body_color

    patch = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    return patch, np.clip(a, 0.0, 1.0)


def render_bird(size: int, flap_phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Render a bird silhouette whose wings flap - a classic false positive."""
    size = max(4, int(size))
    s = size * SUPERSAMPLE
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    alpha = np.zeros((s, s), dtype=np.float32)
    canvas[:] = (38, 36, 34)

    c = s // 2
    span = int(s * 0.46)
    # Wing elevation swings with the flap cycle.
    lift = np.sin(flap_phase) * s * 0.22
    cv2.ellipse(alpha, (c, c), (int(s * 0.10), int(s * 0.055)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    for direction in (-1, 1):
        tip = (int(c + direction * span), int(c - lift))
        mid = (int(c + direction * span * 0.5), int(c - lift * 0.35 - s * 0.05))
        pts = np.array([[c, c], mid, tip], dtype=np.int32)
        cv2.polylines(alpha, [pts], False, 1.0, max(1, int(s * 0.05)), cv2.LINE_AA)

    patch = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    return patch, np.clip(a, 0.0, 1.0)


def degrade(
    patch: np.ndarray,
    alpha: np.ndarray,
    blur_sigma: float,
    haze: float,
    airlight: tuple[int, int, int] = (235, 225, 210),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply defocus blur and atmospheric haze to a sprite.

    Both effects model what actually makes a distant UAS hard to see, and they
    are different problems for a detector:

    *Defocus* spreads the silhouette over more pixels, lowering peak contrast
    and destroying the sharp edges that gradient-based methods rely on.

    *Haze* is airlight scattered into the line of sight. It washes the target
    toward the sky's own brightness, so the target-to-background difference
    collapses while the target's size stays the same. This is the harder of the
    two: a 10 px target at 8 grey levels of contrast is near the noise floor.
    """
    if blur_sigma > 0:
        k = max(3, int(blur_sigma * 4) | 1)
        patch = cv2.GaussianBlur(patch, (k, k), blur_sigma)
        alpha = cv2.GaussianBlur(alpha, (k, k), blur_sigma)

    if haze > 0:
        transmission = float(np.clip(1.0 - haze, 0.02, 1.0))
        air = np.array(airlight, dtype=np.float32)
        # Attenuate the target radiance and add airlight in its place.
        patch = np.clip(
            patch.astype(np.float32) * transmission + air * (1.0 - transmission), 0, 255
        ).astype(np.uint8)
        # Scattering also softens the silhouette's opacity.
        alpha = alpha * (0.35 + 0.65 * transmission)

    return patch, np.clip(alpha, 0.0, 1.0)


def composite(frame: np.ndarray, patch: np.ndarray, alpha: np.ndarray, cx: float, cy: float) -> None:
    """Alpha-blend a sprite into the frame at a floating-point centre, in place."""
    h, w = frame.shape[:2]
    ph, pw = patch.shape[:2]
    x0 = int(round(cx - pw / 2))
    y0 = int(round(cy - ph / 2))
    x1, y1 = x0 + pw, y0 + ph

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        return

    px0, py0 = sx0 - x0, sy0 - y0
    px1, py1 = px0 + (sx1 - sx0), py0 + (sy1 - sy0)

    a = alpha[py0:py1, px0:px1, None]
    roi = frame[sy0:sy1, sx0:sx1].astype(np.float32)
    src = patch[py0:py1, px0:px1].astype(np.float32)
    frame[sy0:sy1, sx0:sx1] = (roi * (1 - a) + src * a).astype(np.uint8)


# ------------------------------------------------------------------- motion
def drone_path(
    t: float,
    width: int,
    height: int,
    size_range: tuple[float, float] = (9.0, 55.0),
) -> tuple[float, float, float]:
    """Position and apparent size at normalised time ``t`` in [0, 1].

    A sum of incommensurate sinusoids gives an organic, non-repeating path that
    is smooth enough for a constant-velocity filter to follow but not trivially
    predictable.
    """
    x = width * (0.12 + 0.76 * (0.5 + 0.5 * np.sin(2 * np.pi * (0.55 * t - 0.25))))
    x += width * 0.05 * np.sin(2 * np.pi * 2.3 * t)

    y = height * (0.30 + 0.22 * np.sin(2 * np.pi * 0.9 * t + 1.1))
    y += height * 0.04 * np.sin(2 * np.pi * 3.7 * t)

    # Apparent size: approaches, hovers, then recedes to near the noise floor.
    distance = 0.5 + 0.5 * np.sin(2 * np.pi * 0.42 * t - 1.2)
    lo, hi = size_range
    size = lo + (hi - lo) * (distance ** 1.6)
    return float(x), float(y), float(size)


def bird_path(t: float, width: int, height: int) -> tuple[float, float, float]:
    """Distractor path: crosses the frame with erratic vertical motion."""
    x = width * (1.15 - 1.35 * ((t * 1.4) % 1.0))
    y = height * (0.42 + 0.10 * np.sin(2 * np.pi * 3.1 * t) + 0.05 * np.sin(2 * np.pi * 7.3 * t))
    size = 14.0 + 5.0 * np.sin(2 * np.pi * 0.8 * t)
    return float(x), float(y), float(size)


# --------------------------------------------------------------------- main
def generate(
    out_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    seconds: float = 20.0,
    camera_pan: float = 0.0,
    noise_sigma: float = 3.0,
    include_bird: bool = True,
    seed: int = 7,
    target_blur: float = 0.0,
    haze: float = 0.0,
    size_range: tuple[float, float] = (9.0, 55.0),
) -> dict:
    rng = np.random.default_rng(seed)
    total_frames = int(round(fps * seconds))

    # Oversized background so the virtual camera can pan across it.
    pan_margin = int(abs(camera_pan) * width * 0.5) + 4
    bg_w = width + 2 * pan_margin

    sky = make_sky(bg_w, height)
    clouds = make_cloud_texture(bg_w, height, rng)
    trees = make_treeline(bg_w, height, rng)

    cloud_color = np.array([252, 250, 248], dtype=np.float32)
    tree_color = np.array([42, 58, 40], dtype=np.float32)

    background = sky * (1 - clouds[..., None]) + cloud_color[None, None, :] * clouds[..., None]
    background = background * (1 - trees[..., None]) + tree_color[None, None, :] * trees[..., None]
    background = np.clip(background, 0, 255).astype(np.uint8)

    # Vignette, applied after compositing so it darkens targets too.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    vignette = np.clip(1.0 - 0.22 * (r ** 2), 0.0, 1.0)[..., None]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    ground_truth: list[dict] = []

    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)

        # Virtual camera pan: slide the crop window across the wide background.
        pan_offset = pan_margin + int(round(camera_pan * width * 0.5 * np.sin(2 * np.pi * 0.25 * t)))
        pan_offset = int(np.clip(pan_offset, 0, bg_w - width))
        frame = background[:, pan_offset : pan_offset + width].copy()

        objects: list[dict] = []

        # --- target ---
        dx, dy, dsize = drone_path(t, width, height, size_range)
        dx_screen = dx  # path is defined in screen space; pan moves the world
        roll = 12.0 * np.sin(2 * np.pi * 1.3 * t)
        patch, alpha = render_quadcopter(int(round(dsize)), roll, rotor_phase=idx * 1.7)
        if target_blur > 0 or haze > 0:
            patch, alpha = degrade(patch, alpha, target_blur, haze)
        composite(frame, patch, alpha, dx_screen, dy)
        half = dsize / 2.0
        objects.append(
            {
                "class": "drone",
                "bbox": [
                    round(dx_screen - half, 1),
                    round(dy - half, 1),
                    round(dx_screen + half, 1),
                    round(dy + half, 1),
                ],
                "size_px": round(dsize, 1),
            }
        )

        # --- distractor ---
        if include_bird:
            bx, by, bsize = bird_path(t, width, height)
            if -bsize < bx < width + bsize:
                bpatch, balpha = render_bird(int(round(bsize)), flap_phase=idx * 0.55)
                if target_blur > 0 or haze > 0:
                    bpatch, balpha = degrade(bpatch, balpha, target_blur, haze)
                composite(frame, bpatch, balpha, bx, by)
                bhalf = bsize / 2.0
                objects.append(
                    {
                        "class": "bird",
                        "bbox": [
                            round(bx - bhalf, 1),
                            round(by - bhalf, 1),
                            round(bx + bhalf, 1),
                            round(by + bhalf, 1),
                        ],
                        "size_px": round(bsize, 1),
                    }
                )

        frame = (frame.astype(np.float32) * vignette)
        if noise_sigma > 0:
            frame += rng.normal(0.0, noise_sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        writer.write(frame)
        ground_truth.append({"frame": idx, "objects": objects})

    writer.release()

    meta = {
        "video": os.path.basename(out_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
        "camera_pan": camera_pan,
        "seed": seed,
        "target_blur": target_blur,
        "haze": haze,
        "size_range": list(size_range),
        "frames_data": ground_truth,
    }
    gt_path = os.path.splitext(out_path)[0] + "_groundtruth.json"
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    return {"video": out_path, "ground_truth": gt_path, "frames": total_frames}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic UAS test clip.")
    ap.add_argument("--out", default="output/test_drone.mp4", help="output .mp4 path")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument(
        "--camera-pan",
        type=float,
        default=0.0,
        help="virtual camera pan amplitude (0 = static tripod, 0.5-1.0 = gimbal sweep)",
    )
    ap.add_argument("--noise", type=float, default=3.0, help="sensor noise sigma")
    ap.add_argument("--no-bird", action="store_true", help="omit the bird distractor")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--target-blur", type=float, default=0.0,
        help="defocus blur sigma applied to targets (try 1.2-2.5 for soft distant targets)",
    )
    ap.add_argument(
        "--haze", type=float, default=0.0,
        help="atmospheric haze 0-1; washes target contrast toward the sky (try 0.5-0.8)",
    )
    ap.add_argument(
        "--size-range", type=float, nargs=2, default=[9.0, 55.0], metavar=("MIN", "MAX"),
        help="apparent target size in px at closest/furthest point",
    )
    args = ap.parse_args()

    result = generate(
        out_path=args.out,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        camera_pan=args.camera_pan,
        noise_sigma=args.noise,
        include_bird=not args.no_bird,
        seed=args.seed,
        target_blur=args.target_blur,
        haze=args.haze,
        size_range=(float(args.size_range[0]), float(args.size_range[1])),
    )
    print(f"Video:        {result['video']}  ({result['frames']} frames)")
    print(f"Ground truth: {result['ground_truth']}")


if __name__ == "__main__":
    main()
