"""HUD colour themes and low-level drawing primitives.

Colours are BGR tuples (OpenCV convention). Themes exist because the right
palette depends on the sensor and the light: amber reads well over daylight sky,
green over low-light and IR, white over thermal. Swapping ``ui.theme`` must
never require touching drawing code, so every draw call takes colours from the
active theme rather than hard-coding them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Theme", "THEMES", "get_theme", "draw_text", "draw_panel", "alpha_rect"]

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """A complete HUD colour scheme."""

    name: str
    primary: BGR        # frames, reticle, general chrome
    secondary: BGR      # labels, less important chrome
    text: BGR           # body text
    dim: BGR            # de-emphasised text, grid lines
    accent: BGR         # highlights, selected values
    lock: BGR           # confirmed target lock
    warn: BGR           # coasting / degraded
    alert: BGR          # errors, loss of track
    track: BGR          # non-primary confirmed tracks
    tentative: BGR      # unconfirmed tracks
    panel_bg: BGR       # panel fill (alpha-blended)
    panel_alpha: float  # 0 = invisible, 1 = opaque


THEMES: dict[str, Theme] = {
    # Default. Pure greyscale: state is carried by *brightness* rather than hue,
    # so a locked target reads as the brightest thing on screen. The HUD keeps
    # its state cues regardless - coasting tracks are drawn dashed and labelled
    # COAST, and the status marker blinks - so no information is lost by
    # dropping colour.
    "white": Theme(
        name="white",
        primary=(240, 240, 240),
        secondary=(185, 185, 185),
        text=(255, 255, 255),
        dim=(130, 130, 130),
        accent=(255, 255, 255),
        lock=(255, 255, 255),
        warn=(190, 190, 190),
        alert=(225, 225, 225),
        track=(175, 175, 175),
        tentative=(110, 110, 110),
        panel_bg=(14, 14, 14),
        panel_alpha=0.66,
    ),
    "tactical_amber": Theme(
        name="tactical_amber",
        primary=(60, 190, 255),
        secondary=(90, 165, 210),
        text=(235, 240, 245),
        dim=(120, 130, 140),
        accent=(80, 220, 255),
        lock=(90, 255, 140),
        warn=(60, 190, 255),
        alert=(70, 70, 250),
        track=(200, 200, 120),
        tentative=(130, 130, 130),
        panel_bg=(18, 16, 14),
        panel_alpha=0.62,
    ),
    "nightvision": Theme(
        name="nightvision",
        primary=(110, 255, 130),
        secondary=(90, 200, 110),
        text=(200, 255, 210),
        dim=(70, 130, 80),
        accent=(160, 255, 190),
        lock=(120, 255, 160),
        warn=(60, 220, 255),
        alert=(80, 80, 255),
        track=(90, 200, 120),
        tentative=(60, 110, 70),
        panel_bg=(8, 20, 10),
        panel_alpha=0.60,
    ),
    "arctic": Theme(
        name="arctic",
        primary=(255, 210, 140),
        secondary=(230, 180, 120),
        text=(250, 245, 240),
        dim=(150, 140, 130),
        accent=(255, 230, 180),
        lock=(150, 255, 200),
        warn=(90, 200, 255),
        alert=(90, 90, 255),
        track=(220, 190, 150),
        tentative=(140, 130, 120),
        panel_bg=(26, 20, 14),
        panel_alpha=0.58,
    ),
    "ir_white": Theme(
        name="ir_white",
        primary=(240, 240, 240),
        secondary=(200, 200, 200),
        text=(255, 255, 255),
        dim=(140, 140, 140),
        accent=(255, 255, 255),
        lock=(120, 255, 255),
        warn=(80, 200, 255),
        alert=(80, 80, 255),
        track=(190, 190, 190),
        tentative=(120, 120, 120),
        panel_bg=(12, 12, 12),
        panel_alpha=0.65,
    ),
}


def get_theme(name: str) -> Theme:
    """Look up a theme, falling back to the default rather than raising."""
    return THEMES.get(str(name).lower(), THEMES["white"])


# ------------------------------------------------------------------ drawing
def alpha_rect(
    img: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: BGR,
    alpha: float,
) -> None:
    """Blend a filled rectangle into ``img`` in place.

    Blends only the affected ROI rather than compositing a full-frame overlay:
    at 1080p that is the difference between a negligible cost and several
    milliseconds per panel, every frame.
    """
    h, w = img.shape[:2]
    x1 = max(0, min(w, int(top_left[0])))
    y1 = max(0, min(h, int(top_left[1])))
    x2 = max(0, min(w, int(bottom_right[0])))
    y2 = max(0, min(h, int(bottom_right[1])))
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = np.array(color, dtype=np.uint8)
    cv2.addWeighted(fill, float(alpha), roi, 1.0 - float(alpha), 0.0, dst=roi)


def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: BGR,
    scale: float = 0.42,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    shadow: bool = True,
) -> int:
    """Draw text with a drop shadow. Returns the rendered pixel width.

    The shadow is not decoration: HUD text sits over unknown imagery, and
    without it light-on-light frames render the readout illegible.
    """
    x, y = int(origin[0]), int(origin[1])
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    return int(tw)


def draw_panel(
    img: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    theme: Theme,
    title: str | None = None,
    corner_len: int = 10,
) -> tuple[int, int]:
    """Draw a bordered, semi-transparent panel with corner ticks.

    Returns the ``(x, y)`` origin for the first line of body content.
    """
    x, y = int(top_left[0]), int(top_left[1])
    w, h = int(size[0]), int(size[1])

    alpha_rect(img, (x, y), (x + w, y + h), theme.panel_bg, theme.panel_alpha)

    # Corner ticks rather than a full border: lighter on the eye, and it keeps
    # more of the underlying image visible.
    for cx, cy, dx, dy in (
        (x, y, 1, 1),
        (x + w, y, -1, 1),
        (x, y + h, 1, -1),
        (x + w, y + h, -1, -1),
    ):
        cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), theme.primary, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), theme.primary, 1, cv2.LINE_AA)

    content_y = y + 16
    if title:
        draw_text(img, title, (x + 10, y + 16), theme.primary, scale=0.40, thickness=1)
        cv2.line(img, (x + 8, y + 23), (x + w - 8, y + 23), theme.secondary, 1, cv2.LINE_AA)
        content_y = y + 40
    return x + 10, content_y


def draw_bar(
    img: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    value: float,
    theme: Theme,
    color: BGR | None = None,
    segments: int = 20,
) -> None:
    """Segmented horizontal bar meter for a normalised value in [0, 1]."""
    x, y = int(top_left[0]), int(top_left[1])
    w, h = int(size[0]), int(size[1])
    value = float(np.clip(value, 0.0, 1.0))
    color = color or theme.accent

    cv2.rectangle(img, (x, y), (x + w, y + h), theme.dim, 1, cv2.LINE_AA)
    gap = 1
    seg_w = max(1, (w - 2 - (segments - 1) * gap) // segments)
    filled = int(round(value * segments))
    for i in range(filled):
        sx = x + 1 + i * (seg_w + gap)
        if sx + seg_w > x + w - 1:
            break
        cv2.rectangle(img, (sx, y + 1), (sx + seg_w, y + h - 1), color, -1)
