"""Edge effects for closed polygon outlines.

Usage:
    canvas.edge_effect = NeonPulseEdge()   # enable
    canvas.edge_effect = None              # disable

Contract:
    class EdgeEffect(ABC):
        def render(self, frame, pts_px, t) -> np.ndarray: ...
"""

from __future__ import annotations

import abc

import cv2
import numpy as np


class EdgeEffect(abc.ABC):
    """Base class for polygon edge renderers."""

    @abc.abstractmethod
    def render(
        self,
        frame: np.ndarray,
        pts_px: list[tuple[int, int]],
        t: float,
    ) -> np.ndarray:
        """Draw effect on top of frame; returns modified frame."""


class NeonPulseEdge(EdgeEffect):
    """Layered neon glow with a slow breathing pulse.

    Three additive layers:
      - Wide soft halo  (large Gaussian)
      - Tight core glow (small Gaussian)
      - Sharp 1-px white spine redrawn on top
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (60, 255, 140),  # BGR: spring-green neon
        pulse_speed: float = 1.8,
        glow_radius: int = 21,
    ) -> None:
        self._color = np.array(color, dtype=np.float32)
        self._pulse_speed = pulse_speed
        self._glow_radius = glow_radius

    def render(
        self,
        frame: np.ndarray,
        pts_px: list[tuple[int, int]],
        t: float,
    ) -> np.ndarray:
        if len(pts_px) < 2:
            return frame

        h, w = frame.shape[:2]
        n = len(pts_px)

        # Breathing envelope: 0.55–1.0
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(t * self._pulse_speed * np.pi))

        # ── Draw a fat line on a scratch canvas for the bloom source ──────
        bloom_src = np.zeros((h, w), dtype=np.uint8)
        for i in range(n):
            cv2.line(bloom_src, pts_px[i], pts_px[(i + 1) % n], 255, 5, cv2.LINE_AA)

        # ── Wide halo ─────────────────────────────────────────────────────
        r_wide = self._glow_radius | 1  # must be odd
        halo = cv2.GaussianBlur(bloom_src, (r_wide, r_wide), 0).astype(np.float32) / 255.0

        # ── Tight core glow ───────────────────────────────────────────────
        r_tight = max(5, r_wide // 4) | 1
        core = cv2.GaussianBlur(bloom_src, (r_tight, r_tight), 0).astype(np.float32) / 255.0

        # Combine: halo at 70% + tight core at 100%, modulated by pulse
        combined = np.clip(halo * 0.7 + core, 0.0, 1.0) * pulse

        # Colour and convert
        glow_bgr = (combined[:, :, np.newaxis] * self._color).astype(np.uint8)

        # ── Additive blend onto frame ──────────────────────────────────────
        out = cv2.add(frame, glow_bgr)

        # ── Sharp spine on top for definition ─────────────────────────────
        spine_bright = int(200 + 55 * pulse)
        spine_color = (spine_bright, spine_bright, spine_bright)
        for i in range(n):
            cv2.line(out, pts_px[i], pts_px[(i + 1) % n], spine_color, 1, cv2.LINE_AA)

        return out
