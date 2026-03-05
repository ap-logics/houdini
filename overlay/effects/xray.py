from __future__ import annotations

import cv2
import numpy as np

from overlay.base import BoxContent

# Neon cyan-blue color for edges (BGR)
_EDGE_COLOR = np.array([220, 180, 20], dtype=np.float32)  # cyan-blue glow


class XRayContent(BoxContent):
    """X-ray / skeleton filter on live camera pixels.

    Runs Canny edge detection on the ROI, colorizes edges in neon cyan on a
    deep-space dark background, and adds a scanline overlay for sci-fi feel.
    """

    def __init__(
        self,
        edge_low: int = 40,
        edge_high: int = 120,
        glow_radius: int = 3,
        scanline_alpha: float = 0.15,
    ) -> None:
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.glow_radius = glow_radius
        self.scanline_alpha = scanline_alpha
        self._time = 0.0

    def update(self, dt: float) -> None:
        self._time += dt

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        if roi is None or roi.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        src = roi if roi.shape[:2] == (h, w) else cv2.resize(roi, (w, h))

        # --- contrast enhance so faint detail shows up ---
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(
            lab[:, :, 0]
        )
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # --- Canny edges ---
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)

        # --- glow: blur edges and add back sharp edges on top ---
        k = self.glow_radius * 2 + 1
        glow = cv2.GaussianBlur(edges, (k, k), 0)
        glow = cv2.addWeighted(glow, 0.6, edges, 0.4, 0)

        edge_f = glow.astype(np.float32) / 255.0

        # --- colorize: neon cyan-blue on deep-space background ---
        # Space bg: very dark blue-purple
        bg = np.array([15, 5, 5], dtype=np.float32)  # BGR dark
        out_f = bg[np.newaxis, np.newaxis, :] + edge_f[:, :, np.newaxis] * _EDGE_COLOR
        out = np.clip(out_f, 0, 255).astype(np.uint8)

        # --- scanlines ---
        if self.scanline_alpha > 0:
            scan = np.ones((h, w), dtype=np.float32)
            scan[::2, :] = 1.0 - self.scanline_alpha
            out = (out.astype(np.float32) * scan[:, :, np.newaxis]).astype(np.uint8)

        return out
