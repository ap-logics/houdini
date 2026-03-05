from __future__ import annotations

import numpy as np

from overlay.base import BoxContent


class SolidContent(BoxContent):
    """Fills the region with a flat BGR colour."""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        # color is (B, G, R), values 0–255
        self.color = color

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        return np.full((h, w, 3), self.color, dtype=np.uint8)
