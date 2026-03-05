from __future__ import annotations

import numpy as np

from overlay.base import BoxContent


class GlitchContent(BoxContent):
    """Digital scan-line / channel-shift glitch applied over the region.

    Stub — not yet implemented.
    """

    def __init__(self, source: BoxContent) -> None:
        self.source = source

    def update(self, dt: float) -> None:
        self.source.update(dt)

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError
