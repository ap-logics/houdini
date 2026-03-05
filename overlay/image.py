from __future__ import annotations

import numpy as np

from overlay.base import BoxContent


class ImageContent(BoxContent):
    """Renders a static image (PNG/JPEG, BGRA if alpha is present).

    Stub — not yet implemented.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError
