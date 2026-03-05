from __future__ import annotations

import numpy as np

from overlay.base import BoxContent


class VideoContent(BoxContent):
    """Plays an mp4 (or any cv2-readable video) looped into the region.

    Stub — not yet implemented.
    """

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError
