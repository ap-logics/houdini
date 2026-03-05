from __future__ import annotations

from collections.abc import Callable

import numpy as np

from overlay.base import BoxContent


class FilterContent(BoxContent):
    """Applies an arbitrary numpy transform to a source content's output.

    Stub — not yet implemented.

    The filter_fn receives the rendered source frame and returns a transformed
    frame of the same shape. Use roi-aware sources for passthrough effects
    (blur, colour grade, pixelate) and generative sources for overlay effects.
    """

    def __init__(
        self,
        source: BoxContent,
        filter_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.source = source
        self.filter_fn = filter_fn

    def update(self, dt: float) -> None:
        self.source.update(dt)

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError
