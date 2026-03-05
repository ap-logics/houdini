from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

# 4 corners in normalized [0,1] space, ordered: TL, TR, BR, BL
Quad = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


class BoxContent(ABC):
    """Produces pixel content for a rectangular region.

    Subclasses must implement render(). The roi parameter carries the existing
    frame pixels under the region — required for filter-style content, optional
    for generative content.
    """

    def update(self, dt: float) -> None:
        """Advance internal state by dt seconds. Default is no-op."""

    @abstractmethod
    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        """Return a BGR or BGRA image of shape (h, w, 3|4).

        Args:
            w, h: pixel dimensions of the target region's bounding box.
            roi:  BGR crop of the frame currently under this region (h×w, 3).
                  Always provided; content that ignores it (generative, video,
                  etc.) can discard it.
        """


class BoxOverlay:
    """One BoxContent composited into one or more quad regions on a frame."""

    def __init__(self, content: BoxContent, alpha: float = 1.0) -> None:
        self.content = content
        self.alpha = alpha  # global opacity, 0.0–1.0
        self._quads: list[Quad] = []

    # --- region management ---------------------------------------------------

    def add_region(self, quad: Quad) -> int:
        """Append a quad; return its index."""
        self._quads.append(quad)
        return len(self._quads) - 1

    def set_region(self, index: int, quad: Quad) -> None:
        self._quads[index] = quad

    def remove_region(self, index: int) -> None:
        self._quads.pop(index)

    def clear_regions(self) -> None:
        self._quads.clear()

    @property
    def regions(self) -> list[Quad]:
        return list(self._quads)

    # --- per-frame -----------------------------------------------------------

    def update(self, dt: float) -> None:
        self.content.update(dt)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Return a new frame with all regions composited in order."""
        out = frame
        for quad in self._quads:
            out = _composite_quad(out, self.content, quad, self.alpha)
        return out


class OverlayStack:
    """Ordered list of BoxOverlay objects applied left-to-right each frame."""

    def __init__(self) -> None:
        self._overlays: list[BoxOverlay] = []

    def add(self, overlay: BoxOverlay) -> int:
        """Append an overlay; return its index."""
        self._overlays.append(overlay)
        return len(self._overlays) - 1

    def remove(self, index: int) -> None:
        self._overlays.pop(index)

    def __getitem__(self, index: int) -> BoxOverlay:
        return self._overlays[index]

    def __len__(self) -> int:
        return len(self._overlays)

    def update(self, dt: float) -> None:
        for ov in self._overlays:
            ov.update(dt)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Chain all overlays; each sees the output of the previous."""
        out = frame
        for ov in self._overlays:
            out = ov.render(out)
        return out


class ShapeOverlay:
    """One BoxContent composited into an arbitrary closed polygon region.

    The shape is defined as a list of normalised [0,1] points (any N >= 3).
    Freehand strokes with overlapping or self-intersecting paths are handled
    correctly via OpenCV's non-zero winding fill rule.

    Call set_polygon() once when the shape is finalised, or every frame for
    live updates. The rasterised mask is cached and only rebuilt on change.
    """

    def __init__(self, content: BoxContent, alpha: float = 1.0) -> None:
        self.content = content
        self.alpha = alpha
        self._points: list[tuple[float, float]] = []
        self._simplified: np.ndarray | None = None   # (N,1,2) int32, pixel space
        self._mask_cache: np.ndarray | None = None   # float32 (fh, fw, 1)
        self._cache_frame_shape: tuple[int, int] | None = None

    def set_polygon(self, points: list[tuple[float, float]]) -> None:
        """Update the shape boundary.

        Args:
            points: closed path in normalised [0,1] coords.  The path does not
                    need to be explicitly closed (first == last); any freehand
                    stroke is accepted.  Self-intersections are filled via the
                    non-zero winding rule.
        """
        self._points = points
        self._simplified = None   # will be rebuilt at next render
        self._mask_cache = None

    def update(self, dt: float) -> None:
        self.content.update(dt)

    def render(self, frame: np.ndarray) -> np.ndarray:
        if len(self._points) < 3:
            return frame

        fh, fw = frame.shape[:2]

        if self._mask_cache is None or self._cache_frame_shape != (fh, fw):
            self._mask_cache = _build_shape_mask(self._points, fw, fh)
            self._cache_frame_shape = (fh, fw)

        return _composite_shape(frame, self.content, self._points,
                                self._mask_cache, self.alpha)


# ---------------------------------------------------------------------------
# Internal compositing helpers
# ---------------------------------------------------------------------------


def _build_shape_mask(
    points: list[tuple[float, float]],
    fw: int,
    fh: int,
) -> np.ndarray:
    """Rasterise a normalised polygon into a full-frame float32 mask (fh,fw,1)."""
    pts_px = np.array([(x * fw, y * fh) for x, y in points], dtype=np.float32)
    # Simplify freehand strokes; epsilon=2px keeps visual fidelity
    simplified = cv2.approxPolyDP(pts_px, epsilon=2.0, closed=True)
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(mask, [simplified.astype(np.int32)], 255)
    return (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]


def _composite_shape(
    frame: np.ndarray,
    content: BoxContent,
    points: list[tuple[float, float]],
    full_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    pts_px = np.array([(x * fw, y * fh) for x, y in points], dtype=np.float32)

    x0 = max(0, int(np.floor(pts_px[:, 0].min())))
    y0 = max(0, int(np.floor(pts_px[:, 1].min())))
    x1 = min(fw, int(np.ceil(pts_px[:, 0].max())))
    y1 = min(fh, int(np.ceil(pts_px[:, 1].max())))
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return frame

    roi = frame[y0:y1, x0:x1].copy()
    content_frame = content.render(bw, bh, roi)

    if content_frame.ndim == 3 and content_frame.shape[2] == 4:
        content_alpha = content_frame[:, :, 3:4].astype(np.float32) / 255.0
        content_bgr = content_frame[:, :, :3]
    else:
        content_alpha = np.ones((bh, bw, 1), dtype=np.float32)
        content_bgr = content_frame

    mask_crop = full_mask[y0:y1, x0:x1]   # (bh, bw, 1) — no copy needed
    combined = mask_crop * content_alpha * alpha

    result = frame.copy()
    result[y0:y1, x0:x1] = (
        content_bgr.astype(np.float32) * combined
        + roi.astype(np.float32) * (1.0 - combined)
    ).astype(np.uint8)
    return result

def _composite_quad(
    frame: np.ndarray,
    content: BoxContent,
    quad: Quad,
    alpha: float,
) -> np.ndarray:
    fh, fw = frame.shape[:2]

    # Normalized → pixel coords (float, for perspective math)
    pts_dst = np.array([(x * fw, y * fh) for x, y in quad], dtype=np.float32)

    # Bounding rect of the quad, clamped to the frame
    x0 = max(0, int(np.floor(pts_dst[:, 0].min())))
    y0 = max(0, int(np.floor(pts_dst[:, 1].min())))
    x1 = min(fw, int(np.ceil(pts_dst[:, 0].max())))
    y1 = min(fh, int(np.ceil(pts_dst[:, 1].max())))
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return frame

    # ROI of the current frame under the bounding rect
    roi = frame[y0:y1, x0:x1].copy()

    # Ask content to fill the bounding-box dimensions
    content_frame = content.render(bw, bh, roi)

    # Split alpha channel if present
    if content_frame.ndim == 3 and content_frame.shape[2] == 4:
        content_alpha = content_frame[:, :, 3:4].astype(np.float32) / 255.0
        content_bgr = content_frame[:, :, :3]
    else:
        content_alpha = np.ones((bh, bw, 1), dtype=np.float32)
        content_bgr = content_frame

    # Perspective warp: content rect (TL,TR,BR,BL) → quad in bbox-local coords
    pts_src = np.array(
        [(0, 0), (bw, 0), (bw, bh), (0, bh)], dtype=np.float32
    )
    pts_dst_local = pts_dst - np.array([x0, y0], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst_local)

    warped_bgr = cv2.warpPerspective(content_bgr, M, (bw, bh))
    warped_alpha = cv2.warpPerspective(
        content_alpha, M, (bw, bh), flags=cv2.INTER_LINEAR
    )
    if warped_alpha.ndim == 2:
        warped_alpha = warped_alpha[:, :, np.newaxis]

    # Polygon mask for the quad (clips warped content to exact quad shape)
    poly = pts_dst_local.astype(np.int32)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Final alpha = polygon mask × content alpha × global opacity
    combined = mask_f * warped_alpha * alpha

    result = frame.copy()
    result[y0:y1, x0:x1] = (
        warped_bgr.astype(np.float32) * combined
        + roi.astype(np.float32) * (1.0 - combined)
    ).astype(np.uint8)
    return result
