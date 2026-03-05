"""Vertex drawing system controlled by hand gestures.

States:
- DRAWING: pinch to place vertices, builds a polygon path
- CLOSED: pinch near a vertex to grab + drag it
- ERASING: wave 3x to toggle — pinch near a vertex to delete it
- Double click (two quick pinches): clear everything, back to DRAWING

Either hand can pinch to interact.

Display vertices are smoothly interpolated toward logical vertices each frame.
"""

import cv2
import numpy as np

GRAB_RADIUS = 0.04
LERP_SPEED = 12.0  # higher = snappier, lower = smoother


def _lerp(a, b, t):
    return a + (b - a) * t


class DrawState:
    def __init__(self):
        self.vertices = []        # logical positions (targets)
        self._display = []        # smoothed positions for rendering
        self.closed = False
        self.erasing = False
        self._grabbed = None

    def clear(self):
        self.vertices.clear()
        self._display.clear()
        self.closed = False
        self.erasing = False
        self._grabbed = None

    def update(self, gesture, dt: float = 0.033):
        if gesture.double_click:
            self.clear()
            return

        if gesture.wave_triggered:
            self.erasing = not self.erasing
            self._grabbed = None

        pinch_pos = gesture.pinch_pos

        if self.erasing:
            if gesture.any_pinch_started and self.vertices:
                idx = self._find_nearest_vertex(pinch_pos)
                if idx is not None:
                    self.vertices.pop(idx)
                    self._display.pop(idx)
                    if len(self.vertices) < 3:
                        self.closed = False
                    if not self.vertices:
                        self.erasing = False
        elif self.closed:
            if gesture.any_pinch_started:
                self._grabbed = self._find_nearest_vertex(pinch_pos)
            elif gesture.any_pinch_ended:
                self._grabbed = None

            if gesture.any_pinching and self._grabbed is not None:
                self.vertices[self._grabbed] = pinch_pos
        else:
            if gesture.any_pinch_started:
                if len(self.vertices) >= 3:
                    dx = pinch_pos[0] - self.vertices[0][0]
                    dy = pinch_pos[1] - self.vertices[0][1]
                    if (dx * dx + dy * dy) ** 0.5 < GRAB_RADIUS:
                        self.closed = True
                        return
                self.vertices.append(pinch_pos)
                self._display.append(pinch_pos)

        # Smooth display positions toward logical targets
        t = min(1.0, LERP_SPEED * dt)
        for i in range(min(len(self._display), len(self.vertices))):
            dx = _lerp(self._display[i][0], self.vertices[i][0], t)
            dy = _lerp(self._display[i][1], self.vertices[i][1], t)
            self._display[i] = (dx, dy)

        # Sync list lengths (safety)
        while len(self._display) < len(self.vertices):
            self._display.append(self.vertices[len(self._display)])
        while len(self._display) > len(self.vertices):
            self._display.pop()

    def update_smooth(self, dt: float = 0.033):
        """Run only the smoothing step (no gesture input)."""
        t = min(1.0, LERP_SPEED * dt)
        for i in range(min(len(self._display), len(self.vertices))):
            dx = _lerp(self._display[i][0], self.vertices[i][0], t)
            dy = _lerp(self._display[i][1], self.vertices[i][1], t)
            self._display[i] = (dx, dy)

    def _find_nearest_vertex(self, pos):
        best_idx = None
        best_dist = GRAB_RADIUS
        for i, (vx, vy) in enumerate(self.vertices):
            d = ((pos[0] - vx) ** 2 + (pos[1] - vy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def render(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame

        if not self._display:
            return out

        # Use smoothed display positions for all rendering
        pts_px = [(int(x * w), int(y * h)) for x, y in self._display]

        # Draw edges
        if len(pts_px) >= 2:
            if self.erasing:
                edge_color = (0, 0, 255)
            elif self.closed:
                edge_color = (0, 255, 200)
            else:
                edge_color = (0, 200, 255)
            n = len(pts_px)
            limit = n if self.closed else n - 1
            for i in range(limit):
                cv2.line(out, pts_px[i], pts_px[(i + 1) % n], edge_color, 2, cv2.LINE_AA)

        # Draw fill if closed
        if self.closed and len(pts_px) >= 3 and not self.erasing:
            overlay = out.copy()
            poly = np.array(pts_px, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 255, 200))
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

        # Draw vertices
        for i, pt in enumerate(pts_px):
            is_grabbed = self.closed and self._grabbed == i
            if self.erasing:
                radius = 7
                color = (0, 0, 255)
                outline = (255, 255, 255)
            elif is_grabbed:
                radius = 8
                color = (255, 255, 255)
                outline = (0, 0, 0)
            else:
                radius = 5
                color = (0, 220, 255)
                outline = (0, 0, 0)
            cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, pt, radius + 2, outline, 1, cv2.LINE_AA)

        # Snap hint when drawing
        if not self.closed and not self.erasing and len(pts_px) >= 3:
            cv2.circle(out, pts_px[0], int(GRAB_RADIUS * w), (0, 100, 100), 1, cv2.LINE_AA)

        return out
