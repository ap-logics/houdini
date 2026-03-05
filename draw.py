"""Vertex drawing system controlled by hand gestures.

States:
- DRAWING: pinch to place vertices, builds a polygon path
- CLOSED: polygon is complete, pinch near a vertex to grab + drag it
- Clap: clear everything, back to DRAWING with empty vertices

The active hand for drawing is the RIGHT hand (index tip position).
"""

import cv2
import numpy as np

GRAB_RADIUS = 0.04  # normalised distance to grab a vertex


class DrawState:
    def __init__(self):
        self.vertices = []       # list of (x, y) normalised
        self.closed = False
        self._grabbed = None     # index of grabbed vertex, or None
        self._was_pinching = False

    def clear(self):
        self.vertices.clear()
        self.closed = False
        self._grabbed = None

    def update(self, gesture):
        """Process gesture events. `gesture` is a GestureDetector."""
        # Clap = reset
        if gesture.clap_triggered:
            self.clear()
            return

        right = gesture.right
        pinch_pos = (
            (right.index_pos[0] + right.thumb_pos[0]) / 2,
            (right.index_pos[1] + right.thumb_pos[1]) / 2,
        )

        if self.closed:
            # Edit mode: grab and drag vertices
            if right.pinch_just_started:
                self._grabbed = self._find_nearest(pinch_pos)
            elif right.pinch_just_ended:
                self._grabbed = None

            if right.pinching and self._grabbed is not None:
                self.vertices[self._grabbed] = pinch_pos
        else:
            # Drawing mode: pinch to place vertex
            if right.pinch_just_started:
                # If near the first vertex and we have 3+, close the shape
                if len(self.vertices) >= 3:
                    dx = pinch_pos[0] - self.vertices[0][0]
                    dy = pinch_pos[1] - self.vertices[0][1]
                    if (dx * dx + dy * dy) ** 0.5 < GRAB_RADIUS:
                        self.closed = True
                        return
                self.vertices.append(pinch_pos)

    def _find_nearest(self, pos):
        best_idx = None
        best_dist = GRAB_RADIUS
        for i, (vx, vy) in enumerate(self.vertices):
            d = ((pos[0] - vx) ** 2 + (pos[1] - vy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the polygon / vertices onto the frame."""
        h, w = frame.shape[:2]
        out = frame

        if not self.vertices:
            return out

        pts_px = [(int(x * w), int(y * h)) for x, y in self.vertices]

        # Draw edges
        if len(pts_px) >= 2:
            color = (0, 255, 200) if self.closed else (0, 200, 255)
            n = len(pts_px)
            limit = n if self.closed else n - 1
            for i in range(limit):
                cv2.line(out, pts_px[i], pts_px[(i + 1) % n], color, 2, cv2.LINE_AA)

        # Draw fill if closed
        if self.closed and len(pts_px) >= 3:
            overlay = out.copy()
            poly = np.array(pts_px, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 255, 200))
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

        # Draw vertices
        for i, pt in enumerate(pts_px):
            is_grabbed = self.closed and self._grabbed == i
            radius = 8 if is_grabbed else 5
            color = (255, 255, 255) if is_grabbed else (0, 220, 255)
            cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, pt, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw cursor hint at first vertex when drawing with 3+ verts
        if not self.closed and len(pts_px) >= 3:
            cv2.circle(out, pts_px[0], int(GRAB_RADIUS * w), (0, 100, 100), 1, cv2.LINE_AA)

        return out
