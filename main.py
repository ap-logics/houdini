import math
import time
import cv2
from camera import open_camera
from hands import get_hands, draw_skeleton
from overlay import OverlayStack, BoxOverlay, ShapeOverlay
from overlay.effects import SolidContent
from overlay.effects.xray import XRayContent

COLORS = [
    (0, 255, 128),
    (0, 128, 255),
    (255, 64, 0),
    (255, 0, 200),
]


def _sync_overlays(
    people: list | None,
    overlays: list[BoxOverlay],
    stack: OverlayStack,
) -> None:
    """Create or update one BoxOverlay per detected person; hide the rest."""
    if people:
        for i, person in enumerate(people):
            quad = tuple(tuple(p) for p in person["box"])
            if i >= len(overlays):
                ov = BoxOverlay(SolidContent(COLORS[i % len(COLORS)]), alpha=0.5)
                ov.add_region(quad)
                stack.add(ov)
                overlays.append(ov)
            else:
                overlays[i].alpha = 0.5
                overlays[i].set_region(0, quad)
        for i in range(len(people), len(overlays)):
            overlays[i].alpha = 0.0
    else:
        for ov in overlays:
            ov.alpha = 0.0


def _star_polygon(
    cx: float, cy: float, outer_r: float, inner_r: float, points: int = 5
) -> list[tuple[float, float]]:
    """Return a normalised star polygon (alternating outer/inner vertices)."""
    verts = []
    for i in range(points):
        for r, angle_offset in ((outer_r, 0), (inner_r, math.pi / points)):
            angle = -math.pi / 2 + i * 2 * math.pi / points + angle_offset
            verts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return verts


def main():
    stack = OverlayStack()
    overlays: list[BoxOverlay] = []

    # ShapeOverlay example — a star in the top-left corner
    star = ShapeOverlay(SolidContent((0, 200, 255)), alpha=0.75)
    star.set_polygon(_star_polygon(cx=0.15, cy=0.15, outer_r=0.10, inner_r=0.04))
    stack.add(star)

    with open_camera() as cam:
        prev = time.time()
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt, prev = now - prev, now

            people = get_hands(frame)
            if people:
                for i, person in enumerate(people):
                    quad = tuple(tuple(p) for p in person["box"])
                    if i >= len(overlays):
                        ov = BoxOverlay(XRayContent(), alpha=0.9)
                        ov.add_region(quad)
                        stack.add(ov)
                        overlays.append(ov)
                    else:
                        overlays[i].alpha = 0.9
                        overlays[i].set_region(0, quad)
                for i in range(len(people), len(overlays)):
                    overlays[i].alpha = 0.0
                draw_skeleton(frame, people, COLORS)
            else:
                for ov in overlays:
                    ov.alpha = 0.0

            stack.update(dt)
            frame = stack.render(frame)
            cv2.imshow("houdini", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
