import time
import cv2
from camera import open_camera
from hands import get_hands, draw_skeleton
from gestures import GestureDetector
from draw import DrawState
from overlay import OverlayStack, ShapeOverlay
from overlay.effects.specter import ElectricSpecterContent
from edge_effects import NeonPulseEdge

COLORS = [
    (0, 255, 128),
    (0, 128, 255),
    (255, 64, 0),
    (255, 0, 200),
]


def main():
    gesture = GestureDetector()
    canvas = DrawState()
    stack = OverlayStack()
    xray_overlay = None
    effects_on = True
    canvas.edge_effect = NeonPulseEdge()

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
                person = people[0]
                gesture.update(person, dt, now)
                canvas.update(gesture, dt)
                draw_skeleton(frame, people, COLORS)
            else:
                # Keep smoothing even with no hands visible
                canvas.update_smooth(dt)

            # When the shape is closed, apply xray effect inside it
            if canvas.closed and len(canvas.vertices) >= 3:
                if xray_overlay is None:
                    xray_overlay = ShapeOverlay(ElectricSpecterContent(), alpha=1.0)
                    stack.add(xray_overlay)
                xray_overlay.set_polygon(canvas.vertices)
                xray_overlay.alpha = 1.0 if effects_on else 0.0
            elif xray_overlay is not None:
                xray_overlay.alpha = 0.0

            stack.update(dt)
            frame = stack.render(frame)

            # Draw the polygon / vertices on top of effects
            frame = canvas.render(frame)

            # HUD
            if canvas.erasing:
                mode = "ERASE"
            elif canvas.closed:
                mode = "EDIT"
            else:
                mode = "DRAW"
            n = len(canvas.vertices)
            glow = "glow:ON" if canvas.edge_effect else "glow:off"
            hud = f"{mode} | {n} verts | pinch=place  wave=erase  dbl-click=clear  e={glow}"
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("houdini", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                effects_on = not effects_on
                canvas.edge_effect = NeonPulseEdge() if effects_on else None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
