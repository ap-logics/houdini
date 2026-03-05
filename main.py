import time
import cv2
from camera import open_camera
from hands import get_hands, draw_skeleton
from gestures import GestureDetector
from draw import DrawState
from overlay import OverlayStack, ShapeOverlay
from overlay.effects.xray import XRayContent

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
                gesture.update(person, dt)
                canvas.update(gesture)
                draw_skeleton(frame, people, COLORS)

            # When the shape is closed, apply xray effect inside it
            if canvas.closed and len(canvas.vertices) >= 3:
                if xray_overlay is None:
                    xray_overlay = ShapeOverlay(XRayContent(), alpha=0.85)
                    stack.add(xray_overlay)
                xray_overlay.set_polygon(canvas.vertices)
                xray_overlay.alpha = 0.85
            elif xray_overlay is not None:
                xray_overlay.alpha = 0.0

            stack.update(dt)
            frame = stack.render(frame)

            # Draw the polygon / vertices on top of effects
            frame = canvas.render(frame)

            # HUD
            mode = "EDIT" if canvas.closed else "DRAW"
            n = len(canvas.vertices)
            hud = f"{mode} | {n} verts | pinch=place  clap=clear"
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("houdini", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
