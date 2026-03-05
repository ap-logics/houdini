import time
import cv2
from camera import open_camera
from hands import get_hands
from overlay import OverlayStack, BoxOverlay
from overlay.effects.obj_content import ObjContent

OBJ_PATH = "objects/gun.obj"

def main():
    stack = OverlayStack()
    overlays: list[BoxOverlay] = []

    with open_camera() as cam:
        prev = time.time()
        while True:
            frame = cam.read()
            if frame is None:
                continue

            now = time.time()
            dt, prev = now - prev, now

            people = get_hands(frame)

            if people:
                for i, person in enumerate(people):
                    quad = tuple(tuple(p) for p in person["box"])
                    if i >= len(overlays):
                        ov = BoxOverlay(ObjContent(OBJ_PATH), alpha=0.9)
                        ov.add_region(quad)
                        stack.add(ov)
                        overlays.append(ov)
                    else:
                        overlays[i].alpha = 0.9
                        overlays[i].set_region(0, quad)
                for i in range(len(people), len(overlays)):
                    overlays[i].alpha = 0.0
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
