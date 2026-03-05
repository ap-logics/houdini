import cv2
import numpy as np
from camera import open_camera
from hands import get_hands, HAND_CONNECTIONS

def main():
    with open_camera() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            people = get_hands(frame)

            COLORS = [
                (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            ]
            if people:
                for idx, person in enumerate(people):
                    color = COLORS[idx % len(COLORS)]
                    box_color = (0, 0, 255)

                    for landmarks in person["hands"]:
                        for a, b in HAND_CONNECTIONS:
                            pt1 = (int(landmarks[a][0] * w), int(landmarks[a][1] * h))
                            pt2 = (int(landmarks[b][0] * w), int(landmarks[b][1] * h))
                            cv2.line(frame, pt1, pt2, color, 2)
                        for lx, ly in landmarks:
                            cv2.circle(frame, (int(lx * w), int(ly * h)), 3, color, -1)

                    box_px = np.array(
                        [(int(x * w), int(y * h)) for x, y in person["box"]],
                        dtype=np.int32,
                    )
                    cv2.polylines(frame, [box_px], isClosed=True, color=box_color, thickness=2)
                    for pt in box_px:
                        cv2.circle(frame, tuple(pt), 6, box_color, -1)

            cv2.imshow("houdini", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
