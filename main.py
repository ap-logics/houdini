import cv2
from camera import open_camera

def main():
    with open_camera() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            cv2.imshow("houdini", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
