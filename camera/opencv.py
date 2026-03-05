from __future__ import annotations

import cv2
import numpy as np

from .base import CameraSource


@CameraSource.register(priority=0)
class OpenCVCamera(CameraSource):
    """Vanilla OpenCV VideoCapture — works everywhere OpenCV does."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None
        self._open()

    @classmethod
    def is_available(cls) -> bool:
        return True  # always available if OpenCV is installed

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCVCamera: could not open device {self._device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
