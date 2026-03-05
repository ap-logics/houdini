from __future__ import annotations

import platform
import re
import subprocess

import cv2
import numpy as np

from .base import CameraSource


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _avfoundation_device_index(preferred_name: str = "FaceTime") -> int:
    """
    Query ffmpeg for AVFoundation video devices and return the index of the
    first device whose name contains *preferred_name*. Falls back to 0.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=3,
        )
        video_section = False
        for line in result.stderr.splitlines():
            if "AVFoundation video devices" in line:
                video_section = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if video_section:
                m = re.search(r"\[(\d+)\]\s+(.+)", line)
                if m and preferred_name.lower() in m.group(2).lower():
                    return int(m.group(1))
    except Exception:
        pass
    return 0


@CameraSource.register(priority=10)
class AVFoundationCamera(CameraSource):
    """
    Apple Silicon-optimised camera via OpenCV's AVFoundation backend.

    Over the generic backend this:
      - Explicitly requests CAP_AVFOUNDATION, skipping V4L2/GStreamer probing.
      - Requests BGRA from the driver to avoid a YUV→BGR conversion (~2 ms/frame on M1).
      - Reduces the OS buffer to 1 frame to minimise hand-tracking latency.
      - Auto-selects the built-in FaceTime HD / Continuity Camera by name.
    """

    def __init__(self, device_index: int | None = None) -> None:
        self._device_index = (
            device_index if device_index is not None
            else _avfoundation_device_index("FaceTime")
        )
        self._cap: cv2.VideoCapture | None = None
        self._open()

    @classmethod
    def is_available(cls) -> bool:
        return _is_apple_silicon()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"AVFoundationCamera: could not open device {self._device_index}"
            )
        cap = self._cap
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # BGRA: driver skips YUV→BGR; we strip alpha ourselves below
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"BGRA"))
        # Single-frame buffer minimises latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
