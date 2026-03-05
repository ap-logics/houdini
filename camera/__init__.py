"""
camera — plug-in camera source system

Usage:
    from camera import open_camera, CameraSource

    cam = open_camera()   # auto-selects best available backend
    frame = cam.read()    # np.ndarray (H, W, 3) BGR, or None on failure
    cam.release()

To add a custom backend:
    from camera import CameraSource

    @CameraSource.register(priority=50)
    class MyCamera(CameraSource):
        @classmethod
        def is_available(cls) -> bool: ...
        def _open(self) -> None: ...
        def read(self) -> np.ndarray | None: ...
        def release(self) -> None: ...
"""

from .base import CameraSource, open_camera

# Import backends so their @register decorators fire on package import.
from . import opencv, avfoundation  # noqa: F401

__all__ = ["CameraSource", "open_camera"]
