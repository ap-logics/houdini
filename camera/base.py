from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import numpy as np

_REGISTRY: list[tuple[int, type["CameraSource"]]] = []  # (priority, cls)


class CameraSource(ABC):
    """Abstract camera source. Higher priority wins during auto-selection."""

    @staticmethod
    def register(priority: int = 0):
        """Class decorator. Higher priority = preferred during auto-select."""
        def decorator(cls: type[CameraSource]) -> type[CameraSource]:
            _REGISTRY.append((priority, cls))
            _REGISTRY.sort(key=lambda t: t[0], reverse=True)
            return cls
        return decorator

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this backend can run on the current machine."""

    @abstractmethod
    def _open(self) -> None:
        """Open the device. Called once by __init__."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return a BGR frame (H, W, 3) or None on failure."""

    @abstractmethod
    def release(self) -> None:
        """Release the camera device."""

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"


def open_camera(device_index: int = 0) -> CameraSource:
    """
    Return the highest-priority available camera backend.
    Tries each registered backend in priority order; raises RuntimeError if
    none succeed.
    """
    errors: list[str] = []
    for _priority, cls in _REGISTRY:
        if not cls.is_available():
            continue
        try:
            sig = inspect.signature(cls.__init__)
            cam = cls(device_index=device_index) if "device_index" in sig.parameters else cls()
            print(f"[camera] using {cam.name}")
            return cam
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")

    raise RuntimeError("No camera backend could be opened.\n" + "\n".join(errors))
