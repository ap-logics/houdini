from __future__ import annotations

import numpy as np
import cv2

from overlay.base import BoxContent


def _load_obj(path: str):
    """Parse OBJ, return (verts Nx3, normals Nx3, tri_vis list of (vi0,vi1,vi2), tri_nis)."""
    verts = []
    normals = []
    tri_vis = []
    tri_nis = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_v = []
                face_n = []
                for tok in parts[1:]:
                    idx = tok.split('/')
                    face_v.append(int(idx[0]) - 1)
                    face_n.append(int(idx[2]) - 1 if len(idx) >= 3 and idx[2] else 0)
                # Triangulate (fan)
                for j in range(1, len(face_v) - 1):
                    tri_vis.append((face_v[0], face_v[j], face_v[j + 1]))
                    tri_nis.append((face_n[0], face_n[j], face_n[j + 1]))

    verts = np.array(verts, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32) if normals else np.zeros_like(verts)

    # Center and normalize to [-1, 1]
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= center
    scale = np.abs(verts).max()
    if scale > 0:
        verts /= scale

    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms

    return verts, normals, np.array(tri_vis, dtype=np.int32), np.array(tri_nis, dtype=np.int32)


def _rotation_y(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rotation_x(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


class ObjContent(BoxContent):
    """Renders a rotating OBJ model using CPU rasterization (no OpenGL needed)."""

    def __init__(self, obj_path: str, color: tuple[int, int, int] = (180, 180, 200)):
        """color is BGR 0-255."""
        self._obj_path = obj_path
        self._color = np.array(color, dtype=np.float32)
        self._angle = 0.0
        self._verts = None
        self._normals = None
        self._tri_vis = None
        self._tri_nis = None
        self._loaded = False

    def _load(self):
        self._verts, self._normals, self._tri_vis, self._tri_nis = _load_obj(self._obj_path)
        self._loaded = True

    def update(self, dt: float) -> None:
        self._angle += 45.0 * dt

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        if not self._loaded:
            self._load()

        # Rotate vertices
        rot = _rotation_y(self._angle) @ _rotation_x(-15)
        rv = self._verts @ rot.T
        rn = self._normals @ rot.T

        # Auto-fit: find bounding box of rotated verts and scale to fill frame
        margin = 0.1  # 10% padding
        x_range = rv[:, 0].max() - rv[:, 0].min()
        y_range = rv[:, 1].max() - rv[:, 1].min()
        x_center = (rv[:, 0].max() + rv[:, 0].min()) / 2
        y_center = (rv[:, 1].max() + rv[:, 1].min()) / 2
        fit_scale = (1 - margin * 2) / max(x_range, y_range, 1e-6)

        # Orthographic projection (no perspective distortion, fills the box)
        proj_x = (rv[:, 0] - x_center) * fit_scale
        proj_y = (rv[:, 1] - y_center) * fit_scale

        # Map to pixel coords (centered)
        sx = ((proj_x + 0.5) * w).astype(np.float32)
        sy = ((0.5 - proj_y) * h).astype(np.float32)

        # Per-face: average Z for depth sorting, face normal for lighting
        v0 = rv[self._tri_vis[:, 0]]
        v1 = rv[self._tri_vis[:, 1]]
        v2 = rv[self._tri_vis[:, 2]]
        face_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3

        # Average vertex normals per face for lighting
        n0 = rn[self._tri_nis[:, 0]]
        n1 = rn[self._tri_nis[:, 1]]
        n2 = rn[self._tri_nis[:, 2]]
        face_normals = (n0 + n1 + n2) / 3
        fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        fn_len[fn_len == 0] = 1
        face_normals /= fn_len

        # Two-sided directional light from front
        light_dir = np.array([0.2, 0.3, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.abs(face_normals @ (-light_dir))  # two-sided
        brightness = 0.35 + 0.65 * diffuse

        # Depth sort (painter's algorithm — back to front)
        order = np.argsort(-face_z)

        # Render
        img = np.zeros((h, w, 4), dtype=np.uint8)

        for idx in order:
            i0, i1, i2 = self._tri_vis[idx]
            pts = np.array([
                [sx[i0], sy[i0]],
                [sx[i1], sy[i1]],
                [sx[i2], sy[i2]],
            ], dtype=np.int32)

            b = brightness[idx]
            color_bgr = np.clip(self._color * b, 0, 255).astype(np.uint8)
            cv2.fillPoly(img, [pts], (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]), 255))

        return img
