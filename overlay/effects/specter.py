from __future__ import annotations

import cv2
import numpy as np

from overlay.base import BoxContent

# Colour palette (BGR)
_BLOOM_COLOR = np.array([220, 180, 20], dtype=np.float32)    # neon cyan-blue
_SPARK_COLOR = np.array([20, 160, 255], dtype=np.float32)    # amber-gold
_DISCHARGE_COLOR = np.array([220, 230, 255], dtype=np.float32)  # near-white

# Particle states
_DRIFTING = 0
_LOCKED_ON = 1

# Physics (all speeds in px/s)
_DRIFT_SPEED = 45.0        # px/s random walk
_LOCK_SPEED = 250.0        # px/s homing speed
_SNAP_THRESHOLD = 30.0     # px — distance at which drifter snaps to edge
_ARRIVAL_THRESHOLD = 2.0   # px — distance considered "arrived"
_SPONT_PROB_PS = 0.2       # spontaneous capture probability per second
_ARC_THRESHOLD = 60.0      # px — max inter-particle distance for arcs
_MAX_ARCS = 25             # cap arc count to keep rendering fast


class ElectricSpecterContent(BoxContent):
    """Ghost-body silhouette bloom + attracted amber spark swarm."""

    def __init__(self, n_particles: int = 150) -> None:
        self._n = n_particles
        self._dt = 1.0 / 30.0

        # Voltage: continuous brownian drift + occasional spike (0–1)
        self._voltage = 0.5
        self._voltage_vel = 0.0
        self._spike_cd = 0.0

        # Cluster burst timer
        self._burst_timer = 0.0
        self._next_burst = float(np.random.uniform(0.3, 1.2))

        # Particle arrays — lazy-initialised on first render
        self._w = 0
        self._h = 0
        self._px: np.ndarray | None = None   # float32 (N,)
        self._py: np.ndarray | None = None
        self._state: np.ndarray | None = None  # int8 (N,)
        self._tx: np.ndarray | None = None   # float32 (N,) — LOCKED_ON target
        self._ty: np.ndarray | None = None
        self._is_ring: np.ndarray | None = None  # bool (N,) — 20% hollow rings

        # Discharge contact points accumulated during _step, drawn in render
        self._discharge_pts: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # BoxContent interface
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        self._dt = dt

        # Brownian voltage drift
        self._voltage_vel += float(np.random.uniform(-1.0, 1.0)) * 8.0 * dt
        self._voltage_vel *= max(0.0, 1.0 - 6.0 * dt)  # damping
        self._voltage += self._voltage_vel * dt
        self._voltage = float(np.clip(self._voltage, 0.15, 1.0))

        # Random voltage spike → fast drop-off
        self._spike_cd -= dt
        if self._spike_cd <= 0 and np.random.rand() < 0.04:
            self._voltage = float(np.random.uniform(0.75, 1.0))
            self._voltage_vel = -5.0
            self._spike_cd = float(np.random.uniform(0.4, 1.2))

        self._burst_timer += dt

    def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray:
        if roi is None or roi.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        src = roi if roi.shape[:2] == (h, w) else cv2.resize(roi, (w, h))

        if self._px is None or self._w != w or self._h != h:
            self._init(w, h)

        # ---- Layer 1: body outline bloom --------------------------------
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        raw_edges = cv2.Canny(gray, 40, 120)

        contours, _ = cv2.findContours(
            raw_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = 0.004 * w * h
        large = [c for c in contours if cv2.contourArea(c) > min_area]

        edge_mask = np.zeros((h, w), dtype=np.uint8)
        if large:
            cv2.drawContours(edge_mask, large, -1, 255, 1)

        # Extract edge pixel coords for particle attraction; subsample if huge
        rows, cols = np.where(edge_mask > 0)
        if len(rows) > 2000:
            sel = np.random.choice(len(rows), 2000, replace=False)
            rows, cols = rows[sel], cols[sel]
        ex = cols.astype(np.float32)
        ey = rows.astype(np.float32)

        # XRay base: all Canny edges (not just large contours) as faint cyan detail
        raw_f = raw_edges.astype(np.float32) / 255.0
        xray_blur = cv2.GaussianBlur(raw_edges.astype(np.float32), (7, 7), 0) / 255.0
        xray_base = np.clip(xray_blur * 0.5 + raw_f * 0.25, 0.0, 1.0)

        # Atmospheric glow: dilate → heavy blur → voltage-modulated brightness
        dilated = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8))
        bloom_f = cv2.GaussianBlur(
            dilated.astype(np.float32), (21, 21), 0
        ) / 255.0 * self._voltage

        # Re-add sharp edge on top for definition
        edge_f = edge_mask.astype(np.float32) / 255.0
        bloom_f = np.clip(bloom_f + edge_f, 0.0, 1.0)

        # Combine xray texture (dim) with body bloom (bright)
        combined_f = np.clip(xray_base * 0.65 + bloom_f, 0.0, 1.0)
        out = (combined_f[:, :, np.newaxis] * _BLOOM_COLOR).astype(np.uint8)

        # ---- Physics tick -----------------------------------------------
        self._step(w, h, ex, ey, self._dt)

        # ---- Layer 2: particle drawing ----------------------------------
        px, py = self._px, self._py

        # Discharge arcs: pairwise upper-triangle distance check
        dxp = px[:, np.newaxis] - px[np.newaxis, :]   # (N, N)
        dyp = py[:, np.newaxis] - py[np.newaxis, :]
        d2p = dxp * dxp + dyp * dyp
        ii, jj = np.triu_indices(self._n, k=1)
        close = d2p[ii, jj] < _ARC_THRESHOLD * _ARC_THRESHOLD
        arc_i, arc_j = ii[close][:_MAX_ARCS], jj[close][:_MAX_ARCS]
        arc_col = (int(_SPARK_COLOR[0]), int(_SPARK_COLOR[1]), int(_SPARK_COLOR[2]))
        for ai, aj in zip(arc_i, arc_j):
            cv2.line(
                out,
                (int(px[ai]), int(py[ai])),
                (int(px[aj]), int(py[aj])),
                arc_col, 1, cv2.LINE_AA,
            )

        # Discharge flash: expanding ring + dot at contact points
        dc_col = (
            int(_DISCHARGE_COLOR[0]),
            int(_DISCHARGE_COLOR[1]),
            int(_DISCHARGE_COLOR[2]),
        )
        for dpx, dpy in self._discharge_pts:
            cv2.circle(out, (dpx, dpy), 12, dc_col, 2)
            cv2.circle(out, (dpx, dpy), 3, dc_col, -1)

        # Particles: amber dots / hollow rings with per-frame size jitter
        sizes = np.random.randint(1, 4, self._n)
        for i in range(self._n):
            ix, iy = int(px[i]), int(py[i])
            if 0 <= ix < w and 0 <= iy < h:
                sz = int(sizes[i])
                if self._is_ring[i]:
                    cv2.circle(out, (ix, iy), sz + 1, arc_col, 1)
                else:
                    cv2.circle(out, (ix, iy), sz, arc_col, -1)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init(self, w: int, h: int) -> None:
        n = self._n
        self._px = np.random.uniform(0, w, n).astype(np.float32)
        self._py = np.random.uniform(0, h, n).astype(np.float32)
        self._state = np.zeros(n, dtype=np.int8)
        self._tx = np.zeros(n, dtype=np.float32)
        self._ty = np.zeros(n, dtype=np.float32)
        self._is_ring = np.random.rand(n) < 0.2
        self._w, self._h = w, h

    def _respawn(self, idx: np.ndarray, w: int, h: int) -> None:
        k = len(idx)
        self._px[idx] = np.random.uniform(0, w, k).astype(np.float32)
        self._py[idx] = np.random.uniform(0, h, k).astype(np.float32)
        self._state[idx] = _DRIFTING
        self._is_ring[idx] = np.random.rand(k) < 0.2

    def _step(
        self, w: int, h: int, ex: np.ndarray, ey: np.ndarray, dt: float
    ) -> None:
        """One physics tick: move particles, handle transitions, cluster burst."""
        px, py, state = self._px, self._py, self._state
        has_edges = len(ex) > 0
        self._discharge_pts = []

        # -- DRIFTING: random walk + edge snap ----------------------------
        d_idx = np.where(state == _DRIFTING)[0]
        if len(d_idx):
            nd = len(d_idx)
            px[d_idx] += np.random.uniform(-1, 1, nd) * _DRIFT_SPEED * dt
            py[d_idx] += np.random.uniform(-1, 1, nd) * _DRIFT_SPEED * dt
            px[d_idx] %= w
            py[d_idx] %= h

            if has_edges:
                # Vectorised nearest-edge distance for all drifting particles
                dpx = px[d_idx, np.newaxis] - ex[np.newaxis, :]  # (D, E)
                dpy = py[d_idx, np.newaxis] - ey[np.newaxis, :]
                d2 = dpx * dpx + dpy * dpy
                near_ei = np.argmin(d2, axis=1)             # (D,)
                near_d2 = d2[np.arange(nd), near_ei]

                snap = near_d2 < _SNAP_THRESHOLD * _SNAP_THRESHOLD
                spont = np.random.rand(nd) < _SPONT_PROB_PS * dt
                local_lock = np.where(snap | spont)[0]

                if len(local_lock):
                    g_idx = d_idx[local_lock]
                    self._tx[g_idx] = ex[near_ei[local_lock]]
                    self._ty[g_idx] = ey[near_ei[local_lock]]
                    state[g_idx] = _LOCKED_ON

        # -- LOCKED_ON: dart toward target with angular kick --------------
        l_idx = np.where(state == _LOCKED_ON)[0]
        if len(l_idx):
            dx = self._tx[l_idx] - px[l_idx]
            dy = self._ty[l_idx] - py[l_idx]
            dist = np.sqrt(dx * dx + dy * dy)

            # Particles that reached their target → discharge + respawn
            arrived = dist < _ARRIVAL_THRESHOLD
            arr_idx = l_idx[arrived]
            if len(arr_idx):
                for i in arr_idx:
                    self._discharge_pts.append((int(px[i]), int(py[i])))
                self._respawn(arr_idx, w, h)

            # Remaining: normalise direction, add angular kick, advance
            m_mask = ~arrived
            m_idx = l_idx[m_mask]
            if len(m_idx):
                dist_m = dist[m_mask]
                nz = dist_m > 0
                nx = np.where(nz, dx[m_mask] / dist_m, 0.0)
                ny = np.where(nz, dy[m_mask] / dist_m, 0.0)

                # Random angular kick for zigzag darting feel
                angle = np.random.uniform(-0.5, 0.5, len(m_idx))
                ca, sa = np.cos(angle), np.sin(angle)
                kx = nx * ca - ny * sa
                ky = nx * sa + ny * ca

                step = np.minimum(_LOCK_SPEED * dt, dist_m)
                px[m_idx] += kx * step
                py[m_idx] += ky * step

        # -- Cluster burst: 8-18 particles snap to one edge anchor --------
        if self._burst_timer >= self._next_burst and has_edges:
            self._burst_timer = 0.0
            self._next_burst = float(np.random.uniform(0.3, 1.2))
            anchor_i = np.random.randint(len(ex))
            burst_n = np.random.randint(8, 19)
            drifting_now = np.where(state == _DRIFTING)[0]
            if len(drifting_now) >= burst_n:
                burst_idx = np.random.choice(drifting_now, burst_n, replace=False)
            else:
                burst_idx = np.random.choice(self._n, min(burst_n, self._n), replace=False)
            self._tx[burst_idx] = ex[anchor_i]
            self._ty[burst_idx] = ey[anchor_i]
            state[burst_idx] = _LOCKED_ON
