# CLAUDE.md — houdini

Hackathon AR tool. Two hands tear open a glowing portal into a Roblox-adjacent particle world. Vibe: *Doctor Strange* meets children's game. Prioritise visual impact and real-time performance over clean architecture.

Run: `uv run main.py`

---

## Stack

Python · OpenCV · MediaPipe · pygame + PyOpenGL · GLSL · numpy

---

## Module contracts (do not break)

**`hands.py`**
```python
def get_portal(frame: np.ndarray) -> dict | None:
    # {"cx": float, "cy": float, "radius": float} — normalised 0–1, or None
```

**`particles.py`**
```python
def get_fbo_texture_id() -> int
def update(dt: float) -> None
def draw() -> None
```

**`portal.py`**
```python
def render(cam_tex_id: int, scene_tex_id: int, cx: float, cy: float, radius: float) -> None
```

`main.py` owns the loop. Mock values for dev: `cx=0.5, cy=0.5, radius=0.3`.

---

## overlay/ package

Quad-mapped content compositing system. A `BoxOverlay` holds one `BoxContent` and N quad regions; an `OverlayStack` chains overlays each frame.

**`Quad`** = `((x,y), (x,y), (x,y), (x,y))` — TL/TR/BR/BL, normalised 0–1.

**`BoxContent` ABC** — all subclasses implement:
```python
def render(self, w: int, h: int, roi: np.ndarray | None) -> np.ndarray: ...
# roi = live BGR pixels under the region; use for filters, ignore for generative content
```

**Live corner updates:** `overlay.set_region(i, new_quad)` — safe to call every frame.

**Arbitrary shape regions — `ShapeOverlay`:**
```python
from overlay import ShapeOverlay

shape = ShapeOverlay(content, alpha=0.8)
shape.set_polygon([(x0,y0), (x1,y1), ...])  # normalised 0–1 points, any N >= 3
stack.add(shape)
```
- Points are a closed path in normalised `[0,1]` coords — no need to repeat the first point at the end
- Self-intersecting / overlapping paths are fine; interior filled via non-zero winding rule
- `set_polygon()` simplifies the stroke with `cv2.approxPolyDP` and caches the rasterised mask — call it once for static shapes, or every frame for live updates
- Content fills the bounding box then is clipped to the polygon; no perspective warp

**Content types:**
- `overlay.effects.SolidContent(color)` — flat BGR fill ✅
- `overlay.video.VideoContent(path)` — mp4 loop (stub)
- `overlay.image.ImageContent(path)` — static image (stub)
- `overlay.filter.FilterContent(source, fn)` — wraps source + numpy transform (stub)
- `overlay.effects.GlitchContent(source)` — scan-line/channel-shift (stub)

---

## Progress

- [x] `camera/` — `CameraSource` ABC, `@register(priority)`, `open_camera()` factory
  - `opencv.py` priority 0 (generic), `avfoundation.py` priority 10 (Apple Silicon)
- [x] `main.py` — cv2.imshow loop; `q` to quit
- [x] `overlay/` — OverlayStack, BoxOverlay, BoxContent; SolidContent implemented
- [ ] `hands.py`
- [ ] `particles.py`
- [ ] `portal.py`

---

## Gotchas

- Portal radius → pixel space: aspect-ratio mapping must be exact or it looks wrong
- PyOpenGL FBO setup is verbose — don't refactor, just make it work
- MediaPipe struggles with fast motion / poor lighting
- No audio until all visuals are solid