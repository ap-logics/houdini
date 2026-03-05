# CLAUDE.md — houdini

Hackathon AR tool. Two hands tear open a glowing portal into a Roblox-adjacent particle world. Vibe: *Doctor Strange* meets children's game. Prioritise visual impact and real-time performance over clean architecture.

Run: `uv run main.py`

---

## Planning & design

Before building any new feature:
1. **Define the gesture/interaction** — what does the user do, what is the visual response?
2. **Spec the module contract** — add it to this file before writing code (see Module contracts below)
3. **Consider frame budget** — we're CPU-bound at ~30 fps; anything heavy needs to be async or cached
4. **Wire it in `main.py` last** — build and test modules independently where possible

Keep this file current. When a module ships, update the Progress checklist and contracts.

---

## Stack

Python · OpenCV · MediaPipe · pygame + PyOpenGL · GLSL · numpy

---

## Module contracts (do not break)

**`hands/`** — MediaPipe hand tracking
```python
def get_hands(frame: np.ndarray) -> list | None:
    # Returns list of person dicts, or None if no hands detected.
    # Each person: {"hands": [landmarks, ...], "handedness": ["Left"|"Right", ...], "box": [...] | None}
    # landmarks: 21 (x, y) normalised 0-1 tuples
    # box: 4-point warped quad (index tips + thumb tips) when two hands paired, else None

def draw_skeleton(frame: np.ndarray, people: list, colors: list[tuple]) -> None:
    # Draws hand landmarks in-place onto frame
```

**`gestures.py`** — gesture state from hand landmarks
```python
class GestureDetector:
    left: HandState
    right: HandState
    wave_triggered: bool       # True for one frame when wave detected
    double_click: bool         # True for one frame when two quick pinches

    def update(self, person: dict, dt: float, now: float) -> None: ...
    # Convenience properties:
    any_pinch_started: bool
    any_pinch_ended: bool
    any_pinching: bool
    pinch_pos: tuple[float, float]  # midpoint of active pinch, normalised 0-1
```

Gestures detected:
- **Pinch** — thumb tip to index tip distance < 0.045 (hysteresis exit at 0.06)
- **Double click** — two pinches within 0.5 s → `double_click = True` for one frame
- **Wave** — 3+ wrist direction reversals within 1.2 s → `wave_triggered = True` for one frame

**`draw.py`** — vertex drawing state machine
```python
class DrawState:
    vertices: list[tuple[float, float]]  # logical vertex positions, normalised 0-1
    closed: bool
    erasing: bool

    def update(self, gesture: GestureDetector, dt: float) -> None: ...
    def update_smooth(self, dt: float) -> None: ...  # smoothing only, no gesture input
    def render(self, frame: np.ndarray) -> np.ndarray: ...
    def clear(self) -> None: ...
```

Draw states:
- **DRAWING** — pinch places vertex; pinch near first vertex (≥3 verts) closes shape
- **EDIT** — closed shape; pinch grabs nearest vertex and drags it
- **ERASE** — wave toggles; pinch deletes nearest vertex; double-click or empty → exit
- **Double click** anywhere → clear everything, back to DRAWING

Display positions are lerp-smoothed toward logical positions each frame (`LERP_SPEED = 12`).

**`particles.py`** — not yet implemented
```python
def get_fbo_texture_id() -> int
def update(dt: float) -> None
def draw() -> None
```

**`portal.py`** — not yet implemented
```python
def render(cam_tex_id: int, scene_tex_id: int, cx: float, cy: float, radius: float) -> None
```

`main.py` owns the loop. Mock values for portal dev: `cx=0.5, cy=0.5, radius=0.3`.

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
- `set_polygon()` simplifies the stroke with `cv2.approxPolyDP` and caches the rasterised mask
- Content fills the bounding box then is clipped to the polygon; no perspective warp

**Content types:**
- `overlay.effects.SolidContent(color)` — flat BGR fill
- `overlay.effects.XRayContent()` — Canny edges + neon cyan glow + scanlines on live ROI
- `overlay.effects.OBJContent(path)` — CPU-rasterised 3D OBJ renderer in overlay box
- `overlay.video.VideoContent(path)` — mp4 loop (stub)
- `overlay.image.ImageContent(path)` — static image (stub)
- `overlay.filter.FilterContent(source, fn)` — wraps source + numpy transform (stub)
- `overlay.effects.GlitchContent(source)` — scan-line/channel-shift (stub)

---

## Current flow (main.py)

1. Capture frame → flip horizontally (mirror mode)
2. `get_hands()` → `GestureDetector.update()` → `DrawState.update()`
3. If shape closed: `ShapeOverlay(XRayContent)` fills it; alpha=0 when not closed
4. `OverlayStack.render()` composites effects
5. `DrawState.render()` draws polygon + vertices on top
6. HUD: mode | vert count | gesture hints

---

## Progress

- [x] `camera/` — `CameraSource` ABC, `@register(priority)`, `open_camera()` factory
  - `opencv.py` priority 0 (generic), `avfoundation.py` priority 10 (Apple Silicon)
- [x] `main.py` — cv2.imshow loop; `q` to quit; mirror flip
- [x] `overlay/` — OverlayStack, BoxOverlay, ShapeOverlay, BoxContent; SolidContent implemented
- [x] `overlay/effects/xray.py` — XRayContent (Canny + neon glow + scanlines)
- [x] `overlay/effects/obj_content.py` — CPU-rasterised 3D OBJ renderer
- [x] `hands/` — MediaPipe landmarker; multi-person pairing; depth-warped box; skeleton draw
- [x] `gestures.py` — pinch (hysteresis), double-click, wave detection per hand
- [x] `draw.py` — vertex state machine: DRAWING / EDIT / ERASE; lerp smoothing
- [ ] `particles.py`
- [ ] `portal.py`

---

## Gotchas

- Portal radius → pixel space: aspect-ratio mapping must be exact or it looks wrong
- PyOpenGL FBO setup is verbose — don't refactor, just make it work
- MediaPipe handedness is mirrored relative to camera — frame is flipped in `main.py` to compensate
- MediaPipe struggles with fast motion / poor lighting
- `get_hands()` only uses `people[0]` in main — multi-person support is tracked but not wired up
- No audio until all visuals are solid
