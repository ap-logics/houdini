# CLAUDE.md — houdini

## What is this?

**houdini** is a real-time AR performance tool built for virality. It uses hand gestures (via MediaPipe) to control a sci-fi portal effect: two hands in frame tear open a glowing circular window into a parallel particle world, sized and positioned by the distance and midpoint between your index fingertips. The visual goal is "I ripped a hole in reality into Roblox" — cinematic, immediate, and absurd enough to be memeable.

This is a hackathon project. Code should be hacky but functional. Prioritise visual impact and real-time performance over clean architecture.

---

## Concept

- **Camera feed** is the base layer — you, your face, your room
- **Two hands** detected → a glowing portal appears between them
- **Inside the portal** is a separate 3D world: a particle system (floating orbs/cubes in a Roblox-adjacent colour palette) slowly rotating around a central axis
- **Portal edge** has GLSL-driven glow and chromatic aberration to sell the sci-fi tear effect
- **No hands / one hand** → portal closes (radius lerps to zero smoothly)
- Future: OBJ scene (e.g. Roblox map chunk) replaces or augments the particle world

---

## Stack

- **Python** — primary language
- **OpenCV** — webcam capture, frame handling
- **MediaPipe** — hand landmark detection
- **pygame + PyOpenGL** — window management and OpenGL context
- **GLSL** — portal composite shader (circle mask, glow ring, chromatic aberration)
- Particle system is pure Python/numpy + OpenGL draw calls; no physics engine

---

## Architecture

The project is split into parallel, loosely coupled tracks:

### Module contracts (do not break these)

**`hands.py`** — hand input
```python
def get_portal(frame: np.ndarray) -> dict | None:
    # Returns {"cx": float, "cy": float, "radius": float}  # all normalised 0.0–1.0
    # Returns None if fewer than 2 hands are detected
```

**`particles.py`** — 3D scene / other world
```python
def get_fbo_texture_id() -> int    # OpenGL texture handle for the rendered scene
def update(dt: float) -> None      # advance particle simulation
def draw() -> None                 # render scene into internal FBO
```

**`portal.py`** — GLSL composite
```python
def render(cam_tex_id: int, scene_tex_id: int, cx: float, cy: float, radius: float) -> None
    # Composites camera + portal scene onto screen
```

`main.py` owns the loop and wires these together. During development, A and C tracks should use mock portal values (`cx=0.5, cy=0.5, radius=0.3`) so they don't depend on Track B being ready.

---

## Visual priorities

1. **Portal visuals** — this is the hero. Edge FX, glow, and the scene inside must look good.
2. **Real-time performance** — target 30fps minimum on a laptop webcam. Don't over-engineer the particle system.
3. **Ease of recording** — the output is a pygame window; user screen-records it. Keep window at a standard aspect ratio (16:9 or 4:3).
4. **Audio** — deprioritised for now; can be added later as a separate track.

---

## Aesthetic reference

- Portal edge: think *Doctor Strange* sling ring portals — sparking, glowing, with a slight warp at the boundary
- World inside: Roblox colour palette (bright primaries, blocky geometry implied by cube particles), slowly drifting as if in zero gravity
- Overall vibe: cursed crossover between high-concept sci-fi and children's game. That tension is the joke and the appeal.

---

## What "done" looks like (MVP)

- [ ] Webcam feed renders fullscreen in pygame window
- [ ] Two hands in frame → glowing circle appears at correct position and size
- [ ] Inside circle: animated particle world visible
- [ ] Portal edge has visible glow (chromatic aberration is bonus)
- [ ] Smooth open/close lerp when hands appear/disappear
- [ ] Recordable at ~30fps on a standard laptop

---

## Known constraints / gotchas

- MediaPipe hand detection struggles with fast movement and poor lighting — good demo lighting matters
- PyOpenGL FBO setup is verbose; don't refactor it, just make it work
- The portal radius in normalised coords needs to be mapped carefully to pixel space in the shader — off-by-one in aspect ratio correction looks bad
- Roblox OBJ integration is planned but out of scope for MVP; the particle system is the placeholder that may become the final look
- Audio/music reactivity is explicitly deprioritised; don't add it until everything visual is solid