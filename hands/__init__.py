import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

MAX_HANDS = 8  # support up to 4 people

_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
)

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

# Skeleton connections for drawing (full hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm
]

WARP_STRENGTH = 0.4  # how much Z difference warps the box
SMOOTH_ALPHA = 0.3   # EMA smoothing for Z (0 = ignore new, 1 = no smoothing)

_frame_ts = 0
_smooth_dz = {}  # person index -> smoothed dz


def _wrist_center(landmarks):
    return np.array([landmarks[WRIST][0], landmarks[WRIST][1]])


def _wrist_z(landmarks):
    return landmarks[WRIST][2]


def _pair_hands(lefts, rights):
    """Greedily pair lefts and rights by wrist proximity. Returns list of (left, right) pairs."""
    pairs = []
    used_rights = set()
    for left in lefts:
        best_dist = float("inf")
        best_idx = None
        left_wrist = _wrist_center(left)
        for j, right in enumerate(rights):
            if j in used_rights:
                continue
            dist = np.linalg.norm(left_wrist - _wrist_center(right))
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx is not None:
            pairs.append((left, rights[best_idx]))
            used_rights.add(best_idx)
    return pairs


def _warp_box(box, dz):
    """Perspective-warp box corners based on depth delta between hands.

    box corners: [left_index, right_index, right_thumb, left_thumb]
    dz > 0 means left hand is farther → shrink left side
    dz < 0 means right hand is farther → shrink right side
    """
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4

    warped = []
    for i, (x, y) in enumerate(box):
        is_left_side = i in (0, 3)  # left_index, left_thumb
        # positive dz = left farther = shrink left toward center
        scale = 1.0 - WARP_STRENGTH * dz if is_left_side else 1.0 + WARP_STRENGTH * dz
        scale = max(0.3, min(1.7, scale))  # clamp to avoid inversion
        wx = cx + (x - cx) * scale
        wy = cy + (y - cy) * scale
        warped.append((wx, wy))
    return warped


def get_hands(frame: np.ndarray) -> list | None:
    """Detect hands and return per-person data.

    Returns None if no hands detected.
    Returns a list of person dicts, each with:
        "hands": [landmarks, ...]  (1 or 2 hands, each 21 (x,y) normalised)
        "handedness": [str, ...]   ("Left" or "Right" per hand)
        "box": [...] or None       (warped box if 2 hands, None if 1)
    """
    global _frame_ts
    _frame_ts += 33

    rgb = frame[:, :, ::-1]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect_for_video(mp_image, _frame_ts)

    if not result.hand_landmarks:
        return None

    # Build per-hand data with handedness
    all_hands = []
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        landmarks_3d = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        landmarks_2d = [(lm.x, lm.y) for lm in hand_landmarks]
        handedness = result.handedness[i][0].category_name
        all_hands.append({
            "landmarks_3d": landmarks_3d,
            "landmarks_2d": landmarks_2d,
            "handedness": handedness,
        })

    # Group into people by pairing left+right hands near each other
    lefts = [h for h in all_hands if h["handedness"] == "Left"]
    rights = [h for h in all_hands if h["handedness"] == "Right"]

    people = []
    used_lefts = set()
    used_rights = set()

    # First: pair lefts and rights by wrist proximity
    if lefts and rights:
        left_3d = [h["landmarks_3d"] for h in lefts]
        right_3d = [h["landmarks_3d"] for h in rights]
        pairs = _pair_hands(left_3d, right_3d)
        for left_lm, right_lm in pairs:
            li = next(i for i, h in enumerate(lefts) if h["landmarks_3d"] is left_lm)
            ri = next(i for i, h in enumerate(rights) if h["landmarks_3d"] is right_lm)
            used_lefts.add(li)
            used_rights.add(ri)

            raw_dz = _wrist_z(left_lm) - _wrist_z(right_lm)
            idx = len(people)
            prev = _smooth_dz.get(idx, 0.0)
            smoothed = prev + SMOOTH_ALPHA * (raw_dz - prev)
            _smooth_dz[idx] = smoothed

            flat_box = [
                (left_lm[INDEX_TIP][0], left_lm[INDEX_TIP][1]),
                (right_lm[INDEX_TIP][0], right_lm[INDEX_TIP][1]),
                (right_lm[THUMB_TIP][0], right_lm[THUMB_TIP][1]),
                (left_lm[THUMB_TIP][0], left_lm[THUMB_TIP][1]),
            ]
            warped_box = _warp_box(flat_box, smoothed)

            left_2d = [(x, y) for x, y, z in left_lm]
            right_2d = [(x, y) for x, y, z in right_lm]
            people.append({
                "hands": [left_2d, right_2d],
                "handedness": ["Left", "Right"],
                "box": warped_box,
            })

    # Then: add unpaired hands as solo entries
    for i, h in enumerate(lefts):
        if i not in used_lefts:
            people.append({
                "hands": [h["landmarks_2d"]],
                "handedness": [h["handedness"]],
                "box": None,
            })
    for i, h in enumerate(rights):
        if i not in used_rights:
            people.append({
                "hands": [h["landmarks_2d"]],
                "handedness": [h["handedness"]],
                "box": None,
            })

    # Clean up stale smoothing entries
    for k in list(_smooth_dz):
        if k >= len(people):
            del _smooth_dz[k]

    return people if people else None


def draw_skeleton(
    frame: np.ndarray,
    people: list,
    colors: list[tuple[int, int, int]],
) -> None:
    """Draw hand landmarks and connections onto frame in-place."""
    h, w = frame.shape[:2]
    for i, person in enumerate(people):
        color = colors[i % len(colors)]
        for landmarks in person["hands"]:
            if len(landmarks) < 21:
                continue
            for a, b in HAND_CONNECTIONS:
                pt1 = (int(landmarks[a][0] * w), int(landmarks[a][1] * h))
                pt2 = (int(landmarks[b][0] * w), int(landmarks[b][1] * h))
                cv2.line(frame, pt1, pt2, color, 2)
            for lx, ly in landmarks:
                cv2.circle(frame, (int(lx * w), int(ly * h)), 3, color, -1)
