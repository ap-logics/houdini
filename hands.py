import os
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
    """Detect hands and return per-person skeleton + perspective-warped box.

    Returns None if no complete hand pairs found.
    Returns a list of person dicts, each with:
        "hands": [left_landmarks, right_landmarks]  (21 (x,y) normalised each)
        "box": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] warped normalised corners
    """
    global _frame_ts
    _frame_ts += 33

    rgb = frame[:, :, ::-1]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect_for_video(mp_image, _frame_ts)

    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return None

    # Bucket hands by handedness — keep Z
    lefts = []
    rights = []
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        handedness = result.handedness[i][0].category_name
        if handedness == "Left":
            lefts.append(landmarks)
        else:
            rights.append(landmarks)

    # If all detected as same handedness, split by x-position of wrist
    if not lefts or not rights:
        all_hands = [[(lm.x, lm.y, lm.z) for lm in h] for h in result.hand_landmarks]
        all_hands.sort(key=lambda lm: lm[WRIST][0])
        mid = len(all_hands) // 2
        lefts = all_hands[mid:]
        rights = all_hands[:mid]

    pairs = _pair_hands(lefts, rights)
    if not pairs:
        return None

    people = []
    for idx, (left, right) in enumerate(pairs):
        # Compute smoothed depth delta
        raw_dz = _wrist_z(left) - _wrist_z(right)
        prev = _smooth_dz.get(idx, 0.0)
        smoothed = prev + SMOOTH_ALPHA * (raw_dz - prev)
        _smooth_dz[idx] = smoothed

        # 2D box corners from thumb/index tips
        flat_box = [
            (left[INDEX_TIP][0], left[INDEX_TIP][1]),
            (right[INDEX_TIP][0], right[INDEX_TIP][1]),
            (right[THUMB_TIP][0], right[THUMB_TIP][1]),
            (left[THUMB_TIP][0], left[THUMB_TIP][1]),
        ]
        warped_box = _warp_box(flat_box, smoothed)

        # Strip Z for skeleton drawing data
        hands_2d = [[(x, y) for x, y, z in h] for h in (left, right)]
        people.append({"hands": hands_2d, "box": warped_box})

    # Clean up stale smoothing entries
    for k in list(_smooth_dz):
        if k >= len(pairs):
            del _smooth_dz[k]

    return people
