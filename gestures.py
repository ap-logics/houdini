"""Gesture detection from hand landmarks.

Detects:
- pinch: thumb tip close to index tip (per hand)
- clap: both wrists rapidly close together
"""

import numpy as np

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

PINCH_ENTER = 0.045   # normalised distance to start pinch
PINCH_EXIT = 0.06     # normalised distance to release pinch
CLAP_DIST = 0.12      # wrists must be this close
CLAP_VELOCITY = 0.3   # wrists must be closing this fast (units/s)


class HandState:
    def __init__(self):
        self.pinching = False
        self.pinch_just_started = False
        self.pinch_just_ended = False
        self.index_pos = (0.0, 0.0)   # normalised
        self.thumb_pos = (0.0, 0.0)

    def update(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        self.thumb_pos = (thumb[0], thumb[1])
        self.index_pos = (index[0], index[1])

        dist = ((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2) ** 0.5

        was_pinching = self.pinching
        if self.pinching:
            if dist > PINCH_EXIT:
                self.pinching = False
        else:
            if dist < PINCH_ENTER:
                self.pinching = True

        self.pinch_just_started = self.pinching and not was_pinching
        self.pinch_just_ended = not self.pinching and was_pinching


class GestureDetector:
    def __init__(self):
        self.left = HandState()
        self.right = HandState()
        self._prev_wrist_dist = None
        self.clap_triggered = False

    def update(self, person: dict, dt: float):
        """Update from a person dict with 'hands': [left_landmarks, right_landmarks]."""
        left_lm, right_lm = person["hands"]
        self.left.update(left_lm)
        self.right.update(right_lm)

        # Clap detection
        lw = np.array([left_lm[WRIST][0], left_lm[WRIST][1]])
        rw = np.array([right_lm[WRIST][0], right_lm[WRIST][1]])
        wrist_dist = np.linalg.norm(lw - rw)

        self.clap_triggered = False
        if self._prev_wrist_dist is not None and dt > 0:
            velocity = (self._prev_wrist_dist - wrist_dist) / dt  # positive = closing
            if wrist_dist < CLAP_DIST and velocity > CLAP_VELOCITY:
                self.clap_triggered = True

        self._prev_wrist_dist = wrist_dist
