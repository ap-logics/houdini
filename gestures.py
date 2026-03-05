"""Gesture detection from hand landmarks.

Detects:
- pinch: thumb tip close to index tip (per hand)
- double click: two quick pinches within CLICK_WINDOW — clears canvas
- wave: 3 rapid horizontal direction changes — toggles erase mode
"""

import numpy as np

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

PINCH_ENTER = 0.045
PINCH_EXIT = 0.06

# Double click (two quick pinches)
CLICK_WINDOW = 0.5  # both pinches must happen within this many seconds

# Wave detection
WAVE_WINDOW = 1.2
WAVE_MIN_REVERSALS = 3
WAVE_MIN_MOVE = 0.025


class HandState:
    def __init__(self):
        self.pinching = False
        self.pinch_just_started = False
        self.pinch_just_ended = False
        self.index_pos = (0.0, 0.0)
        self.thumb_pos = (0.0, 0.0)
        self.wrist_pos = (0.0, 0.0)
        self.active = False

        # Wave tracking
        self._prev_wrist_x = None
        self._moving_dir = 0
        self._reversals = []
        self.wave_triggered = False

        # Double click tracking
        self._pinch_times = []
        self.double_click = False

    def update(self, landmarks, now: float):
        self.active = True
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        wrist = landmarks[WRIST]
        self.thumb_pos = (thumb[0], thumb[1])
        self.index_pos = (index[0], index[1])
        self.wrist_pos = (wrist[0], wrist[1])

        # Pinch
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

        # Double click
        self.double_click = False
        if self.pinch_just_started:
            self._pinch_times.append(now)
        self._pinch_times = [t for t in self._pinch_times if now - t < CLICK_WINDOW]
        if len(self._pinch_times) >= 2:
            self.double_click = True
            self._pinch_times.clear()

        # Wave detection
        self.wave_triggered = False
        wx = wrist[0]
        if self._prev_wrist_x is not None:
            dx = wx - self._prev_wrist_x
            if abs(dx) > WAVE_MIN_MOVE:
                new_dir = 1 if dx > 0 else -1
                if self._moving_dir != 0 and new_dir != self._moving_dir:
                    self._reversals.append(now)
                self._moving_dir = new_dir
        self._prev_wrist_x = wx

        self._reversals = [t for t in self._reversals if now - t < WAVE_WINDOW]
        if len(self._reversals) >= WAVE_MIN_REVERSALS:
            self.wave_triggered = True
            self._reversals.clear()

    def clear(self):
        self.active = False
        self.pinch_just_started = False
        self.pinch_just_ended = False
        self.wave_triggered = False
        self.double_click = False


class GestureDetector:
    def __init__(self):
        self.left = HandState()
        self.right = HandState()
        self.wave_triggered = False
        self.double_click = False

    def update(self, person: dict, dt: float, now: float):
        hands = person["hands"]
        handedness = person["handedness"]

        self.left.clear()
        self.right.clear()
        self.wave_triggered = False
        self.double_click = False

        for lm, label in zip(hands, handedness):
            if label == "Left":
                self.left.update(lm, now)
            else:
                self.right.update(lm, now)

        self.wave_triggered = self.left.wave_triggered or self.right.wave_triggered
        self.double_click = self.left.double_click or self.right.double_click

    @property
    def any_pinch_started(self) -> bool:
        return (self.left.active and self.left.pinch_just_started) or \
               (self.right.active and self.right.pinch_just_started)

    @property
    def any_pinch_ended(self) -> bool:
        return (self.left.active and self.left.pinch_just_ended) or \
               (self.right.active and self.right.pinch_just_ended)

    @property
    def any_pinching(self) -> bool:
        return (self.left.active and self.left.pinching) or \
               (self.right.active and self.right.pinching)

    @property
    def pinch_pos(self) -> tuple[float, float]:
        for h in (self.right, self.left):
            if h.active and h.pinching:
                return (
                    (h.index_pos[0] + h.thumb_pos[0]) / 2,
                    (h.index_pos[1] + h.thumb_pos[1]) / 2,
                )
        for h in (self.right, self.left):
            if h.active:
                return (
                    (h.index_pos[0] + h.thumb_pos[0]) / 2,
                    (h.index_pos[1] + h.thumb_pos[1]) / 2,
                )
        return (0.5, 0.5)
