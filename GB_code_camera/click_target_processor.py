#!/usr/bin/env python3
"""
click_target_processor.py

Shared helper for GB01_record_click_to_preprocess.py and GB01_eval_click_to_preprocess.py.

Idea: instead of the imitation-learning pipeline seeing the raw "top" camera
frame, a human clicks a target pixel once per episode and a crosshair (one
vertical + one horizontal line) is burned into the "top" image at that pixel
before the frame reaches the dataset writer / the policy.

This is implemented as an ObservationProcessorStep, inserted into
robot_observation_processor. That object is applied unconditionally inside
lerobot's record_loop() -- before the branch that checks whether the episode
is teleop-recorded or policy-driven, and before the frame is handed to either
the dataset writer or predict_action(). So the exact same step, used in both
GB01_record_click_to_preprocess.py (teleop) and GB01_eval_click_to_preprocess.py
(policy), guarantees the annotation is identical between recorded training
data and live inference -- there's no separate code path to keep in sync.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from lerobot.processor import ObservationProcessorStep, RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation


class ClickTargetOverlayStep(ObservationProcessorStep):
    """
    Draws a crosshair at self.target_uv on the observation image keyed by
    self.image_key. Call set_target(u, v) before each episode -- the same
    target is then used for every frame of that episode until changed.
    If target_uv is None, the observation passes through unchanged.
    """

    def __init__(self, image_key="top", color=(0, 255, 0), thickness=2):
        self.image_key = image_key
        self.color = color  # (0, 255, 0) = green, same value in RGB and BGR ordering
        self.thickness = thickness
        self.target_uv = None

    def set_target(self, u, v):
        self.target_uv = (int(round(u)), int(round(v)))

    def observation(self, observation: dict) -> dict:
        if self.target_uv is None or self.image_key not in observation:
            return observation

        frame = np.array(observation[self.image_key], copy=True)
        h, w = frame.shape[:2]
        u = max(0, min(w - 1, self.target_uv[0]))
        v = max(0, min(h - 1, self.target_uv[1]))

        cv2.line(frame, (u, 0), (u, h - 1), self.color, self.thickness)
        cv2.line(frame, (0, v), (w - 1, v), self.color, self.thickness)

        new_observation = dict(observation)
        new_observation[self.image_key] = frame
        return new_observation

    def get_config(self) -> dict:
        return {"image_key": self.image_key, "color": self.color, "thickness": self.thickness}

    def transform_features(self, features):
        return features  # pixel content changes, shapes/dtypes don't


def make_robot_observation_processor_with_click_target(overlay_step):
    """
    Build a robot_observation_processor pipeline that runs overlay_step,
    mirroring lerobot's make_default_robot_observation_processor() but with
    our step in place of the default identity one.
    """
    return RobotProcessorPipeline(
        steps=[overlay_step],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def wait_for_click_on_image(image, prompt):
    """
    Show `image` (numpy array) with matplotlib, wait for one click.
    Returns (u, v), or None if the window was closed without clicking.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title(prompt + "  —  click the target, then close this window", fontsize=12)
    ax.axis('off')

    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        return None
    return int(round(pts[0][0])), int(round(pts[0][1]))
