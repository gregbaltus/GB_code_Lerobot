#!/usr/bin/env python3
"""
GB02_so101_2Dcalibration.py

Calibrates the transformation from camera pixel coordinates (u, v) to
robot coordinates (x_robot, shoulder_pan).

Procedure:
  1. The arm visits 8 hardcoded positions spread across the workspace.
  2. For each position the user clicks on the wrist-joint sticker in the
     live camera image.
  3. The script computes a homography from the 8 point pairs and saves it
     to so101_calibration.npy.

Load in the inference script with:
    H = np.load("so101_calibration.npy")
    pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    x_robot, shoulder_pan = pt[0][0]
"""

import math
import time
import traceback
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig

# ---------------------------------------------------------------------------
# 8 calibration positions: (x_robot in meters, shoulder_pan in degrees)
# Spread across the reachable workspace for good homography coverage
# ---------------------------------------------------------------------------
CALIBRATION_POSITIONS = [
#     (0.10, -50.0),  # near-left
#     (0.10,   0.0),  # near-center
#     (0.10,  50.0),  # near-right
#     (0.15, -40.0),  # mid-left
#     (0.15,  40.0),  # mid-right
#     (0.20, -30.0),  # far-left
#     (0.20,   0.0),  # far-center
#     (0.20,  30.0),  # far-right
# ]
    (0.10, -20.0),  # near-left
    (0.10,  30.0),  # near-right
    (0.05, -10.0),  # mid-left
    (0.05,  20.0),  # mid-right
    (0.05, -5.0),  # far-left
    (0.05,  30.0),  # far-right
    #(0.25, -5.0),  # extra-far-left
    #(0.25,  4.0),  # extra-far-right
    (0.15, 35.0),    # extra-near-right
    (0.15,45.0)     # extra-near-left
]


CALIBRATION_FILE = "so101_calibration.npy"

# P-controller parameters
KP             = 0.5
CONTROL_FREQ   = 50
CONVERGENCE_TH = 3.0
TIMEOUT        = 5.0

# Safe y (height) range
Y_MIN, Y_MAX = -0.08, 0.20
Y_DEFAULT    = 0.10


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------
def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r     = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    if r > r_max:
        x *= r_max / r; y *= r_max / r; r = r_max
    if r < r_min and r > 0:
        x *= r_min / r; y *= r_min / r; r = r_min

    cos_t2 = max(-1.0, min(1.0, -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)))
    theta2 = math.pi - math.acos(cos_t2)

    beta   = math.atan2(y, x)
    gamma  = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    return 90 - math.degrees(joint2), math.degrees(joint3) - 90


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def move_to_zero(robot, kp=KP, control_freq=CONTROL_FREQ, duration=3.0):
    print("Moving to zero position …")
    zero = {j: 0.0 for j in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                               "wrist_flex", "wrist_roll", "gripper"]}
    dt = 1.0 / control_freq
    for _ in range(int(duration * control_freq)):
        obs    = robot.get_observation()
        action = {f"{j}.pos": obs[f"{j}.pos"] + kp * (t - obs[f"{j}.pos"])
                  for j, t in zero.items()}
        robot.send_action(action)
        time.sleep(dt)
    print("At zero position.")


def p_control_to_target(robot, targets, kp=KP, control_freq=CONTROL_FREQ,
                        timeout=TIMEOUT, threshold=CONVERGENCE_TH):
    dt        = 1.0 / control_freq
    max_steps = int(timeout * control_freq)
    total_error = float("inf")
    for step in range(max_steps):
        obs         = robot.get_observation()
        action      = {}
        total_error = 0.0
        for joint, target in targets.items():
            current              = obs[f"{joint}.pos"]
            error                = target - current
            total_error         += abs(error)
            action[f"{joint}.pos"] = current + kp * error
        robot.send_action(action)
        if total_error < threshold:
            print(f"  Reached in {(step + 1) / control_freq:.2f}s "
                  f"(error = {total_error:.2f}°)")
            return
        time.sleep(dt)
    print(f"  Timeout (final error = {total_error:.2f}°)")


# ---------------------------------------------------------------------------
# Snapshot + click capture (matplotlib — avoids OpenCV Qt backend issues)
# ---------------------------------------------------------------------------
def capture_snapshot(cap, warmup=5):
    """Discard a few stale frames then return an RGB snapshot."""
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def wait_for_click(cap, prompt, recorded_pixels):
    """Show a camera snapshot in matplotlib, wait for one click. Returns (u, v)."""
    frame_rgb = capture_snapshot(cap)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    # Draw already confirmed points in green
    for idx, (u, v) in enumerate(recorded_pixels):
        ax.plot(u, v, 'go', markersize=8)
        ax.annotate(str(idx + 1), (u + 6, v - 6),
                    color='lime', fontsize=11, fontweight='bold')

    ax.set_title(prompt + "  —  click on the sticker, then close this window",
                 fontsize=12)
    ax.axis('off')

    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        raise KeyboardInterrupt("No click recorded — window closed without clicking.")

    return int(round(pts[0][0])), int(round(pts[0][1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SO-101 Camera-Robot Calibration  —  GB_so101_2Dcalibration")
    print("=" * 55)

    port = input("USB port (default: /dev/ttyACM0): ").strip() or "/dev/ttyACM0"

    y_raw = input(f"Fixed arm height y [{Y_MIN} .. {Y_MAX}] "
                  f"(default {Y_DEFAULT}): ").strip()
    try:
        y_fixed = float(y_raw)
        if not (Y_MIN <= y_fixed <= Y_MAX):
            print(f"Out of range, using default {Y_DEFAULT}")
            y_fixed = Y_DEFAULT
    except ValueError:
        y_fixed = Y_DEFAULT
    print(f"Using height y = {y_fixed} m\n")

    # Camera selection
    available = [i for i in range(5)
                 if cv2.VideoCapture(i).isOpened()]
    if not available:
        print("No cameras found.")
        return
    print(f"Available cameras: {available}")
    cam_idx = int(input(f"Select camera index: ").strip())
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Robot connection
    robot_config = SO101FollowerConfig(port=port, id="my_awesome_follower_arm")
    robot        = SO101Follower(robot_config)

    pixel_points = []  # list of [u, v]
    robot_points = []  # list of [x_robot, shoulder_pan]

    try:
        robot.connect()
        print("Connected.\n")
        move_to_zero(robot)

        print(f"Calibration: {len(CALIBRATION_POSITIONS)} positions to visit.")
        print("For each one: wait for the arm to stop, then click on the sticker.\n")

        for i, (x_robot, pan) in enumerate(CALIBRATION_POSITIONS):
            print(f"--- Point {i + 1}/{len(CALIBRATION_POSITIONS)}"
                  f"  x={x_robot:.3f} m  pan={pan:.1f}° ---")

            shoulder_lift, elbow_flex = inverse_kinematics(x_robot, y_fixed)
            wrist_flex = -shoulder_lift - elbow_flex  # pitch = 0, gripper level

            p_control_to_target(robot, {
                "shoulder_pan":  pan,
                "shoulder_lift": shoulder_lift,
                "elbow_flex":    elbow_flex,
                "wrist_flex":    wrist_flex,
                "wrist_roll":    0.0,
                "gripper":       0.0,
            })

            time.sleep(0.5)  # let the arm fully settle before clicking

            prompt = (f"Point {i + 1}/{len(CALIBRATION_POSITIONS)}"
                      f"  —  click on the sticker")
            u, v = wait_for_click(cap, prompt, pixel_points)

            print(f"  Recorded pixel ({u}, {v})"
                  f" ↔ robot ({x_robot:.3f} m, {pan:.1f}°)\n")

            pixel_points.append([u, v])
            robot_points.append([x_robot, pan])

        # Compute homography: pixels → robot coordinates
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(robot_points, dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC)

        if H is None:
            print("Homography computation failed — check calibration points.")
            return

        # Reprojection error
        src_h  = np.concatenate([src, np.ones((len(src), 1))], axis=1)
        proj   = (H @ src_h.T).T
        proj  /= proj[:, 2:3]
        errors = np.linalg.norm(proj[:, :2] - dst, axis=1)
        print(f"Homography computed successfully.")
        print(f"Reprojection error — mean: {errors.mean():.4f}  "
              f"max: {errors.max():.4f}")

        np.save(CALIBRATION_FILE, H)
        print(f"Calibration saved to {CALIBRATION_FILE}")

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("\nReturning to zero …")
        try:
            move_to_zero(robot)
        except Exception:
            pass
        robot.disconnect()
        cap.release()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
