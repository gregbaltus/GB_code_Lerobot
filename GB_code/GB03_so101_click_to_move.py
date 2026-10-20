#!/usr/bin/env python3
"""
GB03_so101_click_to_move.py

Test script for the camera-robot calibration.
Loads so101_calibration.npy, shows a camera snapshot, lets the user
click a target position, and moves the arm there.

Requires GB02_so101_2Dcalibration.py to have been run first.
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
# Safe ranges
# ---------------------------------------------------------------------------
SAFE_X          = (0.05,   0.23)
SAFE_PAN        = (-90.0,  110.0)
SAFE_Y          = (-0.08,  0.20)
SAFE_PITCH      = (-70.0,  120.0)
SAFE_WRIST_ROLL = (-150.0, 150.0)
SAFE_GRIPPER    = (0.0,    100.0)

CALIBRATION_FILE = "so101_calibration.npy"

# P-controller parameters
KP             = 0.5
CONTROL_FREQ   = 50
CONVERGENCE_TH = 8.0
TIMEOUT        = 5.0


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
    dt          = 1.0 / control_freq
    max_steps   = int(timeout * control_freq)
    total_error = float("inf")
    for step in range(max_steps):
        obs         = robot.get_observation()
        action      = {}
        total_error = 0.0
        for joint, target in targets.items():
            current                = obs[f"{joint}.pos"]
            error                  = target - current
            total_error           += abs(error)
            action[f"{joint}.pos"] = current + kp * error
        robot.send_action(action)
        if total_error < threshold:
            print(f"  Reached in {(step + 1) / control_freq:.2f}s "
                  f"(error = {total_error:.2f}°)")
            return
        time.sleep(dt)
    print(f"  Timeout (final error = {total_error:.2f}°)")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def capture_snapshot(cap, warmup=5):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------
def pixel_to_robot(u, v, H):
    """Map pixel (u, v) → (x_robot, shoulder_pan) using the calibration homography."""
    pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    return float(pt[0][0][0]), float(pt[0][0][1])


def robot_to_pixel(x_robot, pan, H_inv):
    """Map (x_robot, shoulder_pan) → pixel (u, v) using the inverse homography."""
    pt = cv2.perspectiveTransform(
        np.array([[[x_robot, pan]]], dtype=np.float32), H_inv)
    return int(round(pt[0][0][0])), int(round(pt[0][0][1]))


# ---------------------------------------------------------------------------
# Snapshot display + click
# ---------------------------------------------------------------------------
def show_and_click(cap, last_robot_pos, H_inv, move_count):
    """
    Show a fresh camera snapshot and wait for one click.
    - Green dot  : last robot position (re-projected to pixels).
    - Returns (u, v), or None if the window was closed without clicking.
    """
    frame_rgb = capture_snapshot(cap)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    if last_robot_pos is not None:
        pu, pv = robot_to_pixel(*last_robot_pos, H_inv)
        ax.plot(pu, pv, 'go', markersize=14, markeredgecolor='white',
                markeredgewidth=1.5, label="last position")
        ax.annotate("last", (pu + 8, pv - 10),
                    color='lime', fontsize=10, fontweight='bold')

    ax.set_title(f"Move {move_count} — click your target, then close this window",
                 fontsize=13)
    ax.axis('off')

    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        return None
    return int(round(pts[0][0])), int(round(pts[0][1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SO-101 Click-to-Move  —  GB03")
    print("=" * 40)

    # Load calibration
    try:
        H     = np.load(CALIBRATION_FILE)
        H_inv = np.linalg.inv(H)
        print(f"Loaded calibration: {CALIBRATION_FILE}\n")
    except FileNotFoundError:
        print(f"Calibration file '{CALIBRATION_FILE}' not found.")
        print("Run GB02_so101_2Dcalibration.py first.")
        return

    port = input("USB port (default: /dev/ttyACM0): ").strip() or "/dev/ttyACM0"

    def ask_float(label, lo, hi, default):
        raw = input(f"  {label} [{lo} .. {hi}] (default {default}): ").strip()
        try:
            v = float(raw)
            return v if lo <= v <= hi else default
        except ValueError:
            return default

    print("\nFixed arm parameters (y should match what you used during calibration):")
    y_fixed    = ask_float("y — height (m)",            *SAFE_Y,          0.0)
    pitch      = ask_float("pitch (°)",                 *SAFE_PITCH,      0.0)
    wrist_roll = ask_float("wrist_roll (°)",            *SAFE_WRIST_ROLL, 0.0)
    gripper    = ask_float("gripper  0=closed 100=open", *SAFE_GRIPPER,   0.0)

    # Camera
    available = [i for i in range(5) if cv2.VideoCapture(i).isOpened()]
    if not available:
        print("No cameras found.")
        return
    print(f"\nAvailable cameras: {available}")
    cam_idx = int(input("Select camera index: ").strip())
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Robot
    robot_config = SO101FollowerConfig(port=port, id="my_awesome_follower_arm")
    robot        = SO101Follower(robot_config)

    last_robot_pos = None
    move_count     = 0

    try:
        robot.connect()
        print("Connected.")
        move_to_zero(robot)

        print("\nClose the snapshot window without clicking to exit at any time.\n")

        while True:
            move_count += 1
            click = show_and_click(cap, last_robot_pos, H_inv, move_count)

            if click is None:
                print("Window closed — exiting.")
                break

            u, v = click
            x_robot, pan = pixel_to_robot(u, v, H)

            print(f"\n  Clicked pixel : u={u}  v={v}")
            print(f"  Mapped to     : x={x_robot:.3f} m   pan={pan:.1f}°")

            # Safety check
            if not (SAFE_X[0] <= x_robot <= SAFE_X[1]):
                print(f"  ! x={x_robot:.3f} m out of safe range {SAFE_X} — click again.")
                move_count -= 1
                continue
            if not (SAFE_PAN[0] <= pan <= SAFE_PAN[1]):
                print(f"  ! pan={pan:.1f}° out of safe range {SAFE_PAN} — click again.")
                move_count -= 1
                continue

            shoulder_lift, elbow_flex = inverse_kinematics(x_robot, y_fixed)
            wrist_flex = -shoulder_lift - elbow_flex + pitch

            print("  Moving …")
            p_control_to_target(robot, {
                "shoulder_pan":  pan,
                "shoulder_lift": shoulder_lift,
                "elbow_flex":    elbow_flex,
                "wrist_flex":    wrist_flex,
                "wrist_roll":    wrist_roll,
                "gripper":       gripper,
            })

            last_robot_pos = (x_robot, pan)

            again = input("\n  Move again? (y/n): ").strip().lower()
            if again != 'y':
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
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
