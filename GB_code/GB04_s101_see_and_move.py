#!/usr/bin/env python3
"""
GB04_s101_see_and_move.py

YOLO-based see-and-move script.
The user names an object; YOLO detects it in a camera snapshot;
the robot moves its wrist over the detected object using the
calibration homography from GB02.
"""

import math
import time
import traceback
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics import YOLO

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

CALIBRATION_FILE         = "so101_calibration.npy"
YOLO_MODEL_FILE          = "yolo11x.pt"
DETECTION_DISPLAY_SECONDS = 3.0

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
    return frame  # BGR — YOLO expects BGR, matplotlib gets cvtColor'd later


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------
def pixel_to_robot(u, v, H):
    pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    return float(pt[0][0][0]), float(pt[0][0][1])


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
def detect_object(model, frame_bgr, obj_name):
    """
    Run YOLO on frame_bgr. Return (u, v, conf, bbox) for the highest-confidence
    detection whose class name matches obj_name (exact first, then partial).
    Returns None if nothing found.
    """
    # Match class name
    class_ids = [k for k, v in model.names.items()
                 if v.lower() == obj_name.lower()]
    if not class_ids:
        class_ids = [k for k, v in model.names.items()
                     if obj_name.lower() in v.lower()]
    if not class_ids:
        return None

    results = model(frame_bgr, verbose=False)[0]

    best_conf = -1.0
    best_box  = None
    for box in results.boxes:
        if int(box.cls[0]) not in class_ids:
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box  = box

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    return int(round(u)), int(round(v)), best_conf, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Detection display
# ---------------------------------------------------------------------------
def show_detection(frame_bgr, obj_name, detection, display_seconds):
    """Show snapshot with bounding box for display_seconds, then auto-close."""
    u, v, conf, (x1, y1, x2, y2) = detection
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.plot(u, v, 'r+', markersize=18, markeredgewidth=2)
    ax.text(x1, y1 - 8, f"{obj_name}  {conf:.2f}",
            color='red', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='none'))

    ax.set_title(f"Detected: {obj_name} — moving in {display_seconds:.0f}s …",
                 fontsize=13)
    ax.axis('off')

    plt.draw()
    plt.pause(display_seconds)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SO-101 See-and-Move  —  GB04")
    print("=" * 40)

    # Load calibration
    try:
        H = np.load(CALIBRATION_FILE)
        print(f"Loaded calibration: {CALIBRATION_FILE}")
    except FileNotFoundError:
        print(f"Calibration file '{CALIBRATION_FILE}' not found.")
        print("Run GB02_so101_2Dcalibration.py first.")
        return

    # Load YOLO
    print(f"Loading YOLO model: {YOLO_MODEL_FILE} …")
    model = YOLO(YOLO_MODEL_FILE)
    print("YOLO ready.\n")

    port = input("USB port (default: /dev/ttyACM0): ").strip() or "/dev/ttyACM0"

    def ask_float(label, lo, hi, default):
        raw = input(f"  {label} [{lo} .. {hi}] (default {default}): ").strip()
        try:
            v = float(raw)
            return v if lo <= v <= hi else default
        except ValueError:
            return default

    print("\nFixed arm parameters (y should match what you used during calibration):")
    y_fixed    = ask_float("y — height (m)",             *SAFE_Y,          0.0)
    pitch      = ask_float("pitch (°)",                  *SAFE_PITCH,      0.0)
    wrist_roll = ask_float("wrist_roll (°)",             *SAFE_WRIST_ROLL, 0.0)
    gripper    = ask_float("gripper  0=closed 100=open", *SAFE_GRIPPER,    0.0)

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

    try:
        robot.connect()
        print("Connected.")
        move_to_zero(robot)

        print("\nType an object name at each prompt (e.g. 'cup', 'bottle'), or 'q' to quit.\n")

        while True:
            obj_name = input("Object to find: ").strip()
            if not obj_name or obj_name.lower() == 'q':
                break

            print("  Capturing snapshot and running YOLO …")
            frame_bgr = capture_snapshot(cap)
            detection = detect_object(model, frame_bgr, obj_name)

            if detection is None:
                print(f"  '{obj_name}' not detected. Check the name or try again.\n")
                # Print known YOLO classes as a hint
                known = sorted(model.names.values())
                print(f"  Known classes include: {', '.join(known[:20])} …\n")
                continue

            u, v, conf, _ = detection
            x_robot, pan  = pixel_to_robot(u, v, H)

            print(f"  Found '{obj_name}'  conf={conf:.2f}  pixel=({u}, {v})")
            print(f"  Mapped to: x={x_robot:.3f} m   pan={pan:.1f}°")

            # Safety check before moving
            if not (SAFE_X[0] <= x_robot <= SAFE_X[1]):
                print(f"  ! x={x_robot:.3f} m out of safe range {SAFE_X} — skipping.\n")
                continue
            if not (SAFE_PAN[0] <= pan <= SAFE_PAN[1]):
                print(f"  ! pan={pan:.1f}° out of safe range {SAFE_PAN} — skipping.\n")
                continue

            # Show detection window, then move
            show_detection(frame_bgr, obj_name, detection, DETECTION_DISPLAY_SECONDS)

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
            print()

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
