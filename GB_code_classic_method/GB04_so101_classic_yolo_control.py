#!/usr/bin/env python3
"""
GB04_so101_classic_yolo_control.py

Same scripted hover/approach/descend/pull-back trajectory as
GB03_so101_classic_click_control.py, with the manual click replaced by the
project's existing fine-tuned YOLO weed detector
(GB_code_permabot_labo/GB02_see_weeds_yolo.py's find_plant_blobs_yolo(),
copied here unchanged -- same self-contained-script convention used
everywhere else in this project).

Cross-view association: YOLO runs independently on each camera view and
shows its own numbered detections (same picker UI as
GB_code_permabot_labo/GB03_see_weeds_preprocess.py's
show_detections_for_selection()); you confirm which numbered detection in
each view is the SAME physical weed. This keeps the association step simple
and verifiable rather than guessing an automatic match across views -- build
this ONLY after GB03's trajectory is confirmed working with manual clicks.

Build/run this last.
"""

import math
import os
import time
import traceback

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ultralytics import YOLO

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower

PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm_binnette"

CAMERAS = {
    "top": dict(index="/dev/video2", width=640, height=480, fps=30, fourcc="MJPG"),
    "right": dict(index="/dev/video4", width=640, height=480, fps=30, fourcc=None),
    "left": dict(index="/dev/video6", width=640, height=480, fps=30, fourcc=None),
}
WARMUP_FRAMES = 5

# ---------------------------------------------------------------------------
# YOLO -- same MODEL_DIR convention as GB_code_permabot_labo/GB02_see_weeds_yolo.py
# ---------------------------------------------------------------------------
MODEL_DIR = "../GB_code_permabot_labo/yolo_PermaBotLaboWeedDetection_pretest"
CONFIDENCE_THRESHOLD = 0.25


def find_plant_blobs_yolo(frame_bgr, model):
    """Copied from GB_code_permabot_labo/GB02_see_weeds_yolo.py unchanged."""
    results = model.predict(source=frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        classification = model.names[class_id]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append({
            "center": (cx, cy),
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "classification": classification,
            "confidence": confidence,
        })
    return detections


def show_detections_for_selection(frame_bgr, detections, camera_name):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    numbered = []
    for d in detections:
        x, y, w, h = d["bbox"]
        is_weed = "weed" in d["classification"].lower()
        color = "red" if is_weed else "deepskyblue"
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"))
        if is_weed:
            numbered.append(d)
            label = f"#{len(numbered)} {d['classification']} {d['confidence']:.2f}"
        else:
            label = f"{d['classification']} {d['confidence']:.2f}"
        ax.text(x, y - 6, label, color=color, fontsize=9, fontweight="bold",
                 bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))

    ax.set_title(f"'{camera_name}': {len(numbered)} weed(s) -- pick a number in the terminal", fontsize=12)
    ax.axis("off")
    plt.show(block=False)
    plt.pause(0.1)
    return fig, numbered


# ---------------------------------------------------------------------------
# Arm control -- copied verbatim from GB_code/GB01_so101_coord_control.py,
# identical to GB03_so101_classic_click_control.py's copy.
# ---------------------------------------------------------------------------
def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    if r > r_max:
        x *= r_max / r; y *= r_max / r; r = r_max
    if r < r_min and r > 0:
        x *= r_min / r; y *= r_min / r; r = r_min

    cos_t2 = max(-1.0, min(1.0, -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)))
    theta2 = math.pi - math.acos(cos_t2)

    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    return 90 - math.degrees(joint2), math.degrees(joint3) - 90


SAFE_RANGES = {
    "x": (0.05, 0.23),
    "y": (-0.08, 0.20),
    "shoulder_pan": (-90.0, 110.0),
    "pitch": (-70.0, 120.0),
    "wrist_roll": (-150.0, 150.0),
    "gripper": (0.0, 100.0),
}

KP = 0.5
CONTROL_FREQ = 50
CONVERGENCE_TH = 3.0
TIMEOUT = 5.0

HOVER_HEIGHT_OFFSET = 0.06
APPROACH_OFFSET = 0.02
DESCEND_Z = None
PULLBACK_DISTANCE = 0.04
WRIST_PITCH = 0.0
WRIST_ROLL = 0.0
GRIPPER = 0.0


def move_to_zero(robot, kp=KP, control_freq=CONTROL_FREQ, duration=3.0):
    print("Moving to zero position ...")
    zero = {j: 0.0 for j in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                               "wrist_flex", "wrist_roll", "gripper"]}
    dt = 1.0 / control_freq
    for _ in range(int(duration * control_freq)):
        obs = robot.get_observation()
        action = {f"{j}.pos": obs[f"{j}.pos"] + kp * (t - obs[f"{j}.pos"]) for j, t in zero.items()}
        robot.send_action(action)
        time.sleep(dt)
    print("At zero position.")


def p_control_to_target(robot, targets, kp=KP, control_freq=CONTROL_FREQ,
                         timeout=TIMEOUT, threshold=CONVERGENCE_TH):
    dt = 1.0 / control_freq
    max_steps = int(timeout * control_freq)
    total_error = float("inf")
    for step in range(max_steps):
        obs = robot.get_observation()
        action = {}
        total_error = 0.0
        for joint, target in targets.items():
            current = obs[f"{joint}.pos"]
            error = target - current
            total_error += abs(error)
            action[f"{joint}.pos"] = current + kp * error
        robot.send_action(action)
        if total_error < threshold:
            print(f"  Reached in {(step + 1) / control_freq:.2f}s (error = {total_error:.2f}deg)")
            return
        time.sleep(dt)
    print(f"  Timeout (final error = {total_error:.2f}deg)")


def run_removal_trajectory(robot, target_xyz):
    tx, ty, tz = target_xyz
    pan = math.degrees(math.atan2(ty, tx))
    r = math.hypot(tx, ty)
    print(f"  Target (robot frame): x={tx:.3f} y={ty:.3f} z={tz:.3f}  -> pan={pan:.1f}deg r={r:.3f}m")

    descend_z = DESCEND_Z if DESCEND_Z is not None else tz
    waypoints = [
        ("hover", r, descend_z + HOVER_HEIGHT_OFFSET),
        ("approach", r + APPROACH_OFFSET, descend_z + HOVER_HEIGHT_OFFSET),
        ("descend", r + APPROACH_OFFSET, descend_z),
        ("pullback", r + APPROACH_OFFSET - PULLBACK_DISTANCE, descend_z),
        ("retract", r, descend_z + HOVER_HEIGHT_OFFSET),
    ]
    for label, wr, wz in waypoints:
        wr = min(max(wr, SAFE_RANGES["x"][0]), SAFE_RANGES["x"][1])
        wz = min(max(wz, SAFE_RANGES["y"][0]), SAFE_RANGES["y"][1])
        shoulder_lift, elbow_flex = inverse_kinematics(wr, wz)
        wrist_flex = -shoulder_lift - elbow_flex + WRIST_PITCH
        print(f"  -> {label}: r={wr:.3f} z={wz:.3f}")
        p_control_to_target(robot, {
            "shoulder_pan": pan, "shoulder_lift": shoulder_lift, "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex, "wrist_roll": WRIST_ROLL, "gripper": GRIPPER,
        })
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Triangulation -- identical math to GB02/GB03
# ---------------------------------------------------------------------------
def undistort_point(u, v, camera_matrix, dist_coeffs):
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x, y = undistorted[0, 0]
    return np.array([x, y, 1.0])


def triangulate_dlt(rays_and_origins):
    A, b = [], []
    for origin, direction in rays_and_origins:
        d = direction / np.linalg.norm(direction)
        P = np.eye(3) - np.outer(d, d)
        A.append(P)
        b.append(P @ origin)
    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)
    point, *_ = np.linalg.lstsq(A, b, rcond=None)
    return point


def open_camera(name):
    cfg = CAMERAS[name]
    kwargs = dict(index_or_path=cfg["index"], width=cfg["width"], height=cfg["height"], fps=cfg["fps"])
    if cfg["fourcc"]:
        kwargs["fourcc"] = cfg["fourcc"]
    cam = OpenCVCamera(OpenCVCameraConfig(**kwargs))
    cam.connect()
    return cam


def capture_snapshot(cam, warmup=WARMUP_FRAMES):
    for _ in range(warmup):
        cam.read()
    return cam.read()


def pick_weed_in_view(cam, name, yolo_model):
    frame_rgb = capture_snapshot(cam)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    detections = find_plant_blobs_yolo(frame_bgr, yolo_model)
    weed_count = sum(1 for d in detections if "weed" in d["classification"].lower())
    if weed_count == 0:
        print(f"  '{name}': no weeds detected.")
        return None

    fig, numbered = show_detections_for_selection(frame_bgr, detections, name)
    choice = input(f"  '{name}': which weed is the target? [1-{len(numbered)}] (or 's' to skip this view): ").strip().lower()
    plt.close(fig)
    if choice == "s" or not (choice.isdigit() and 1 <= int(choice) <= len(numbered)):
        return None
    return numbered[int(choice) - 1]


def main():
    print("GB04 -- Classic YOLO-to-control (final step, no IL)")
    print("=" * 50)

    checkpoint_path = f"{MODEL_DIR}/weights/best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"No YOLO checkpoint at {checkpoint_path} -- check MODEL_DIR.")
        return
    print(f"Loading YOLO model: {checkpoint_path}")
    yolo_model = YOLO(checkpoint_path)

    names_raw = input(f"Cameras to triangulate with (comma-separated, >=2, from {list(CAMERAS.keys())}): ").strip()
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if len(names) < 2:
        print("Need at least 2 cameras.")
        return

    transforms, intrinsics = {}, {}
    for name in names:
        try:
            transforms[name] = np.load(f"camera_{name}_to_robot.npy")
            data = np.load(f"camera_{name}_intrinsics.npy", allow_pickle=True).item()
            intrinsics[name] = (data["camera_matrix"], data["dist_coeffs"])
        except FileNotFoundError as e:
            print(f"Missing calibration file for '{name}': {e}. Run GB01 first.")
            return

    cams = {name: open_camera(name) for name in names}
    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(robot_config)

    try:
        robot.connect()
        print("Connected.")
        move_to_zero(robot)

        while True:
            cmd = input("\nPress Enter to detect+remove a weed ('q' to quit): ").strip().lower()
            if cmd == "q":
                break

            rays = []
            for name in names:
                chosen = pick_weed_in_view(cams[name], name, yolo_model)
                if chosen is None:
                    continue
                u, v = chosen["center"]
                camera_matrix, dist_coeffs = intrinsics[name]
                ray_cam = undistort_point(u, v, camera_matrix, dist_coeffs)
                cam_to_robot = transforms[name]
                origin_robot = cam_to_robot[:3, 3]
                direction_robot = cam_to_robot[:3, :3] @ ray_cam
                rays.append((origin_robot, direction_robot))

            if len(rays) < 2:
                print("  Need the same weed selected in at least 2 views -- skipping.")
                continue

            target_xyz = triangulate_dlt(rays)
            run_removal_trajectory(robot, target_xyz)
            print("  Done -- returning to zero.")
            move_to_zero(robot)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("\nReturning to zero ...")
        try:
            move_to_zero(robot)
        except Exception:
            pass
        if robot.is_connected:
            robot.disconnect()
        for cam in cams.values():
            cam.disconnect()
        plt.close("all")
        print("Done.")


if __name__ == "__main__":
    main()
