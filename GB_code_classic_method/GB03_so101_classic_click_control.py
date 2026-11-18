#!/usr/bin/env python3
"""
GB03_so101_classic_click_control.py

The explicit non-YOLO version of the classical (no imitation learning)
weeding pipeline: YOU click the weed in 2+ camera views, the script
triangulates its 3D position in the robot's own base frame (reusing
GB02_so101_triangulate_test.py's math), then runs a pure scripted
trajectory -- no policy, no learning -- to remove it:

    hover above target -> move to approach offset -> descend to the
    measured depth -> pull-back/uprooting stroke -> retract

This exists so triangulation + the control trajectory can be validated (and
paper results produced) independently of detector quality -- YOLO is only
wired in afterwards, in GB04, once this is confirmed working.

Motion reuses GB_code/GB01_so101_coord_control.py's primitives UNCHANGED:
the 2-link planar inverse_kinematics(r, z) for reach/height, with
shoulder_pan set from atan2(target_y, target_x) -- the same reduction from a
3D target to (pan + planar r/z) already used throughout this project, so no
new IK was written here.

SAFETY: dry-run with DESCEND_Z set well above the soil first (see
SAFE_RANGES / TRAJECTORY constants below) to confirm waypoint ordering
before letting the hoe actually descend into the bed.
"""

import math
import time
import traceback

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower

# ---------------------------------------------------------------------------
# Robot -- same as GB01_so101_charuco_calibrate.py (no teleop needed here,
# the arm is scripted end to end).
# ---------------------------------------------------------------------------
PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm_binnette"

CAMERAS = {
    "top": dict(index="/dev/video2", width=640, height=480, fps=30, fourcc="MJPG"),
    "right": dict(index="/dev/video4", width=640, height=480, fps=30, fourcc=None),
    "left": dict(index="/dev/video6", width=640, height=480, fps=30, fourcc=None),
}
WARMUP_FRAMES = 5

# ---------------------------------------------------------------------------
# Inverse kinematics -- copied verbatim from GB_code/GB01_so101_coord_control.py
# (this project's convention: self-contained scripts, not cross-folder imports)
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


# ---------------------------------------------------------------------------
# Trajectory shape -- TUNE THESE before real soil contact.
# ---------------------------------------------------------------------------
HOVER_HEIGHT_OFFSET = 0.06   # meters above the measured target, first waypoint
APPROACH_OFFSET = 0.02       # meters "a little after" the target along the pan ray
DESCEND_Z = None             # meters -- if None, uses the triangulated z directly
PULLBACK_DISTANCE = 0.04     # meters, pulled back toward the robot base after descending
WRIST_PITCH = 0.0
WRIST_ROLL = 0.0
GRIPPER = 0.0


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


def wait_for_click(frame_rgb, camera_name):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)
    ax.set_title(f"'{camera_name}' -- click the weed, then close/confirm", fontsize=12)
    ax.axis("off")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise KeyboardInterrupt(f"No click recorded for camera '{camera_name}'.")
    return pts[0]


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


def triangulate_click(names, cams, transforms, intrinsics):
    rays = []
    for name in names:
        frame_rgb = capture_snapshot(cams[name])
        u, v = wait_for_click(frame_rgb, name)
        camera_matrix, dist_coeffs = intrinsics[name]
        ray_cam = undistort_point(u, v, camera_matrix, dist_coeffs)
        cam_to_robot = transforms[name]
        origin_robot = cam_to_robot[:3, 3]
        direction_robot = cam_to_robot[:3, :3] @ ray_cam
        rays.append((origin_robot, direction_robot))
    return triangulate_dlt(rays)


def run_removal_trajectory(robot, target_xyz):
    """hover -> approach offset -> descend -> pull-back -> retract."""
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
            "shoulder_pan": pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
            "wrist_flex": wrist_flex,
            "wrist_roll": WRIST_ROLL,
            "gripper": GRIPPER,
        })
        time.sleep(0.3)


def main():
    print("GB03 -- Classic click-to-control (no YOLO, no IL)")
    print("=" * 50)

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
            cmd = input("\nPress Enter to click+remove a weed ('q' to quit): ").strip().lower()
            if cmd == "q":
                break

            target_xyz = triangulate_click(names, cams, transforms, intrinsics)
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
