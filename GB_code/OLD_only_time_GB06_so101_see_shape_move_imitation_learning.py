#!/usr/bin/env python3
"""
GB06_so101_see_shape_move_imitation_learning.py

Same as GB05_so101_see_shape_and_move.py (HSV colour + shape detection,
move the wrist over the chosen piece using the GB02 calibration homography),
but once the arm is hovering over the piece it hands off to a trained ACT
policy (act_CameraOnGripperColoredPiece) for a fixed duration to attempt the
actual grasp — no dataset recording, no leader arm.

Only "yellow hexagon" is supported by the grasp policy for now; other pieces
are still detected and hovered over, but the imitation-learning phase is
skipped with a message.

Requires GB02_so101_2Dcalibration.py to have been run first.
"""

import math
import random
import time
import traceback
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

# ---------------------------------------------------------------------------
# Safe ranges
# ---------------------------------------------------------------------------
SAFE_X          = (0.05,   0.23)
SAFE_PAN        = (-90.0,  110.0)
SAFE_Y          = (-0.08,  0.20)

CALIBRATION_FILE         = "so101_calibration.npy"
DETECTION_DISPLAY_SECONDS = 3.0
MIN_AREA                 = 400    # pixels²

# P-controller parameters (approach phase)
KP             = 0.5
CONTROL_FREQ   = 50
CONVERGENCE_TH = 8.0
TIMEOUT        = 5.0

# Fixed wrist parameters for the approach pose (no longer prompted)
FIXED_PITCH      = 90.0   # degrees
FIXED_WRIST_ROLL = 0.0    # degrees
FIXED_GRIPPER    = 0.0    # 0 = closed

# ---------------------------------------------------------------------------
# Imitation-learning grasp phase
# ---------------------------------------------------------------------------
POLICY_PATH      = "act_CameraOnGripperColoredPiece"
TRAIN_DATASET_ID = "Greynar/CameraOnGripperColoredPiece"  # read-only: stats/feature shapes only, nothing recorded
GRASP_TASK       = "Grab the yellow hexagone"

GRASP_CAMERA_NAME     = "onGripper"
# Device index/path is picked interactively at startup (see main()) — USB
# enumeration order means /dev/videoN can shift between reboots/reconnects.
GRASP_CAMERA_WIDTH    = 480
GRASP_CAMERA_HEIGHT   = 640
GRASP_CAMERA_FPS      = 30
GRASP_CAMERA_ROTATION = Cv2Rotation.ROTATE_270   # -90 degrees — must match training

GRASP_DURATION_S = 20.0   # fixed duration for now — tune after a few tests

SUPPORTED_GRASP_PIECES = {"yellow hexagon"}


# ---------------------------------------------------------------------------
# Shape checks (same thresholds as displaySeeForm.py / GB05)
# ---------------------------------------------------------------------------
def _is_circle(cnt):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    return perim > 0 and (4 * np.pi * area / perim ** 2) > 0.70

def _is_star(cnt):
    area      = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return hull_area > 0 and (area / hull_area) < 0.82

def _is_triangle(cnt):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    return len(approx) == 3

def _is_hexagon(cnt):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    return 5 <= len(approx) <= 8


# ---------------------------------------------------------------------------
# Piece definitions
# ---------------------------------------------------------------------------
PIECES = [
    {
        "label":       "orange circle",
        "hsv_ranges":  [((3, 130, 130), (22, 255, 255))],
        "draw_color":  (0, 140, 255),
        "shape_check": _is_circle,
    },
    {
        "label":       "green star",
        "hsv_ranges":  [((72, 120, 60), (97, 255, 190))],
        "draw_color":  (0, 180, 0),
        "shape_check": _is_star,
    },
    {
        "label":       "red triangle",
        "hsv_ranges":  [((158, 120, 80), (179, 255, 230))],
        "draw_color":  (0, 0, 220),
        "shape_check": _is_triangle,
    },
    {
        "label":       "yellow hexagon",
        "hsv_ranges":  [((15, 60, 80), (38, 255, 255))],
        "draw_color":  (0, 210, 230),
        "shape_check": _is_hexagon,
    },
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def build_mask(hsv, hsv_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def find_all_instances(hsv, piece):
    """Return a list of all detections for this piece (colour + shape match)."""
    mask = build_mask(hsv, piece["hsv_ranges"])

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        if piece["shape_check"] and not piece["shape_check"](cnt):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        results.append({"center": (cx, cy), "bbox": (x, y, w, h)})

    return results


def match_piece(name):
    """Return the first PIECES entry whose label contains `name` (case-insensitive)."""
    name_lower = name.strip().lower()
    for p in PIECES:
        if name_lower in p["label"].lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def show_detection(frame_bgr, piece, all_instances, chosen, display_seconds):
    """
    Show all detected instances (white box) and highlight the chosen one
    (coloured box + crosshair). Auto-closes after display_seconds.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    for inst in all_instances:
        x, y, w, h = inst["bbox"]
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=1, edgecolor='white', facecolor='none',
                     linestyle='--'))

    cx, cy     = chosen["center"]
    x, y, w, h = chosen["bbox"]
    color_hex  = "#{:02x}{:02x}{:02x}".format(*piece["draw_color"][::-1])  # BGR→RGB
    ax.add_patch(patches.Rectangle((x, y), w, h,
                 linewidth=2, edgecolor=color_hex, facecolor='none'))
    ax.plot(cx, cy, '+', color=color_hex, markersize=18, markeredgewidth=2)
    ax.text(x, y - 8, piece["label"],
            color=color_hex, fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='none'))

    n = len(all_instances)
    title = f"Found {n}× {piece['label']}"
    if n > 1:
        title += "  — one chosen at random"
    title += f"  — moving in {display_seconds:.0f}s …"
    ax.set_title(title, fontsize=13)
    ax.axis('off')

    plt.draw()
    plt.pause(display_seconds)
    plt.close(fig)


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
# Camera + homography
# ---------------------------------------------------------------------------
def capture_snapshot(cap, warmup=5):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def pixel_to_robot(u, v, H):
    pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    return float(pt[0][0][0]), float(pt[0][0][1])


# ---------------------------------------------------------------------------
# Imitation-learning grasp phase
# ---------------------------------------------------------------------------
def load_grasp_policy():
    """Load the ACT policy + processors once, up front. No dataset is created —
    TRAIN_DATASET_ID is only used to read normalization stats / feature shapes."""
    print(f"Loading grasp policy '{POLICY_PATH}' …")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = POLICY_PATH

    ds_meta = LeRobotDatasetMetadata(TRAIN_DATASET_ID)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )
    print("Grasp policy ready.\n")
    return policy, preprocessor, postprocessor


def run_grasp_policy(robot, policy, preprocessor, postprocessor, robot_action_processor,
                      robot_observation_processor, dataset_features,
                      duration_s=GRASP_DURATION_S, fps=GRASP_CAMERA_FPS, task=GRASP_TASK):
    """Run the ACT policy for a fixed duration, then stop. No dataset recording."""
    policy.reset()

    dt    = 1.0 / fps
    steps = int(duration_s * fps)
    print(f"  Running grasp policy for {duration_s:.1f}s ({steps} steps) …")

    for _ in range(steps):
        obs           = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        obs_frame     = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        action_values = predict_action(
            observation=obs_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        act_processed = make_robot_action(action_values, dataset_features)
        robot_action  = robot_action_processor((act_processed, obs))
        robot.send_action(robot_action)

        time.sleep(dt)

    print("  Grasp attempt finished (fixed duration elapsed).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SO-101 See-Shape-Move-and-Grasp (imitation learning)  —  GB06")
    print("=" * 60)

    try:
        H = np.load(CALIBRATION_FILE)
        print(f"Loaded calibration: {CALIBRATION_FILE}")
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

    print("\nFixed arm parameter (y should match calibration height):")
    y_fixed = ask_float("y — height (m)", *SAFE_Y, 0.0)
    print(f"Using pitch={FIXED_PITCH}°  wrist_roll={FIXED_WRIST_ROLL}°  "
          f"gripper={FIXED_GRIPPER} (fixed, not prompted)\n")

    # Detection camera (GB05 part)
    available = [i for i in range(5) if cv2.VideoCapture(i).isOpened()]
    if not available:
        print("No cameras found.")
        return
    print(f"Available cameras: {available}")
    cam_idx = int(input("Select detection camera index (GB05 piece-finding camera): ").strip())
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Grasp / imitation-learning camera — must be the SAME physical camera
    # (onGripper) used when recording the training dataset for POLICY_PATH,
    # so the images match what the policy was trained on. Resolution, fps
    # and rotation below are already fixed to match training; only the
    # device index can vary machine-to-machine / reboot-to-reboot.
    print(f"\nSelect the camera used for the imitation-learning grasp phase.")
    print(f"This MUST be the same physical camera used to record the training "
          f"dataset for '{POLICY_PATH}' (mounted on the gripper), matching the "
          f"trained resolution ({GRASP_CAMERA_WIDTH}x{GRASP_CAMERA_HEIGHT}), "
          f"fps ({GRASP_CAMERA_FPS}) and rotation ({GRASP_CAMERA_ROTATION.name}).")
    grasp_available = [i for i in available if i != cam_idx]
    if not grasp_available:
        print("No other camera available for the grasp phase.")
        return
    print(f"Available cameras: {grasp_available}")
    grasp_cam_idx = int(input("Select imitation-learning (onGripper) camera index: ").strip())
    if grasp_cam_idx == cam_idx:
        print("  ! Warning: same index as the detection camera — this is probably wrong.")

    # Robot — owns the onGripper camera used by the grasp policy
    robot_config = SO101FollowerConfig(
        port=port,
        id="my_awesome_follower_arm",
        cameras={
            GRASP_CAMERA_NAME: OpenCVCameraConfig(
                index_or_path=grasp_cam_idx,
                width=GRASP_CAMERA_WIDTH,
                height=GRASP_CAMERA_HEIGHT,
                fps=GRASP_CAMERA_FPS,
                rotation=GRASP_CAMERA_ROTATION,
            )
        },
    )
    robot = SO101Follower(robot_config)

    # Load the grasp policy once, up front
    policy, preprocessor, postprocessor = load_grasp_policy()
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
        ),
    )

    known = "  |  ".join(p["label"] for p in PIECES)
    print(f"Known pieces: {known}")
    print(f"Grasp (imitation learning) supported for: {', '.join(SUPPORTED_GRASP_PIECES)}")
    print("Enter 'initial' to return to zero position.")
    print("Partial names accepted (e.g. 'star', 'orange'). Type 'q' to quit.\n")

    try:
        robot.connect()
        print("Connected.")
        move_to_zero(robot)

        while True:
            name = input("Piece to find: ").strip()
            if not name or name.lower() == 'q':
                break

            if name.lower() == "initial":
                move_to_zero(robot)
                continue

            piece = match_piece(name)
            if piece is None:
                print(f"  Unknown piece '{name}'. Known: {known}\n")
                continue

            print(f"  Looking for '{piece['label']}' …")
            frame_bgr = capture_snapshot(cap)
            hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            instances = find_all_instances(hsv, piece)

            if not instances:
                print(f"  '{piece['label']}' not detected. Try again.\n")
                continue

            chosen       = random.choice(instances)
            u, v         = chosen["center"]
            x_robot, pan = pixel_to_robot(u, v, H)

            print(f"  Found {len(instances)}× '{piece['label']}'"
                  + (" — chosen at random" if len(instances) > 1 else ""))
            print(f"  Pixel=({u}, {v})   x={x_robot:.3f} m   pan={pan:.1f}°")

            if not (SAFE_X[0] <= x_robot <= SAFE_X[1]):
                print(f"  ! x={x_robot:.3f} m out of safe range {SAFE_X} — skipping.\n")
                continue
            if not (SAFE_PAN[0] <= pan <= SAFE_PAN[1]):
                print(f"  ! pan={pan:.1f}° out of safe range {SAFE_PAN} — skipping.\n")
                continue

            show_detection(frame_bgr, piece, instances, chosen,
                           DETECTION_DISPLAY_SECONDS)

            shoulder_lift, elbow_flex = inverse_kinematics(x_robot, y_fixed)
            wrist_flex = -shoulder_lift - elbow_flex + FIXED_PITCH

            print("  Moving over piece …")
            p_control_to_target(robot, {
                "shoulder_pan":  pan,
                "shoulder_lift": shoulder_lift,
                "elbow_flex":    elbow_flex,
                "wrist_flex":    wrist_flex,
                "wrist_roll":    FIXED_WRIST_ROLL,
                "gripper":       FIXED_GRIPPER,
            })

            if piece["label"] in SUPPORTED_GRASP_PIECES:
                run_grasp_policy(robot, policy, preprocessor, postprocessor,
                                  robot_action_processor, robot_observation_processor,
                                  dataset_features)
            else:
                print(f"  Imitation learning not supported yet for '{piece['label']}'.")

            move_to_zero(robot)
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
