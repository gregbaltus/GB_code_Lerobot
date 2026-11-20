#!/usr/bin/env python3
"""
GB01_so101_charuco_calibrate.py

Hand-eye calibration: finds the fixed transform from one camera's frame to
the SO-101's own base frame, using a printed ChArUco board rigidly attached
near the wrist as the calibration target. Run once per camera.

Two modes, chosen per camera at the top of main():

  "direct" (video4 "right", video6 "left", and video2 "top" IF its angle
  check passes -- see GB_code_classic_method notes): the standard eye-to-hand
  hand-eye calibration. You teleop the arm (leader arm drives the follower,
  same as every other GB_code_* "reset phase") to a pose where the camera
  can see the board, press the RIGHT ARROW to capture that pose, repeat for
  ~15-20 poses (vary both position AND orientation -- rotation diversity is
  what makes cv2.calibrateHandEye() well-conditioned), then ESC to solve.
  LEFT ARROW discards the current attempt and retries the same pose.

  "chained" (fallback for video2 if it can't see the board at useful angles
  with the arm): no robot motion. Lay the printed board flat on the soil,
  visible to BOTH video2 and an already-calibrated reference camera (video4
  or video6) at once; press Enter to capture each simultaneous pair; repeat
  for a handful of placements, then solve. Requires that reference camera's
  saved cameraN_to_robot.npy to already exist.

Why this needs two solving stages, not one: cv2.calibrateHandEye() needs the
board's pose IN EACH CAMERA FRAME per capture, which needs that camera's
intrinsics (focal length, distortion) -- which we don't have yet. Rather than
a separate intrinsic-calibration script, this reuses the exact same captured
frames: cv2.aruco.calibrateCameraCharuco() solves intrinsics first from all
the captured board detections, then those intrinsics are used to solve each
frame's board pose via solvePnP, THEN hand-eye calibration runs on top.

Eye-to-hand vs eye-in-hand: cv2.calibrateHandEye() is written for a camera
mounted ON the arm (eye-in-hand). These are static/world-mounted cameras
watching the arm instead (eye-to-hand). Per OpenCV's own documentation, this
is handled by inverting the gripper2base transforms before passing them in
(base2gripper instead of gripper2base) -- the function then returns
cam2base instead of cam2gripper. Done explicitly below, not silently.

Output: camera{N}_to_robot.npy -- a 4x4 matrix mapping a point in that
camera's frame to the SO-101 base frame. Consumed by
GB02_so101_triangulate_test.py and onward.
"""

import math
import os
import time
import traceback

import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener, is_headless

# ---------------------------------------------------------------------------
# Robot / teleop -- same physical setup as GB_code_permabot_labo scripts
# ---------------------------------------------------------------------------
PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm_binnette"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID = "my_awesome_leader_arm"

FPS = 30
MAX_POSE_SECONDS = 120  # safety cap per teleop-to-pose window, not a real limit

# ---------------------------------------------------------------------------
# Cameras -- same names/indices as GB_code_permabot_labo/GB03_see_weeds_preprocess.py
# ---------------------------------------------------------------------------
CAMERAS = {
    "top": dict(index="/dev/video2", width=640, height=480, fps=30, fourcc="MJPG"),
    "right": dict(index="/dev/video4", width=640, height=480, fps=30, fourcc=None),
    "left": dict(index="/dev/video6", width=640, height=480, fps=30, fourcc=None),
}

# ---------------------------------------------------------------------------
# ChArUco board -- UPDATE these once a printed size is chosen from
# charuco_boards_multi_A4.pdf (currently set to the 50mm / 5x5 option).
# Must exactly match the printed board -- these are physical mm, not pixels.
# ---------------------------------------------------------------------------
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_M = 0.010   # 10mm
CHARUCO_MARKER_M = 0.007   # 7mm
CHARUCO_DICT = "DICT_5X5_50"

MIN_CORNERS_FOR_FRAME = 6  # discard a capture if fewer charuco corners were seen

# ---------------------------------------------------------------------------
# Kinematics -- URDF fetched from the official SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
# placo must also be installed (`pip install placo`) in this env.
#
# MUST be an absolute path -- placo resolves the URDF's relative mesh
# references (assets/*.stl) relative to the URDF path as given; a relative
# "so101_new_calib.urdf" fails with "Mesh ... could not be found" even when
# assets/ sits right next to it and cwd is this directory. Verified: FK at
# the zero pose returns a plausible pose and kinematics.joint_names comes
# back as exactly ["shoulder_pan", "shoulder_lift", "elbow_flex",
# "wrist_flex", "wrist_roll", "gripper"] -- matches this project's own
# joint names, so no name-order gotchas going into joint_pos_deg_from_obs().
# ---------------------------------------------------------------------------
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_new_calib.urdf")
# "gripper_frame_link" (NOT "gripper") -- the URDF defines this as a dummy
# link fixed to gripper_link via a FIXED joint, purpose-built as the tool
# reference frame. Verified empirically: target_frame_name="gripper" silently
# resolves to a DIFFERENT point ~7cm away (likely gripper_link's own origin,
# before the gripper_frame_joint offset). Both are rigid w.r.t. wrist_roll
# and unaffected by the gripper/jaw joint angle (confirmed: FK output is
# identical whether the jaw is closed or opened 80 degrees) -- so this
# wouldn't have broken calibration, but gripper_frame_link is the correct,
# intentional one.
TARGET_FRAME_NAME = "gripper_frame_link"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CHARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dictionary
    )
    return board


def joint_pos_deg_from_obs(obs, kinematics):
    """Build the FK input array by NAME, in kinematics.joint_names order --
    never assume positional order matches this project's own JOINT_NAMES,
    since that depends on the URDF's own joint naming."""
    try:
        values = [obs[f"{name}.pos"] for name in kinematics.joint_names]
    except KeyError as e:
        raise RuntimeError(
            f"URDF joint name {e} not found in robot observation. "
            f"kinematics.joint_names={kinematics.joint_names} vs "
            f"this project's joint names={JOINT_NAMES}. Check the URDF."
        ) from e
    return np.array(values, dtype=np.float64)


def detect_charuco(frame_rgb, board, detector):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < MIN_CORNERS_FOR_FRAME:
        return None
    return charuco_corners, charuco_ids


def solve_intrinsics(charuco_corners_list, charuco_ids_list, board, image_size):
    print(f"Solving intrinsics from {len(charuco_corners_list)} captured frames ...")
    reproj_err, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
        charuco_corners_list, charuco_ids_list, board, image_size, None, None
    )
    print(f"  Intrinsics reprojection error: {reproj_err:.4f} px")
    return camera_matrix, dist_coeffs


def board_pose_in_camera(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs):
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def rt_to_4x4(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_4x4(T):
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Direct mode: robot FK poses + board-in-camera poses -> cv2.calibrateHandEye()
# ---------------------------------------------------------------------------
def run_direct_capture(robot, teleop, camera_name, kinematics, board, detector):
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    gripper2base_list = []
    charuco_corners_list = []
    charuco_ids_list = []
    image_size = None

    listener, events = init_keyboard_listener()
    print(
        "\nTeleop the arm so the board is visible to this camera.\n"
        "  RIGHT ARROW = capture this pose\n"
        "  LEFT ARROW  = discard, retry same pose\n"
        "  ESC         = done capturing, solve calibration\n"
    )

    try:
        while not events["stop_recording"]:
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=MAX_POSE_SECONDS,
                display_data=False,
            )

            if events["stop_recording"]:
                break

            if events["rerecord_episode"]:
                print("  Discarded -- retry this pose.")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                continue

            # RIGHT ARROW: capture
            events["exit_early"] = False
            obs = robot.get_observation()
            frame_rgb = obs[camera_name]
            if image_size is None:
                image_size = (frame_rgb.shape[1], frame_rgb.shape[0])

            detection = detect_charuco(frame_rgb, board, detector)
            if detection is None:
                print(f"  Board not detected (need >= {MIN_CORNERS_FOR_FRAME} corners) -- pose NOT recorded.")
                continue

            charuco_corners, charuco_ids = detection
            joint_pos_deg = joint_pos_deg_from_obs(obs, kinematics)
            gripper2base = kinematics.forward_kinematics(joint_pos_deg)

            gripper2base_list.append(gripper2base)
            charuco_corners_list.append(charuco_corners)
            charuco_ids_list.append(charuco_ids)
            print(f"  Captured pose {len(gripper2base_list)} "
                  f"({len(charuco_corners)} corners detected).")
    finally:
        if not is_headless() and listener:
            listener.stop()

    if len(gripper2base_list) < 6:
        raise RuntimeError(
            f"Only {len(gripper2base_list)} poses captured -- need at least ~6, "
            "ideally 15-20 with varied position AND orientation. Re-run and capture more."
        )

    camera_matrix, dist_coeffs = solve_intrinsics(charuco_corners_list, charuco_ids_list, board, image_size)

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []
    for gripper2base, corners, ids in zip(gripper2base_list, charuco_corners_list, charuco_ids_list):
        pose = board_pose_in_camera(corners, ids, board, camera_matrix, dist_coeffs)
        if pose is None:
            continue
        R_tc, t_tc = pose
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)

        # Eye-to-hand trick: invert gripper2base -> base2gripper before feeding
        # calibrateHandEye(), so the result is cam2base instead of cam2gripper.
        base2gripper = invert_4x4(gripper2base)
        R_gripper2base.append(base2gripper[:3, :3])
        t_gripper2base.append(base2gripper[:3, 3])

    print(f"Solving hand-eye calibration from {len(R_gripper2base)} usable poses ...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    )
    camera_to_robot = rt_to_4x4(R_cam2base, t_cam2base.reshape(3))
    return camera_to_robot, camera_matrix, dist_coeffs


# ---------------------------------------------------------------------------
# Chained mode: fallback when a camera can't directly see the board on the
# arm at a useful angle. No robot motion -- board held/placed statically,
# visible to this camera AND an already-calibrated reference camera at once.
# ---------------------------------------------------------------------------
def run_chained_capture(robot, camera_name, reference_camera_name, board, detector):
    ref_path = f"camera_{reference_camera_name}_to_robot.npy"
    if not os.path.exists(ref_path):
        raise RuntimeError(f"Reference camera transform {ref_path} not found -- calibrate it first (direct mode).")
    ref_to_robot = np.load(ref_path)

    print(
        f"\nHold the board flat and static where BOTH '{camera_name}' and "
        f"'{reference_camera_name}' can see it. Press Enter to capture each "
        f"placement, 'q' when done (need at least 3-4 placements, varied position).\n"
    )

    cam_corners_list, cam_ids_list = [], []
    ref_corners_list, ref_ids_list = [], []
    cam_image_size = ref_image_size = None

    while True:
        cmd = input("Capture placement> ").strip().lower()
        if cmd == "q":
            break

        obs = robot.get_observation()
        cam_frame = obs[camera_name]
        ref_frame = obs[reference_camera_name]
        if cam_image_size is None:
            cam_image_size = (cam_frame.shape[1], cam_frame.shape[0])
            ref_image_size = (ref_frame.shape[1], ref_frame.shape[0])

        cam_det = detect_charuco(cam_frame, board, detector)
        ref_det = detect_charuco(ref_frame, board, detector)
        if cam_det is None or ref_det is None:
            print("  Board not detected in both views -- placement NOT recorded.")
            continue

        cam_corners_list.append(cam_det[0]); cam_ids_list.append(cam_det[1])
        ref_corners_list.append(ref_det[0]); ref_ids_list.append(ref_det[1])
        print(f"  Captured placement {len(cam_corners_list)}.")

    if len(cam_corners_list) < 3:
        raise RuntimeError(f"Only {len(cam_corners_list)} placements captured -- need at least 3-4.")

    cam_matrix, cam_dist = solve_intrinsics(cam_corners_list, cam_ids_list, board, cam_image_size)
    ref_matrix, ref_dist = solve_intrinsics(ref_corners_list, ref_ids_list, board, ref_image_size)

    candidates = []
    for cam_c, cam_i, ref_c, ref_i in zip(cam_corners_list, cam_ids_list, ref_corners_list, ref_ids_list):
        cam_pose = board_pose_in_camera(cam_c, cam_i, board, cam_matrix, cam_dist)
        ref_pose = board_pose_in_camera(ref_c, ref_i, board, ref_matrix, ref_dist)
        if cam_pose is None or ref_pose is None:
            continue
        cam_from_board = rt_to_4x4(*cam_pose)
        ref_from_board = rt_to_4x4(*ref_pose)
        # ref_from_cam = ref_from_board @ inv(cam_from_board)
        ref_from_cam = ref_from_board @ invert_4x4(cam_from_board)
        camera_to_robot_k = ref_to_robot @ ref_from_cam
        candidates.append(camera_to_robot_k)

    if not candidates:
        raise RuntimeError("No valid simultaneous detections -- could not solve.")

    translations = np.stack([T[:3, 3] for T in candidates])
    spread = np.linalg.norm(translations - translations.mean(axis=0), axis=1)
    print(f"  {len(candidates)} candidate transforms, translation spread "
          f"mean={spread.mean() * 1000:.1f}mm max={spread.max() * 1000:.1f}mm "
          f"(large spread = unreliable, re-capture with more varied placements).")

    # Simple average (translation: arithmetic mean; rotation: mean via SVD
    # orthogonalization) -- adequate for a fallback path, not the primary method.
    R_mean_raw = np.mean([T[:3, :3] for T in candidates], axis=0)
    U, _, Vt = np.linalg.svd(R_mean_raw)
    R_mean = U @ Vt
    t_mean = translations.mean(axis=0)
    return rt_to_4x4(R_mean, t_mean), cam_matrix, cam_dist


def main():
    print("GB01 -- SO-101 ChArUco Hand-Eye Calibration")
    print("=" * 50)

    if not os.path.exists(URDF_PATH):
        print(f"URDF not found at '{URDF_PATH}'.")
        print("Download it from: https://github.com/TheRobotStudio/SO-ARM100/"
              "blob/main/Simulation/SO101/so101_new_calib.urdf")
        return

    try:
        kinematics = RobotKinematics(URDF_PATH, TARGET_FRAME_NAME)
    except ImportError as e:
        print(f"{e}\nInstall with: pip install placo")
        return
    print(f"URDF joint names (FK input order): {kinematics.joint_names}")
    print(f"This project's joint names:        {JOINT_NAMES}")
    if set(kinematics.joint_names) - {"gripper"} != set(JOINT_NAMES) - {"gripper"}:
        print("  WARNING: joint name sets differ -- double check before trusting FK output.\n")

    cam_name = input(f"Camera to calibrate {list(CAMERAS.keys())}: ").strip()
    if cam_name not in CAMERAS:
        print(f"Unknown camera '{cam_name}'.")
        return

    mode = input("Mode [direct/chained] (default: direct): ").strip().lower() or "direct"

    cam_cfg = CAMERAS[cam_name]
    kwargs = dict(index_or_path=cam_cfg["index"], width=cam_cfg["width"],
                  height=cam_cfg["height"], fps=cam_cfg["fps"])
    if cam_cfg["fourcc"]:
        kwargs["fourcc"] = cam_cfg["fourcc"]
    cameras = {cam_name: OpenCVCameraConfig(**kwargs)}

    if mode == "chained":
        ref_name = input(f"Reference camera (already calibrated) {list(CAMERAS.keys())}: ").strip()
        ref_cfg = CAMERAS[ref_name]
        ref_kwargs = dict(index_or_path=ref_cfg["index"], width=ref_cfg["width"],
                           height=ref_cfg["height"], fps=ref_cfg["fps"])
        if ref_cfg["fourcc"]:
            ref_kwargs["fourcc"] = ref_cfg["fourcc"]
        cameras[ref_name] = OpenCVCameraConfig(**ref_kwargs)

    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras=cameras)
    robot = SO101Follower(robot_config)

    teleop = None
    if mode == "direct":
        teleop_config = SO101LeaderConfig(port=TELEOP_PORT, id=TELEOP_ID)
        teleop = SO101Leader(teleop_config)

    board = build_board()
    detector = cv2.aruco.CharucoDetector(board)

    try:
        robot.connect()
        if teleop:
            teleop.connect()
        print("Connected.")

        if mode == "direct":
            camera_to_robot, camera_matrix, dist_coeffs = run_direct_capture(
                robot, teleop, cam_name, kinematics, board, detector
            )
        elif mode == "chained":
            camera_to_robot, camera_matrix, dist_coeffs = run_chained_capture(
                robot, cam_name, ref_name, board, detector
            )
        else:
            print(f"Unknown mode '{mode}'.")
            return

        out_path = f"camera_{cam_name}_to_robot.npy"
        np.save(out_path, camera_to_robot)
        np.save(f"camera_{cam_name}_intrinsics.npy",
                {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs})
        print(f"\nSaved {out_path}")
        print(camera_to_robot)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("\nDisconnecting ...")
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
