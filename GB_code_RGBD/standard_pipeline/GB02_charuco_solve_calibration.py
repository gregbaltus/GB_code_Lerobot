#!/usr/bin/env python3
"""
GB02_charuco_solve_calibration.py

Second half of OAK-D hand-eye calibration (eye-to-hand): loads the
frame_NNN.png + poses.npy captured by GB01_charuco_capture_poses.py,
detects the ChArUco board in each frame, solves camera intrinsics and
per-frame board pose, then solves cv2.calibrateHandEye() to get OAK-D's
fixed pose relative to the robot base.

Runs in the separate `charuco_calib` env (opencv-contrib-python + numpy
only, no robot/depthai/torch) -- see README-env.txt for why.

Eye-to-hand vs eye-in-hand: cv2.calibrateHandEye() is written for a camera
mounted ON the arm (eye-in-hand). OAK-D is static/world-mounted instead
(eye-to-hand). Per OpenCV's own documentation, this is handled by
inverting the gripper2base transforms before passing them in
(base2gripper instead of gripper2base) -- the function then returns
cam2base instead of cam2gripper. Done explicitly below, not silently.

Output: oakd_to_robot.npy -- a 4x4 matrix mapping a point in OAK-D's RGB
camera frame to the SO-101 base frame. oakd_intrinsics.npy -- camera
matrix + distortion coefficients (needed to turn a pixel+depth reading
into a camera-frame 3D point before applying this transform).

Run with: /home/gregory/miniconda3/envs/charuco_calib/bin/python GB02_charuco_solve_calibration.py
"""

import glob
import os

import cv2
import numpy as np

from charuco_calib_common import invert_4x4, rt_to_4x4

# ---------------------------------------------------------------------------
# ChArUco board -- UPDATE these to match the printed board before running.
# OAK-D sits ~75cm above the board (much farther than the old wrist-mounted-
# board setups this script's logic is adapted from), so a small board
# (e.g. 10mm squares) may not be reliably detectable at that distance --
# pick a larger square size than you would for a close-up camera, and
# confirm with a few test captures before doing the full 15-20 pose run.
# ---------------------------------------------------------------------------
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_M = 0.030   # 30mm -- placeholder, confirm against your printed board
CHARUCO_MARKER_M = 0.022   # 22mm -- placeholder, confirm against your printed board
CHARUCO_DICT = "DICT_5X5_50"

MIN_CORNERS_FOR_FRAME = 6  # discard a capture if fewer charuco corners were seen

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charuco_capture")


def build_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CHARUCO_DICT))
    return cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y), CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dictionary
    )


def detect_charuco(frame_bgr, board, detector):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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


def main():
    print("GB02 -- OAK-D ChArUco calibration: solve")
    print("=" * 50)

    poses_path = os.path.join(CAPTURE_DIR, "poses.npy")
    if not os.path.exists(poses_path):
        print(f"{poses_path} not found -- run GB01_charuco_capture_poses.py first "
              f"(in the lerobot env).")
        return
    gripper2base_list = np.load(poses_path)

    frame_paths = sorted(glob.glob(os.path.join(CAPTURE_DIR, "frame_*.png")))
    if len(frame_paths) != len(gripper2base_list):
        print(f"Mismatch: {len(frame_paths)} frames vs {len(gripper2base_list)} poses "
              f"in {CAPTURE_DIR} -- capture data looks corrupted/partial.")
        return
    if len(frame_paths) < 6:
        print(f"Only {len(frame_paths)} poses captured -- need at least ~6, ideally 15-20. "
              f"Re-run GB01 and capture more.")
        return

    board = build_board()
    detector = cv2.aruco.CharucoDetector(board)

    charuco_corners_list, charuco_ids_list = [], []
    kept_gripper2base = []
    image_size = None

    for frame_path, gripper2base in zip(frame_paths, gripper2base_list):
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            print(f"  Could not read {frame_path} -- skipping.")
            continue
        if image_size is None:
            image_size = (frame_bgr.shape[1], frame_bgr.shape[0])

        detection = detect_charuco(frame_bgr, board, detector)
        if detection is None:
            print(f"  Board not detected in {os.path.basename(frame_path)} "
                  f"(need >= {MIN_CORNERS_FOR_FRAME} corners) -- skipping.")
            continue

        charuco_corners, charuco_ids = detection
        charuco_corners_list.append(charuco_corners)
        charuco_ids_list.append(charuco_ids)
        kept_gripper2base.append(gripper2base)
        print(f"  {os.path.basename(frame_path)}: {len(charuco_corners)} corners detected.")

    if len(kept_gripper2base) < 6:
        print(f"\nOnly {len(kept_gripper2base)} usable frames (board detected) -- need at "
              f"least ~6. Re-run GB01, check board size/visibility, and capture more.")
        return

    camera_matrix, dist_coeffs = solve_intrinsics(charuco_corners_list, charuco_ids_list, board, image_size)

    R_base2gripper, t_base2gripper = [], []
    R_target2cam, t_target2cam = [], []
    for gripper2base, corners, ids in zip(kept_gripper2base, charuco_corners_list, charuco_ids_list):
        pose = board_pose_in_camera(corners, ids, board, camera_matrix, dist_coeffs)
        if pose is None:
            continue
        R_tc, t_tc = pose
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)

        # Eye-to-hand trick: invert gripper2base -> base2gripper before feeding
        # calibrateHandEye(), so the result is cam2base instead of cam2gripper.
        base2gripper = invert_4x4(gripper2base)
        R_base2gripper.append(base2gripper[:3, :3])
        t_base2gripper.append(base2gripper[:3, 3])

    print(f"\nSolving hand-eye calibration from {len(R_base2gripper)} usable poses ...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, R_target2cam, t_target2cam
    )
    oakd_to_robot = rt_to_4x4(R_cam2base, t_cam2base.reshape(3))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oakd_to_robot.npy")
    np.save(out_path, oakd_to_robot)
    np.save(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "oakd_intrinsics.npy"),
        {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
    )
    print(f"\nSaved {out_path}")
    print(oakd_to_robot)


if __name__ == "__main__":
    main()
