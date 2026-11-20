"""
charuco_calib_common.py

Tiny shared helpers for the OAK-D hand-eye calibration pair
(GB01_charuco_capture_poses.py + GB02_charuco_solve_calibration.py).

Pure numpy, no cv2/depthai/lerobot imports -- this file is the only thing
both scripts share, since they run in two different conda envs (see
README-env.txt) that don't agree on which cv2 build is installed.
"""

import numpy as np


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
