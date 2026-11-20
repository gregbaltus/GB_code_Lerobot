standard_pipeline/ -- which conda env runs which script
=========================================================

Two envs are needed here, split specifically at the ArUco-detection
boundary:

  lerobot env (existing, unchanged)
  ---------------------------------
  Has: robot control, depthai (OAK-D), placo/RobotKinematics, torch, etc.
  Does NOT have: cv2.aruco -- only plain opencv-python is installed,
  not opencv-contrib-python.

  charuco_calib env (new, created 2026-11-20)
  --------------------------------------------
  Has: opencv-contrib-python + numpy only (cv2.aruco works here).
  Does NOT have: depthai, placo, lerobot, torch -- no robot/camera
  hardware access, doesn't need it.

Why two envs instead of just installing opencv-contrib-python into
lerobot: the lerobot env's opencv build is a conda package
(opencv/libopencv/py-opencv) that `ultralytics` (the YOLO weed detector)
also depends on. A dry-run showed that removing/replacing it would
cascade into uninstalling ultralytics, scipy, pandas, and more -- so
swapping cv2 in place there was too risky. Splitting the hand-eye
calibration into a hardware-capture script and a pure-image-processing
solve script sidesteps that entirely: only the solve step needs
cv2.aruco, and it needs no robot/camera hardware, so it can live in a
small, separate, disk-cheap env instead.

Script -> env:
  GB01_charuco_capture_poses.py      -> lerobot env
      /home/gregory/miniconda3/envs/lerobot/bin/python GB01_charuco_capture_poses.py

  GB02_charuco_solve_calibration.py  -> charuco_calib env
      /home/gregory/miniconda3/envs/charuco_calib/bin/python GB02_charuco_solve_calibration.py

charuco_calib_common.py is shared (pure numpy, no cv2/depthai/lerobot
imports) so it works unchanged in both envs.

Any other script added to standard_pipeline/ that needs cv2.aruco
(ChArUco/ArUco detection) but not robot/camera hardware should also run
in charuco_calib env. Everything else in this project keeps using the
lerobot env as before.
