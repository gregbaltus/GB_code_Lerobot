#!/usr/bin/env python3
"""
GB01_charuco_capture_poses.py

First half of OAK-D hand-eye calibration (eye-to-hand): OAK-D is static,
overhead, ~75cm over the dirt; a printed ChArUco board is mounted rigidly
on the gripper (same mount point as the pre-pivot plan's wrist-mounted
board -- see GB_code_classic_method/charuco_mount_location.png). This
script only CAPTURES data -- it does no ArUco detection itself. That's
deliberate: see README-env.txt in this folder for why the ArUco solving
step (GB02_charuco_solve_calibration.py) has to run in a different conda
env than this one, and splitting the script in two at exactly this
boundary (capture vs. solve) is what makes that possible.

For each captured pose, saves:
  - charuco_capture/frame_NNN.png       -- the OAK-D RGB frame at that pose
  - charuco_capture/poses.npy           -- stacked (N, 4, 4) gripper2base
                                            FK matrices, same order as the
                                            frame_NNN.png files

Procedure: teleop the arm (leader -> follower, same as every other
GB_code_* teleop script) so the board is visible to OAK-D, varying BOTH
position and orientation across captures -- rotation diversity is what
makes cv2.calibrateHandEye() well-conditioned.

  RIGHT ARROW = capture this pose (saves frame + FK pose)
  LEFT ARROW  = discard, keep teleoperating (nothing saved)
  ESC         = done capturing

Need at least ~6 poses, ideally 15-20, before running GB02.

Run with: /home/gregory/miniconda3/envs/lerobot/bin/python GB01_charuco_capture_poses.py
"""

import os
import time
import traceback

import cv2
import depthai as dai
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so_leader.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener, is_headless

# ---------------------------------------------------------------------------
# Robot / teleop -- same ports as every other GB_code_RGBD script
# ---------------------------------------------------------------------------
PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm_binnette"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID = "my_awesome_leader_arm"

FPS = 30

# ---------------------------------------------------------------------------
# OAK-D RGB only -- no depth/stereo needed here, this script never touches it
# ---------------------------------------------------------------------------
RGB_SIZE = (1280, 800)  # matches GB02_oakd_two_camera_teleop.py -- keep in sync
OAKD_FPS = 15

# ---------------------------------------------------------------------------
# Kinematics -- same URDF already fetched for the pre-pivot classic-method
# plan; reused in place rather than duplicated (robot kinematics aren't
# pipeline-specific). MUST be an absolute path -- placo resolves the URDF's
# relative mesh references against the path as given.
# ---------------------------------------------------------------------------
URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "GB_code_classic_method", "so101_new_calib.urdf",
)
URDF_PATH = os.path.normpath(URDF_PATH)
# "gripper_frame_link" (NOT "gripper") -- the URDF's purpose-built tool
# reference frame, rigid w.r.t. the gripper/jaw joint. See
# GB_code_classic_method/GB01_so101_charuco_calibrate.py for the full
# reasoning (verified empirically there, not re-derived here).
TARGET_FRAME_NAME = "gripper_frame_link"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charuco_capture")


def joint_pos_deg_from_obs(obs, kinematics):
    """Build the FK input array by NAME, in kinematics.joint_names order --
    never assume positional order matches this project's own joint names."""
    try:
        values = [obs[f"{name}.pos"] for name in kinematics.joint_names]
    except KeyError as e:
        raise RuntimeError(
            f"URDF joint name {e} not found in robot observation. "
            f"kinematics.joint_names={kinematics.joint_names}. Check the URDF."
        ) from e
    return np.array(values, dtype=np.float64)


def main():
    print("GB01 -- OAK-D ChArUco calibration: pose capture")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with dai.Pipeline(dai.Device()) as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"OAK-D connected: {device.getDeviceName()}  USB speed: {device.getUsbSpeed().name}")

        rgb_cam = pipeline.create(dai.node.Camera)
        rgb_cam.setSensorType(dai.CameraSensorType.COLOR)
        rgb_cam.build(dai.CameraBoardSocket.CAM_A, sensorFps=OAKD_FPS)
        cap = dai.ImgFrameCapability()
        cap.size.fixed(RGB_SIZE)
        cap.fps.fixed(OAKD_FPS)
        rgb_queue = rgb_cam.requestOutput(cap, True).createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print("OAK-D pipeline running.")

        robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras={})
        robot = SO101Follower(robot_config)
        teleop_config = SO101LeaderConfig(port=TELEOP_PORT, id=TELEOP_ID)
        teleop = SO101Leader(teleop_config)

        teleop_action_processor, robot_action_processor, _robot_observation_processor = make_default_processors()

        last_rgb = None
        gripper2base_list = []
        listener = None

        try:
            robot.connect()
            teleop.connect()
            print("Robot + teleop connected.")

            listener, events = init_keyboard_listener()
            print(
                "\nTeleop the arm so the board is visible to OAK-D.\n"
                "  RIGHT ARROW = capture this pose\n"
                "  LEFT ARROW  = discard, keep teleoperating\n"
                "  ESC         = done capturing\n"
            )

            dt = 1.0 / FPS
            while not events["stop_recording"]:
                start = time.perf_counter()

                obs = robot.get_observation()

                rgb_pkt = rgb_queue.tryGet()
                if rgb_pkt is not None:
                    last_rgb = cv2.cvtColor(rgb_pkt.getCvFrame(), cv2.COLOR_BGR2RGB)

                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
                robot.send_action(robot_action_to_send)

                if events["exit_early"]:
                    if events["stop_recording"]:
                        break
                    elif events["rerecord_episode"]:
                        print("  Discarded -- keep teleoperating.")
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                    else:
                        # Right arrow: capture.
                        if last_rgb is None:
                            print("  No OAK-D frame received yet -- not captured, try again.")
                        else:
                            joint_pos_deg = joint_pos_deg_from_obs(obs, kinematics)
                            gripper2base = kinematics.forward_kinematics(joint_pos_deg)
                            idx = len(gripper2base_list)
                            frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:03d}.png")
                            cv2.imwrite(frame_path, cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
                            gripper2base_list.append(gripper2base)
                            print(f"  Captured pose {idx + 1} -> {frame_path}")
                        events["exit_early"] = False

                elapsed = time.perf_counter() - start
                time.sleep(max(dt - elapsed, 0.0))

        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            print("\nDisconnecting ...")
            if robot.is_connected:
                robot.disconnect()
            if teleop.is_connected:
                teleop.disconnect()
            if not is_headless() and listener:
                listener.stop()
            pipeline.stop()

            if gripper2base_list:
                poses_path = os.path.join(OUTPUT_DIR, "poses.npy")
                np.save(poses_path, np.stack(gripper2base_list))
                print(f"\nSaved {len(gripper2base_list)} poses to {poses_path}")
                if len(gripper2base_list) < 6:
                    print("WARNING: fewer than 6 poses -- calibrateHandEye() needs at least "
                          "~6, ideally 15-20 with varied position AND orientation.")
                print(f"Next: run GB02_charuco_solve_calibration.py "
                      f"(in the charuco_calib env -- see README-env.txt).")
            else:
                print("\nNo poses captured -- nothing saved.")
            print("Done.")


if __name__ == "__main__":
    main()
