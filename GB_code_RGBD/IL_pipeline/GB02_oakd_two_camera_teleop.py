#!/usr/bin/env python3
"""
GB02_oakd_two_camera_teleop.py

Teleoperates the SO-101 (leader -> follower, same ports as the rest of this
project) while treating the single OAK-D RGB-D camera as TWO lerobot
"cameras": the real RGB stream, and the depth map re-encoded as an 8-bit
3-channel image under its own camera key -- the trick discussed in chat:
ACT (and lerobot's observation/visualization pipeline in general) has no
notion of "this must be real RGB", it just expects named, image-shaped
tensors. A depth map encoded as an image satisfies that with zero changes
to anything else.

Why a hand-rolled loop instead of reusing record_loop() like every other
GB_code_* teleop script: record_loop() pulls camera frames through
robot.get_observation(), which only knows about cameras wired into
SO101FollowerConfig(cameras=...) -- i.e. OpenCVCameraConfig/v4l2 devices.
OAK-D isn't one of those (it talks over the depthai SDK, confirmed in
GB01_oakd_see_streams.py), so its frames have to be pulled separately and
merged in by hand. The teleop/action-processing sequence below is copied
EXACTLY from record_loop()'s own body (lerobot/scripts/lerobot_record.py)
so the robot is driven identically to every other teleop script in this
project -- only the observation side (adding the two OAK-D keys before
display) is different.

DEPTH ENCODING -- fixed range, not per-frame normalized (per Intel's own
RealSense depth-colorization guidance: per-frame normalization makes the
same real-world distance encode differently frame to frame, which would
mislead a policy trying to learn from it). Plain linear grayscale
(replicated into 3 channels), not a colormap or byte-split: ACT's vision
backbone is SHARED across all camera views (confirmed in
lerobot/policies/act/modeling_act.py -- one `self.backbone` called per
camera), and a grayscale-style encoding stays closer to natural-image
statistics than a colormap (perceptually non-uniform, can fabricate false
edges) or a byte-split (fully precise but looks like noise to a backbone,
especially one pretrained on ImageNet).

DEPTH_MIN_MM/DEPTH_MAX_MM below are placeholders -- TUNE to the actual
working distance range of your setup once you know it; this determines how
much of the 8-bit range is "wasted" vs usefully spent on your real depths.

Press ESC to stop (same keybinding as every other GB_code_* script, via
lerobot's own init_keyboard_listener()).

Run with: /home/gregory/miniconda3/envs/lerobot/bin/python GB02_oakd_two_camera_teleop.py
(depthai was installed into the `lerobot` conda env for this script --
confirmed to coexist fine with the SO101Follower/Leader imports.)
"""

import time
import traceback

import cv2
import depthai as dai
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so_leader.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ---------------------------------------------------------------------------
# Robot / teleop -- same ports as GB_code_classic_method / GB_code_permabot_labo
# ---------------------------------------------------------------------------
PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm_binnette"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID = "my_awesome_leader_arm"

FPS = 30

# ---------------------------------------------------------------------------
# OAK-D streams
# ---------------------------------------------------------------------------
RGB_SIZE = (1280, 800)   # matches the mono/depth resolution -- keeps both
                         # streams spatially comparable
OAKD_FPS = 15

# ---------------------------------------------------------------------------
# Optional third camera -- a plain v4l2/OpenCV 2D camera, wired in through
# SO101FollowerConfig(cameras=...) same as every other GB_code_* script
# (not through the OAK-D/depthai path), so robot.get_observation() returns
# it automatically under THIRD_CAMERA_NAME, already converted to RGB by
# OpenCVCameraConfig's default color_mode -- no manual BGR2RGB needed here.
# ---------------------------------------------------------------------------
ENABLE_THIRD_CAMERA = True
THIRD_CAMERA_NAME = "video2"
THIRD_CAMERA_INDEX = "/dev/video2"
THIRD_CAMERA_WIDTH = 640
THIRD_CAMERA_HEIGHT = 480
THIRD_CAMERA_FPS = 30

# ---------------------------------------------------------------------------
# Depth -> fake-RGB encoding -- FIXED range, tune to your real workspace
# ---------------------------------------------------------------------------
DEPTH_MIN_MM = 600   # anything closer clips to 0
DEPTH_MAX_MM = 900   # anything farther clips to 255
# Tuned from a live probe with the camera mounted ~75cm over the dirt:
# 95% of valid depth pixels fell between 621-849mm (bulk at 600-850mm), so
# this 300mm window spends the full 8-bit range on the real working surface
# instead of wasting precision on distances that never occur. NOTE: this
# range clips anything closer than 600mm (e.g. a hand/hoe passing right in
# front of the camera) to 0 -- widen the window if that needs to stay visible.


def encode_depth_as_image(depth_mm, min_mm=DEPTH_MIN_MM, max_mm=DEPTH_MAX_MM):
    """uint16 depth (mm) -> uint8 3-channel 'image' via fixed-range linear
    scaling, replicated across channels. Invalid pixels (depthai uses both
    0 and 65535 as 'no reading') are mapped to 0 (near) -- same as clipping
    anything closer than min_mm, since neither should be acted on anyway."""
    valid_depth = np.where(depth_mm >= 65535, 0, depth_mm)
    clipped = np.clip(valid_depth, min_mm, max_mm)
    scaled = ((clipped - min_mm) / (max_mm - min_mm) * 255).astype(np.uint8)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def main():
    print("GB02 -- OAK-D two-camera-trick teleop")
    print("=" * 50)

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

        left_cam = pipeline.create(dai.node.Camera)
        left_cam.setSensorType(dai.CameraSensorType.MONO)
        left_cam.build(dai.CameraBoardSocket.CAM_B, sensorFps=OAKD_FPS)
        right_cam = pipeline.create(dai.node.Camera)
        right_cam.setSensorType(dai.CameraSensorType.MONO)
        right_cam.build(dai.CameraBoardSocket.CAM_C, sensorFps=OAKD_FPS)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        left_cam.requestFullResolutionOutput().link(stereo.left)
        right_cam.requestFullResolutionOutput().link(stereo.right)
        depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print("OAK-D pipeline running.")

        cameras = {}
        if ENABLE_THIRD_CAMERA:
            cameras[THIRD_CAMERA_NAME] = OpenCVCameraConfig(
                index_or_path=THIRD_CAMERA_INDEX,
                width=THIRD_CAMERA_WIDTH,
                height=THIRD_CAMERA_HEIGHT,
                fps=THIRD_CAMERA_FPS,
            )
        # OAK-D itself is NOT a v4l2 camera (depthai SDK, not OpenCVCameraConfig) --
        # only a genuinely separate 2D camera belongs in this dict.
        robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras=cameras)
        robot = SO101Follower(robot_config)
        teleop_config = SO101LeaderConfig(port=TELEOP_PORT, id=TELEOP_ID)
        teleop = SO101Leader(teleop_config)

        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        last_rgb = None
        last_depth_encoded = None
        listener = None

        try:
            robot.connect()
            teleop.connect()
            print("Robot + teleop connected.")

            init_rerun(session_name="oakd_two_camera_trick")
            listener, events = init_keyboard_listener()
            print("\nTeleoperating -- move the leader arm. Press ESC to stop.\n")

            dt = 1.0 / FPS
            while not events["stop_recording"]:
                start = time.perf_counter()

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)

                rgb_pkt = rgb_queue.tryGet()
                if rgb_pkt is not None:
                    # getCvFrame() returns BGR (OpenCV convention) -- rr.Image()
                    # expects RGB, so without this conversion colors swap (red/blue),
                    # e.g. skin tones rendering blue.
                    last_rgb = cv2.cvtColor(rgb_pkt.getCvFrame(), cv2.COLOR_BGR2RGB)
                depth_pkt = depth_queue.tryGet()
                if depth_pkt is not None:
                    last_depth_encoded = encode_depth_as_image(depth_pkt.getCvFrame())

                if last_rgb is not None:
                    obs_processed["oakd_rgb"] = last_rgb
                if last_depth_encoded is not None:
                    obs_processed["oakd_depth_encoded"] = last_depth_encoded

                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
                robot.send_action(robot_action_to_send)

                log_rerun_data(observation=obs_processed, action=act_processed_teleop)

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
            print("Done.")


if __name__ == "__main__":
    main()
