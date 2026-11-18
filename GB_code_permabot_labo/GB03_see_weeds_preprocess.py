#!/usr/bin/env python3
"""
GB03_see_weeds_preprocess.py

Detects weeds with GB02_see_weeds_yolo.py's fine-tuned YOLO model (reused as-is),
then instead of physically moving the arm over one, burns a crosshair at its
pixel location into the live "top" camera feed -- same
click_target_processor.ClickTargetOverlayStep mechanism GB01_record/eval_click_permabot.py
use, except the target pixel comes from a detection the user picks, not a
human click or a random choice.

Unlike the shape-grasping GB03 (GB_code_camera/GB03_see_shape_preprocess.py),
which asks for a piece NAME and picks randomly among same-type matches, this
shows every detection at once (crop plants for context, weeds numbered) and
lets you pick a SPECIFIC weed by number -- there's only one target category
("weed") here, not several, so deliberate selection makes more sense than
random choice among same-category matches.

Crosshair colour is fixed RED, matching what GB01_record_click_permabot.py
was recorded with (CROSSHAIR_COLOR there) -- not per-category like the shape
pipeline, since there's only one target category here.

No robot movement, no calibration/homography/inverse-kinematics -- the
crosshair-conditioned policy is expected to reach the weed and remove it
(hoe strike) from vision alone -- no separate "drop" phase like the shape
pipeline's GB04 needed, since a hoe doesn't hold anything to carry
elsewhere; the strike itself is the complete action. The eval attempt is
NOT recorded to a dataset -- runs the policy live, reusing GB03's
settle-based early-stop. Only the teleop-driven reset phase between
attempts reuses record_loop() (dataset=None, policy=None).

Usage: press Enter to detect, then type a weed number to target it (or 'r'
       to retake the detection, 'q' to quit).
"""

import time
import traceback
from collections import deque

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

from click_target_processor import (
    ClickTargetOverlayStep,
    make_robot_observation_processor_with_click_target,
)
from GB02_see_weeds_yolo import find_plant_blobs_yolo

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so_leader.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from lerobot.utils.utils import get_safe_torch_device, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ---------------------------------------------------------------------------
# Robot / teleop -- same as GB01_eval_click_permabot.py
# ---------------------------------------------------------------------------
PORT        = "/dev/ttyACM1"
ROBOT_ID    = "my_awesome_follower_arm_binnette"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID   = "my_awesome_leader_arm"

# ---------------------------------------------------------------------------
# Cameras -- same 3 cameras / settings as GB01_eval_click_permabot.py.
# "top" doubles as both the YOLO detection camera and the crosshair-marked
# policy input -- must be the same physical feed so detected pixel
# coordinates line up exactly with what the crosshair gets burned into.
# ---------------------------------------------------------------------------
TOP_CAMERA_NAME    = "top"
TOP_CAMERA_INDEX   = "/dev/video2"
TOP_CAMERA_WIDTH   = 640
TOP_CAMERA_HEIGHT  = 480
TOP_CAMERA_FPS     = 30
TOP_CAMERA_FOURCC  = "MJPG"

RIGHT_CAMERA_NAME   = "right"
RIGHT_CAMERA_INDEX  = "/dev/video4"
RIGHT_CAMERA_WIDTH  = 640
RIGHT_CAMERA_HEIGHT = 480
RIGHT_CAMERA_FPS    = 30

LEFT_CAMERA_NAME    = "left"
LEFT_CAMERA_INDEX   = "/dev/video6"
LEFT_CAMERA_WIDTH   = 640
LEFT_CAMERA_HEIGHT  = 480
LEFT_CAMERA_FPS     = 30

FPS = 30

# ---------------------------------------------------------------------------
# YOLO model -- same MODEL_DIR convention as GB02_see_weeds_yolo.py: points
# at the model's FOLDER (not the .pt file directly), looks for
# weights/best.pt inside it. Change this to swap in a newer model.
# ---------------------------------------------------------------------------
MODEL_DIR             = "yolo_PermaBotLaboWeedDetection_pretest"
CONFIDENCE_THRESHOLD  = 0.25

# ---------------------------------------------------------------------------
# Policy / task -- a single policy here (unlike the shape pipeline's
# per-piece dict), since there's only one target category: weed.
# ---------------------------------------------------------------------------
POLICY_PATH      = "Greynar/act_PermaBotLabo_v1"   # <- edit to the policy you're evaluating
TRAIN_DATASET_ID = "Greynar/PermaBotLabo_v1"  # <- read-only: stats/feature shapes only, nothing recorded
RENAME_MAP       = {}
SINGLE_TASK      = "Remove the weed marked with a crosshair"

RESET_TIME_S = 60
DISPLAY_DATA = True

# ---------------------------------------------------------------------------
# Eval (imitation-learning) phase -- settle-based early stop, identical
# mechanism to GB_code_camera/GB03_see_shape_preprocess.py.
# ---------------------------------------------------------------------------
EVAL_DURATION_S      = 20.0
GRIPPER_SETTLE_DELTA = 3.0
ARM_SETTLE_DELTA     = 3.0
SETTLE_WINDOW_S      = 2.0


def _settle_delta_for(joint_key):
    return GRIPPER_SETTLE_DELTA if joint_key.startswith("gripper") else ARM_SETTLE_DELTA


# ---------------------------------------------------------------------------
# Crosshair -- fixed, matches what GB01_record_click_permabot.py recorded with
# ---------------------------------------------------------------------------
CROSSHAIR_COLOR     = (255, 0, 0)  # RED, RGB
CROSSHAIR_THICKNESS = 2


# ---------------------------------------------------------------------------
# Detection display + selection -- weeds get a number, crop shown for
# context only. Non-blocking so the image stays visible while you type your
# choice in the terminal.
# ---------------------------------------------------------------------------
def show_detections_for_selection(frame_bgr, detections):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    numbered = []  # index i -> detection for weed number i+1
    for d in detections:
        x, y, w, h = d["bbox"]
        is_weed = "weed" in d["classification"].lower()
        color = "red" if is_weed else "deepskyblue"
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=2, edgecolor=color, facecolor='none'))

        if is_weed:
            numbered.append(d)
            label = f"#{len(numbered)} {d['classification']} {d['confidence']:.2f}"
        else:
            label = f"{d['classification']} {d['confidence']:.2f}"
        ax.text(x, y - 6, label, color=color, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

    ax.set_title(f"{len(numbered)} weed(s) found -- pick a number in the terminal", fontsize=13)
    ax.axis('off')

    plt.show(block=False)
    plt.pause(0.1)
    return fig, numbered


# ---------------------------------------------------------------------------
# Eval (imitation-learning) phase
# ---------------------------------------------------------------------------
def load_eval_policy():
    print(f"Loading policy '{POLICY_PATH}' ...")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = POLICY_PATH

    ds_meta = LeRobotDatasetMetadata(TRAIN_DATASET_ID)

    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=RENAME_MAP)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, RENAME_MAP),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    print("Policy ready.\n")
    return policy, preprocessor, postprocessor


def run_eval_policy(robot, policy, preprocessor, postprocessor, robot_action_processor,
                     robot_observation_processor, dataset_features, task,
                     duration_s=EVAL_DURATION_S, fps=FPS, settle_window_s=SETTLE_WINDOW_S):
    """Run the policy until it settles or duration_s elapses. No dataset
    recording -- robot_observation_processor still burns the crosshair into
    the "top" frame the policy sees."""
    policy.reset()

    dt           = 1.0 / fps
    steps        = int(duration_s * fps)
    window_steps = int(settle_window_s * fps)
    pos_history  = deque(maxlen=window_steps)

    print(f"  Running policy for up to {duration_s:.1f}s "
          f"(early stop if gripper moves < {GRIPPER_SETTLE_DELTA} and other joints "
          f"move < {ARM_SETTLE_DELTA}, over {settle_window_s:.1f}s) ...")

    for step in range(steps):
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

        if DISPLAY_DATA:
            log_rerun_data(observation=obs_processed, action=act_processed)

        positions = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if len(pos_history) == window_steps:
            oldest  = pos_history[0]
            settled = all(abs(positions[j] - oldest[j]) < _settle_delta_for(j)
                          for j in positions)
            if settled:
                elapsed = (step + 1) / fps
                print(f"  Settled after {elapsed:.1f}s "
                      f"(gripper < {GRIPPER_SETTLE_DELTA}, other joints < {ARM_SETTLE_DELTA}, "
                      f"over the last {settle_window_s:.1f}s).")
                return
        pos_history.append(positions)

        time.sleep(dt)

    print(f"  Eval attempt finished -- hit the {duration_s:.1f}s timeout without settling.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GB03 -- See Weeds, Crosshair-Preprocess, Eval  (no robot movement to reach the weed)")
    print("=" * 60)

    checkpoint_path = f"{MODEL_DIR}/weights/best.pt"
    print(f"Loading YOLO model: {checkpoint_path}")
    yolo_model = YOLO(checkpoint_path)
    print(f"Classes: {yolo_model.names}\n")

    cameras = {
        TOP_CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=TOP_CAMERA_INDEX,
            width=TOP_CAMERA_WIDTH, height=TOP_CAMERA_HEIGHT, fps=TOP_CAMERA_FPS,
            fourcc=TOP_CAMERA_FOURCC,
        ),
        RIGHT_CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=RIGHT_CAMERA_INDEX,
            width=RIGHT_CAMERA_WIDTH, height=RIGHT_CAMERA_HEIGHT, fps=RIGHT_CAMERA_FPS,
        ),
        LEFT_CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=LEFT_CAMERA_INDEX,
            width=LEFT_CAMERA_WIDTH, height=LEFT_CAMERA_HEIGHT, fps=LEFT_CAMERA_FPS,
        ),
    }

    robot_config = SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras=cameras)
    robot = SO101Follower(robot_config)

    teleop_config = SO101LeaderConfig(port=TELEOP_PORT, id=TELEOP_ID)
    teleop = SO101Leader(teleop_config)

    overlay_step = ClickTargetOverlayStep(
        image_key=TOP_CAMERA_NAME, color=CROSSHAIR_COLOR, thickness=CROSSHAIR_THICKNESS
    )

    teleop_action_processor, robot_action_processor, _ = make_default_processors()
    robot_observation_processor = make_robot_observation_processor_with_click_target(overlay_step)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
        ),
    )

    policy, preprocessor, postprocessor = load_eval_policy()

    if DISPLAY_DATA:
        init_rerun(session_name="recording")

    print("Press Enter to detect. Type a weed number to target it, "
          "'r' to retake the detection, 'q' to quit.\n")

    listener = None
    try:
        robot.connect()
        teleop.connect()
        print("Connected.")

        listener, events = init_keyboard_listener()

        while not events["stop_recording"]:
            cmd = input("Detect> ").strip().lower()
            if cmd == 'q':
                break

            frame_rgb_robot = robot.get_observation()[TOP_CAMERA_NAME]
            frame_bgr = cv2.cvtColor(frame_rgb_robot, cv2.COLOR_RGB2BGR)
            detections = find_plant_blobs_yolo(frame_bgr, yolo_model)

            weed_count = sum(1 for d in detections if "weed" in d["classification"].lower())
            if weed_count == 0:
                print("  No weeds detected. Try again.\n")
                continue

            fig, numbered = show_detections_for_selection(frame_bgr, detections)

            chosen = None
            while True:
                choice = input(f"  Which weed to remove? [1-{len(numbered)}] "
                               f"(or 'r' to retake, 'q' to quit): ").strip().lower()
                if choice == 'q':
                    plt.close(fig)
                    events["stop_recording"] = True
                    break
                if choice == 'r':
                    plt.close(fig)
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(numbered):
                    chosen = numbered[int(choice) - 1]
                    plt.close(fig)
                    break
                print(f"  Invalid choice '{choice}'.")

            if chosen is None:
                print()
                continue

            u, v = chosen["center"]
            overlay_step.set_target(u, v)
            print(f"  Crosshair set at {overlay_step.target_uv} "
                  f"(conf={chosen['confidence']:.2f})")

            run_eval_policy(robot, policy, preprocessor, postprocessor,
                             robot_action_processor, robot_observation_processor,
                             dataset_features, task=SINGLE_TASK)

            if not events["stop_recording"]:
                log_say("Reset the environment", True)
                overlay_step.target_uv = None
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=RESET_TIME_S,
                    display_data=DISPLAY_DATA,
                )
            print()

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
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
