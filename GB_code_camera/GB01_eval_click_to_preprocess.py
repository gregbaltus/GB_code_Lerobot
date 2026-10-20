#!/usr/bin/env python3
"""
GB01_eval_click_to_preprocess.py

Runs a trained policy and records the eval episodes, same as:

    lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
        --robot.cameras="{ top: {...}, front: {...} }" \
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \
        --dataset.repo_id=... --dataset.single_task=... \
        --policy.path=... --display_data=true

...except before each episode, a snapshot of the "top" camera is shown and
you click the target pixel once. A crosshair is then burned into every frame
of the "top" image the POLICY sees for that episode (and into what gets
recorded), using the exact same click_target_processor.ClickTargetOverlayStep
class as GB01_record_click_to_preprocess.py -- so the policy is shown the same
kind of annotated image at inference time that it was trained on.

Teleop is still connected (matches your original eval command) and is used
for the between-episode reset phase, not to drive the recorded episodes --
the policy drives those.

All parameters are constants below -- edit them before each eval session.
NUM_EPISODES defaults to 10 here instead of lerobot-record's default of 50.
"""

from click_target_processor import (
    ClickTargetOverlayStep,
    make_robot_observation_processor_with_click_target,
    wait_for_click_on_image,
)

from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader.so101_leader.so101_leader import SO101Leader
from lerobot.teleoperators.so_leader.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ---------------------------------------------------------------------------
# Robot / teleop
# ---------------------------------------------------------------------------
PORT        = "/dev/ttyACM1"
ROBOT_ID    = "my_awesome_follower_arm"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID   = "my_awesome_leader_arm"

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
TOP_CAMERA_NAME     = "top"
TOP_CAMERA_INDEX    = "/dev/video2"
TOP_CAMERA_WIDTH    = 640
TOP_CAMERA_HEIGHT   = 480
TOP_CAMERA_FPS      = 30

SECOND_CAMERA_NAME     = "onGripper"
SECOND_CAMERA_INDEX    = "/dev/video4"
SECOND_CAMERA_WIDTH    = 480
SECOND_CAMERA_HEIGHT   = 640
SECOND_CAMERA_FPS      = 30
SECOND_CAMERA_ROTATION = Cv2Rotation.ROTATE_270   # -90 degrees

# Third camera is optional -- architecture choice, not a per-run convenience:
# must match whatever GB01_record_click_to_preprocess.py used for the dataset
# that trained POLICY_PATH.
USE_THIRD_CAMERA    = True
FRONT_CAMERA_NAME   = "front"
FRONT_CAMERA_INDEX  = "/dev/video6"
FRONT_CAMERA_WIDTH  = 640
FRONT_CAMERA_HEIGHT = 480
FRONT_CAMERA_FPS    = 30

# ---------------------------------------------------------------------------
# Policy / dataset / task
# ---------------------------------------------------------------------------
POLICY_PATH     = "Greynar/act__ClickTargetPreprocessThreeCamerasSetUpOneMultiplePieces"    # <- edit to the policy you're evaluating
DATASET_REPO_ID = "Greynar/eval_act__ClickTargetPreprocessThreeCamerasSetUpOneMultiplePieces"
SINGLE_TASK     = "Grab the yellow hexagon marked with a crosshair"
NUM_EPISODES    = 5  # lerobot-record's own default is 50 -- overridden here
FPS             = 30
EPISODE_TIME_S  = 90
RESET_TIME_S    = 60
PUSH_TO_HUB     = True
DISPLAY_DATA    = True
RESUME          = True  # True = append new episodes to the existing DATASET_REPO_ID dataset
                           # (equivalent to lerobot-record --resume=true), instead of creating it fresh

# Maps our dataset's real camera names to whatever POLICY_PATH was actually
# trained with. Taken directly from this checkpoint's own train_config.json
# ("rename_map") -- not guessed, so it's safe to trust. This checkpoint
# (act__ClickTargetPreprocessThreeCamerasSetUpOneMultiplePieces) was trained
# with rename_map={} (empty) -- it expects the native top/onGripper/front
# names as-is, unlike the earlier SmolVLA checkpoint which used camera1/2/3.
# If you evaluate a different POLICY_PATH, check ITS train_config.json for
# its own rename_map before assuming this one still applies.
RENAME_MAP = {}

# ---------------------------------------------------------------------------
# Crosshair appearance -- keep identical to GB01_record_click_to_preprocess.py
# (colors are RGB -- OpenCVCameraConfig default)
# ---------------------------------------------------------------------------
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN  = (0, 255, 0)
ORANGE = (255, 165, 0)

CROSSHAIR_COLOR     = YELLOW
CROSSHAIR_THICKNESS = 2


def main():
    print("GB01 — Click-Target Preprocessing: EVAL")
    print("=" * 60)

    cameras = {
        TOP_CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=TOP_CAMERA_INDEX,
            width=TOP_CAMERA_WIDTH, height=TOP_CAMERA_HEIGHT, fps=TOP_CAMERA_FPS,
        ),
        SECOND_CAMERA_NAME: OpenCVCameraConfig(
            index_or_path=SECOND_CAMERA_INDEX,
            width=SECOND_CAMERA_WIDTH, height=SECOND_CAMERA_HEIGHT, fps=SECOND_CAMERA_FPS,
            rotation=SECOND_CAMERA_ROTATION,
        ),
    }
    if USE_THIRD_CAMERA:
        cameras[FRONT_CAMERA_NAME] = OpenCVCameraConfig(
            index_or_path=FRONT_CAMERA_INDEX,
            width=FRONT_CAMERA_WIDTH, height=FRONT_CAMERA_HEIGHT, fps=FRONT_CAMERA_FPS,
        )

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

    if RESUME:
        dataset = LeRobotDataset(DATASET_REPO_ID)
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            DATASET_REPO_ID,
            FPS,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
        )

    print(f"Loading policy '{POLICY_PATH}' …")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = POLICY_PATH
    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, RENAME_MAP),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    print("Policy ready.\n")

    if DISPLAY_DATA:
        init_rerun(session_name="recording")

    listener = None
    try:
        robot.connect()
        teleop.connect()
        print("Connected.")

        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                print(f"\n--- Episode {recorded_episodes + 1}/{NUM_EPISODES} ---")

                top_frame = robot.get_observation()[TOP_CAMERA_NAME]
                click = wait_for_click_on_image(
                    top_frame, f"Episode {recorded_episodes + 1}/{NUM_EPISODES} — click the target"
                )
                if click is None:
                    print("No click recorded — stopping.")
                    break
                overlay_step.set_target(*click)
                print(f"  Target set to pixel {overlay_step.target_uv}")

                log_say(f"Recording episode {dataset.num_episodes}", True)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_S,
                    single_task=SINGLE_TASK,
                    display_data=DISPLAY_DATA,
                )

                # Give time to manually reset the environment (no crosshair during this phase)
                if not events["stop_recording"] and (
                    (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
                ):
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
                        single_task=SINGLE_TASK,
                        display_data=DISPLAY_DATA,
                    )

                if events["rerecord_episode"]:
                    print("Re-recording episode …")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        print("\nFinalizing …")
        if 'dataset' in locals() and dataset is not None:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()
        if PUSH_TO_HUB and 'dataset' in locals() and dataset is not None:
            dataset.push_to_hub()
        print("Done.")


if __name__ == "__main__":
    main()
