#!/usr/bin/env python3
"""
test_policy_inference.py

Standalone test: verifies get_observation() / policy inference / send_action()
work on this LeRobot install. No dataset writing, no leader arm.
Runs the ACT policy for a fixed duration, then stops.

Mirrors what `lerobot-record --policy.path=...` does internally
(see lerobot/scripts/lerobot_record.py: record() / record_loop()),
minus dataset creation.
"""

import time
import traceback

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
PORT             = "/dev/ttyACM0"
POLICY_PATH      = "act_CameraOnGripperColoredPiece"
TRAIN_DATASET_ID = "Greynar/CameraOnGripperColoredPiece"  # from train_config.json — read-only, stats/shapes only, nothing recorded
TASK             = "Grab the yellow hexagone"
FPS              = 30
DURATION_S       = 8.0   # fixed duration for now

robot_config = SO101FollowerConfig(
    port=PORT,
    id="my_awesome_follower_arm",
    cameras={
        "onGripper": OpenCVCameraConfig(
            index_or_path="/dev/video4",
            width=480, height=640, fps=FPS,
            rotation=Cv2Rotation.ROTATE_270,   # -90 degrees
        )
    },
)


def main():
    robot = SO101Follower(robot_config)

    print("Loading policy config …")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = POLICY_PATH

    print(f"Loading dataset metadata from {TRAIN_DATASET_ID} (stats/shapes only) …")
    ds_meta = LeRobotDatasetMetadata(TRAIN_DATASET_ID)

    print("Building policy + pre/post-processors …")
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )

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

    try:
        robot.connect()
        print("Connected.")

        obs = robot.get_observation()
        print("get_observation() keys:", list(obs.keys()))   # <- the check you wanted

        policy.reset()

        dt    = 1.0 / FPS
        steps = int(DURATION_S * FPS)
        print(f"Running policy for {DURATION_S:.1f}s ({steps} steps) …")

        for i in range(steps):
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
                task=TASK,
                robot_type=robot.robot_type,
            )
            act_processed = make_robot_action(action_values, dataset_features)
            robot_action  = robot_action_processor((act_processed, obs))
            robot.send_action(robot_action)

            time.sleep(dt)

        print("Fixed duration elapsed — stopping.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

