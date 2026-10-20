#!/usr/bin/env python3
"""
GB04_see_shape_preprocess_drop.py

Same as GB03_see_shape_preprocess.py (detect piece -> burn a crosshair on the
live "top" feed -> run the piece's crosshair-conditioned policy live, no
robot movement to reach the piece), with one addition: once the eval attempt
finishes (piece supposedly grasped), replay a single pre-recorded episode
that moves the arm to an exact drop location, opens the gripper, and returns
to its initial position -- then the normal teleop-driven reset phase runs as
before.

The drop replay is open-loop (no policy, no dataset recording) -- it just
reads the DROP_DATASET_REPO_ID episode's `action` column in order and sends
each action via robot.send_action(), at that dataset's fps. This is the same
mechanism as lerobot's own lerobot_replay.py (see that script), adapted to
reuse the already-connected robot instead of connecting a separate one.
The drop episode was recorded WITHOUT any cameras (replay never looks at
images, only actions -- see the discussion that led to this script), so
loading it is cheap and has no camera/feature overlap with the eval phase's
3-camera dataset_features.

Runs automatically after every eval attempt for a supported piece -- there's
no way to automatically detect whether the grasp actually succeeded (no force
sensor, no vision check), so if the grasp failed the replay is just a
harmless no-op (opens an empty gripper at the drop spot). The replay itself
is an uninterruptible loop, same precedent as lerobot_replay.py and GB03's
run_eval_policy -- no keyboard exit-early check during the ~few-second replay.

Usage: type a piece name (partial match accepted, e.g. "star", "orange").
       A fresh detection snapshot is taken for every query. Type 'q' to quit.
"""

import random
import time
import traceback
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from click_target_processor import (
    ClickTargetOverlayStep,
    make_robot_observation_processor_with_click_target,
)
from GB02_see_shape import PIECES, find_all_instances, match_piece

from lerobot.cameras import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
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
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, log_say
from lerobot.utils.visualization_utils import init_rerun

# ---------------------------------------------------------------------------
# Robot / teleop
# ---------------------------------------------------------------------------
PORT        = "/dev/ttyACM1"
ROBOT_ID    = "my_awesome_follower_arm"

TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID   = "my_awesome_leader_arm"

# ---------------------------------------------------------------------------
# Cameras -- same 3 cameras / settings as GB01_eval_click_to_preprocess.py.
# "top" doubles as both the GB02 detection camera and the crosshair-marked
# policy input -- must be the same physical feed so detected pixel
# coordinates line up exactly with what the crosshair gets burned into.
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

USE_THIRD_CAMERA    = True
FRONT_CAMERA_NAME   = "front"
FRONT_CAMERA_INDEX  = "/dev/video6"
FRONT_CAMERA_WIDTH  = 640
FRONT_CAMERA_HEIGHT = 480
FRONT_CAMERA_FPS    = 30

FPS = 30

# ---------------------------------------------------------------------------
# Detection display
# ---------------------------------------------------------------------------
DETECTION_DISPLAY_SECONDS = 3.0

# ---------------------------------------------------------------------------
# Policy / task -- one separate policy per piece (a different strategy than
# GB06's single shared checkpoint): each shape gets its own trained checkpoint.
# A piece with policy_path=None has no trained policy yet -- it still gets
# detected + a crosshair burned in for inspection, but the eval phase is
# skipped with a placeholder message. rename_map maps our dataset's real
# camera names to whatever that specific checkpoint was trained with -- check
# ITS OWN train_config.json ("rename_map") before assuming {} still applies
# (see GB01_eval_click_to_preprocess.py). Policies are loaded lazily (only the
# first time a piece is actually requested) and cached for the rest of the run.
# ---------------------------------------------------------------------------
TASK_TEMPLATE = "Grab the {piece} marked with a crosshair"  # <- {piece} filled in with the detected piece's label

PIECE_POLICIES = {
    "yellow hexagon": {
        "policy_path":      "Greynar/act__ClickTargetPreprocessThreeCamerasSetUpOneMultiplePieces",
        "train_dataset_id": "Greynar/ClickTargetPreprocessThreeCamerasSetUpOneMultiplePieces",
        "rename_map":       {},
    },
    "red triangle": {
        "policy_path":      None,  # <- placeholder: training in progress, fill in tomorrow
        "train_dataset_id": None,
        "rename_map":       {},
    },
    "green star": {
        "policy_path":      None,  # <- placeholder: not trained yet
        "train_dataset_id": None,
        "rename_map":       {},
    },
    "orange circle": {
        "policy_path":      None,  # <- placeholder: not trained yet
        "train_dataset_id": None,
        "rename_map":       {},
    },
}

RESET_TIME_S = 60
DISPLAY_DATA = True

# ---------------------------------------------------------------------------
# Eval (imitation-learning) phase -- settle-based early stop, identical
# mechanism to GB06_so101_see_shape_move_imitation_learning.py's grasp phase.
# ---------------------------------------------------------------------------
EVAL_DURATION_S      = 20.0   # hard ceiling -- early-stop below should usually end it sooner
GRIPPER_SETTLE_DELTA = 1.0    # 0-100 (closed-open) scale
ARM_SETTLE_DELTA     = 3.0    # degrees, applies to the other 5 joints
SETTLE_WINDOW_S      = 5.0


def _settle_delta_for(joint_key):
    return GRIPPER_SETTLE_DELTA if joint_key.startswith("gripper") else ARM_SETTLE_DELTA


# ---------------------------------------------------------------------------
# Drop replay -- single pre-recorded episode (no cameras), open-loop replay
# of its `action` column. See lerobot/scripts/lerobot_replay.py.
# ---------------------------------------------------------------------------
DROP_DATASET_REPO_ID = "Greynar/drop_to_exact_location"
DROP_EPISODE_INDEX   = 0


# ---------------------------------------------------------------------------
# Crosshair colour per piece (RGB -- matches OpenCVCameraConfig/lerobot's RGB
# image convention, same values GB01_record/eval_click_to_preprocess.py use)
# ---------------------------------------------------------------------------
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN  = (0, 255, 0)
ORANGE = (255, 165, 0)

PIECE_CROSSHAIR_COLOR = {
    "orange circle":  ORANGE,
    "green star":     GREEN,
    "red triangle":   RED,
    "yellow hexagon": YELLOW,
}
CROSSHAIR_THICKNESS = 2


# ---------------------------------------------------------------------------
# Detection display (piece already RGB -- unlike GB02/GB05/GB06 this frame
# comes from robot.get_observation(), not a raw BGR cv2.VideoCapture read)
# ---------------------------------------------------------------------------
def show_detection(frame_rgb, piece, all_instances, chosen, display_seconds):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    for inst in all_instances:
        x, y, w, h = inst["bbox"]
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=1, edgecolor='white', facecolor='none',
                     linestyle='--'))

    cx, cy     = chosen["center"]
    x, y, w, h = chosen["bbox"]
    color_hex  = "#{:02x}{:02x}{:02x}".format(*piece["draw_color"][::-1])  # BGR->RGB
    ax.add_patch(patches.Rectangle((x, y), w, h,
                 linewidth=2, edgecolor=color_hex, facecolor='none'))
    ax.plot(cx, cy, '+', color=color_hex, markersize=18, markeredgewidth=2)
    ax.text(x, y - 8, piece["label"],
            color=color_hex, fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='none'))

    n = len(all_instances)
    title = f"Found {n}x {piece['label']}"
    if n > 1:
        title += "  -- one chosen at random"
    title += f"  -- crosshair in {display_seconds:.0f}s ..."
    ax.set_title(title, fontsize=13)
    ax.axis('off')

    plt.draw()
    plt.pause(display_seconds)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eval (imitation-learning) phase
# ---------------------------------------------------------------------------
def load_eval_policy(policy_path, train_dataset_id, rename_map):
    """Load one piece's policy + processors. train_dataset_id is only used to
    read normalization stats / feature shapes -- nothing is recorded."""
    print(f"Loading policy '{policy_path}' ...")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path

    ds_meta = LeRobotDatasetMetadata(train_dataset_id)

    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    print("Policy ready.\n")
    return policy, preprocessor, postprocessor


def run_eval_policy(robot, policy, preprocessor, postprocessor, robot_action_processor,
                     robot_observation_processor, dataset_features, task,
                     duration_s=EVAL_DURATION_S, fps=FPS, settle_window_s=SETTLE_WINDOW_S):
    """
    Run the policy until it settles or duration_s elapses, whichever comes
    first. "Settled" = every joint (all 6, gripper included) has moved less
    than its own threshold over the last settle_window_s seconds. No dataset
    recording -- robot_observation_processor still burns the crosshair into
    the "top" frame the policy sees, via the same overlay_step used elsewhere.
    """
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
# Drop replay
# ---------------------------------------------------------------------------
def load_drop_episode():
    """Load the pre-recorded drop episode's actions once, up front. The
    dataset was recorded without cameras -- replay only ever needs the
    ACTION column, same as lerobot_replay.py."""
    print(f"Loading drop episode '{DROP_DATASET_REPO_ID}' episode {DROP_EPISODE_INDEX} ...")
    dataset = LeRobotDataset(DROP_DATASET_REPO_ID, episodes=[DROP_EPISODE_INDEX])
    # Filter to only this episode's rows since episodes are chunked in dataset v3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == DROP_EPISODE_INDEX)
    actions        = episode_frames.select_columns(ACTION)
    action_names   = dataset.features[ACTION]["names"]
    print(f"Drop episode ready ({len(episode_frames)} frames).\n")
    return actions, action_names, dataset.fps


def run_drop_replay(robot, robot_action_processor, actions, action_names, fps=None):
    """
    Open-loop replay of the pre-recorded drop episode: move to the exact
    drop location, open the gripper, return to the initial position. Same
    mechanism as lerobot's own lerobot_replay.py, reusing the already-
    connected robot (with its cameras) instead of connecting a separate
    camera-free one -- the extra camera observations are simply unused here.
    Uninterruptible, like lerobot_replay.py and run_eval_policy above.
    """
    fps = fps or FPS
    dt  = 1.0 / fps
    print(f"  Replaying drop episode ({len(actions)} frames) ...")

    for idx in range(len(actions)):
        start_t = time.perf_counter()

        action_array = actions[idx][ACTION]
        action = {name: action_array[i] for i, name in enumerate(action_names)}

        obs             = robot.get_observation()
        processed_action = robot_action_processor((action, obs))
        robot.send_action(processed_action)

        elapsed = time.perf_counter() - start_t
        precise_sleep(max(dt - elapsed, 0.0))

    print("  Drop replay finished.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GB04 -- See Shape, Crosshair-Preprocess, Eval, Drop  (no robot movement to reach the piece)")
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

    overlay_step = ClickTargetOverlayStep(image_key=TOP_CAMERA_NAME, thickness=CROSSHAIR_THICKNESS)

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

    loaded_policies = {}  # piece label -> (policy, preprocessor, postprocessor), filled in lazily

    drop_actions, drop_action_names, drop_fps = load_drop_episode()

    if DISPLAY_DATA:
        init_rerun(session_name="recording")

    known = "  |  ".join(p["label"] for p in PIECES)
    supported = [label for label, cfg in PIECE_POLICIES.items() if cfg["policy_path"] is not None]
    print(f"Known pieces: {known}")
    print(f"Eval (imitation learning) supported for: {', '.join(supported) if supported else '(none yet)'}")
    print("Partial names accepted (e.g. 'star', 'orange'). Type 'q' to quit.\n")

    listener = None
    try:
        robot.connect()
        teleop.connect()
        print("Connected.")

        listener, events = init_keyboard_listener()

        while not events["stop_recording"]:
            name = input("Piece to find: ").strip()
            if not name or name.lower() == 'q':
                break

            piece = match_piece(name)
            if piece is None:
                print(f"  Unknown piece '{name}'. Known: {known}\n")
                continue

            print(f"  Looking for '{piece['label']}' ...")
            frame_rgb = robot.get_observation()[TOP_CAMERA_NAME]
            hsv       = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            instances = find_all_instances(hsv, piece)

            if not instances:
                print(f"  '{piece['label']}' not detected. Try again.\n")
                continue

            chosen = random.choice(instances)
            u, v   = chosen["center"]
            print(f"  Found {len(instances)}x '{piece['label']}'"
                  + (" -- chosen at random" if len(instances) > 1 else ""))
            print(f"  Pixel=({u}, {v})")

            show_detection(frame_rgb, piece, instances, chosen, DETECTION_DISPLAY_SECONDS)

            overlay_step.color = PIECE_CROSSHAIR_COLOR[piece["label"]]
            overlay_step.set_target(u, v)
            print(f"  Crosshair set at {overlay_step.target_uv} (colour={overlay_step.color})")

            piece_policy_cfg = PIECE_POLICIES[piece["label"]]
            if piece_policy_cfg["policy_path"] is not None:
                if piece["label"] not in loaded_policies:
                    loaded_policies[piece["label"]] = load_eval_policy(
                        piece_policy_cfg["policy_path"],
                        piece_policy_cfg["train_dataset_id"],
                        piece_policy_cfg["rename_map"],
                    )
                policy, preprocessor, postprocessor = loaded_policies[piece["label"]]

                task = TASK_TEMPLATE.format(piece=piece["label"])
                run_eval_policy(robot, policy, preprocessor, postprocessor,
                                 robot_action_processor, robot_observation_processor,
                                 dataset_features, task=task)

                # Piece supposedly grasped -- no explicit action is sent between
                # here and the replay's first frame, so the gripper just holds
                # its last commanded (closed) position in the meantime.
                run_drop_replay(robot, robot_action_processor,
                                 drop_actions, drop_action_names, fps=drop_fps)
            else:
                print(f"  Imitation learning not supported yet for '{piece['label']}'.")

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
