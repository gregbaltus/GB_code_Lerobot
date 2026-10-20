#!/usr/bin/env python3
"""
GB01_so101_coord_control.py

Move the SO-101 arm to a user-specified Cartesian target.
The user provides: x (forward reach), y (height), shoulder_pan, pitch,
wrist_roll, and gripper. Inverse kinematics computes shoulder_lift and
elbow_flex. A P-controller drives the arm to the target, then the script
loops back and asks for the next target.

Type 'q' at any prompt to exit cleanly (arm returns to zero first).
"""

import math
import time
import traceback

from lerobot.robots.so_follower.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so_follower.so101_follower.config_so101_follower import SO101FollowerConfig


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    2-link planar IK for the SO-101 arm.
    Returns (shoulder_lift_deg, elbow_flex_deg).
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r     = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    if r > r_max:
        x *= r_max / r; y *= r_max / r; r = r_max
    if r < r_min and r > 0:
        x *= r_min / r; y *= r_min / r; r = r_min

    cos_t2 = max(-1.0, min(1.0, -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)))
    theta2 = math.pi - math.acos(cos_t2)

    beta   = math.atan2(y, x)
    gamma  = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    return 90 - math.degrees(joint2), math.degrees(joint3) - 90

# ---------------------------------------------------------------------------
# Safe input ranges (conservative subset of hardware limits)
# ---------------------------------------------------------------------------
SAFE_RANGES = {
    "x":            (0.05,   0.23),   # meters  — forward reach
    "y":            (-0.08,  0.20),   # meters  — height
    "shoulder_pan": (-90.0,  110.0),  # degrees — left / right rotation
    "pitch":        (-70.0,  120.0),  # degrees — wrist tilt (wrist_flex coupling)
    "wrist_roll":   (-150.0, 150.0),  # degrees — wrist rotation
    "gripper":      (0.0,    100.0),  # 0 = closed, 100 = fully open
}

# ---------------------------------------------------------------------------
# P-controller parameters
# ---------------------------------------------------------------------------
KP             = 0.5   # proportional gain
CONTROL_FREQ   = 50    # Hz
CONVERGENCE_TH = 3.0   # degrees — total joint error to declare "arrived"
TIMEOUT        = 5.0   # seconds — give up after this long


# ---------------------------------------------------------------------------
# Input helper
# ---------------------------------------------------------------------------
def ask_float(label, low, high):
    """Prompt the user for a float in [low, high].  Returns None on 'q'."""
    while True:
        raw = input(f"  {label} [{low} .. {high}]: ").strip()
        if raw.lower() == "q":
            return None
        try:
            val = float(raw)
        except ValueError:
            print("    → Please enter a number or 'q' to quit.")
            continue
        if low <= val <= high:
            return val
        print(f"    → Out of range. Must be between {low} and {high}.")


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def move_to_zero(robot, kp=KP, control_freq=CONTROL_FREQ, duration=3.0):
    """P-control all joints to the zero position over `duration` seconds."""
    print("Moving to zero position …")
    zero = {
        "shoulder_pan":  0.0,
        "shoulder_lift": 0.0,
        "elbow_flex":    0.0,
        "wrist_flex":    0.0,
        "wrist_roll":    0.0,
        "gripper":       0.0,
    }
    dt    = 1.0 / control_freq
    steps = int(duration * control_freq)
    for _ in range(steps):
        obs    = robot.get_observation()
        action = {
            f"{j}.pos": obs[f"{j}.pos"] + kp * (target - obs[f"{j}.pos"])
            for j, target in zero.items()
        }
        robot.send_action(action)
        time.sleep(dt)
    print("At zero position.")


def p_control_to_target(robot, targets, kp=KP, control_freq=CONTROL_FREQ,
                        timeout=TIMEOUT, threshold=CONVERGENCE_TH):
    """
    Run a P-control loop until all joints are within `threshold` degrees of
    their targets, or `timeout` seconds have elapsed.
    """
    dt        = 1.0 / control_freq
    max_steps = int(timeout * control_freq)

    for step in range(max_steps):
        obs         = robot.get_observation()
        action      = {}
        total_error = 0.0

        for joint, target in targets.items():
            current              = obs[f"{joint}.pos"]
            error                = target - current
            total_error         += abs(error)
            action[f"{joint}.pos"] = current + kp * error

        robot.send_action(action)

        if total_error < threshold:
            print(f"  Reached target in {(step + 1) / control_freq:.2f}s "
                  f"(total error = {total_error:.2f}°)")
            return

        time.sleep(dt)

    print(f"  Timeout after {timeout:.1f}s (final total error = {total_error:.2f}°)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SO-101 Coordinate Control  —  GB01")
    print("=" * 45)

    port = input("USB port (default: /dev/ttyACM0): ").strip() or "/dev/ttyACM0"
    print(f"Connecting on {port} …")

    robot_config = SO101FollowerConfig(port=port, id="my_awesome_follower_arm")
    robot        = SO101Follower(robot_config)

    try:
        robot.connect()
        print("Connected.\n")

        move_to_zero(robot)

        print("\nReady.  Enter target coordinates (type 'q' at any prompt to exit).\n")

        while True:
            print("-" * 45)

            x = ask_float("x  — forward reach  (m)", *SAFE_RANGES["x"])
            if x is None:
                break

            y = ask_float("y  — height         (m)", *SAFE_RANGES["y"])
            if y is None:
                break

            pan = ask_float("shoulder_pan       (°)", *SAFE_RANGES["shoulder_pan"])
            if pan is None:
                break

            pitch = ask_float("pitch / wrist_flex (°)", *SAFE_RANGES["pitch"])
            if pitch is None:
                break

            wrist_roll = ask_float("wrist_roll         (°)", *SAFE_RANGES["wrist_roll"])
            if wrist_roll is None:
                break

            gripper = ask_float("gripper  0=closed 100=open", *SAFE_RANGES["gripper"])
            if gripper is None:
                break

            # Inverse kinematics: (x, y) is the wrist joint target position
            shoulder_lift, elbow_flex = inverse_kinematics(x, y)

            # Wrist flex keeps the gripper at the requested pitch angle
            wrist_flex = -shoulder_lift - elbow_flex + pitch

            print(f"\n  IK → shoulder_lift={shoulder_lift:.2f}°  "
                  f"elbow_flex={elbow_flex:.2f}°  "
                  f"wrist_flex={wrist_flex:.2f}°")
            print("  Moving …")

            p_control_to_target(robot, {
                "shoulder_pan":  pan,
                "shoulder_lift": shoulder_lift,
                "elbow_flex":    elbow_flex,
                "wrist_flex":    wrist_flex,
                "wrist_roll":    wrist_roll,
                "gripper":       gripper,
            })

        print("\nExiting — returning to zero position …")
        move_to_zero(robot)

    except KeyboardInterrupt:
        print("\nCtrl+C — returning to zero position …")
        try:
            move_to_zero(robot)
        except Exception:
            pass

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        robot.disconnect()
        print("Disconnected. Goodbye.")


if __name__ == "__main__":
    main()
