#!/usr/bin/env python3
"""
GB02_so101_triangulate_test.py

Pure validation of GB01's calibration output -- no robot motion, no YOLO.
Loads 2+ cameraN_to_robot.npy transforms (from GB01), shows a live snapshot
from each selected camera, lets you click the SAME physical point in each
view, triangulates it to a 3D point in the SO-101's own base frame, and
prints the result so you can sanity-check it against a ruler/known
reference before trusting it in GB03/GB04's real control loop.

Click the point in each camera window in turn (one click per window, window
closes automatically after the click -- same UX as
GB_code/GB02_so101_2Dcalibration.py's wait_for_click()).

This only needs camera INTRINSICS (saved by GB01 as cameraN_intrinsics.npy)
plus the camera-to-robot transforms -- no board, no robot connection.
"""

import traceback

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

CAMERAS = {
    "top": dict(index="/dev/video2", width=640, height=480, fps=30, fourcc="MJPG"),
    "right": dict(index="/dev/video4", width=640, height=480, fps=30, fourcc=None),
    "left": dict(index="/dev/video6", width=640, height=480, fps=30, fourcc=None),
}

WARMUP_FRAMES = 5


def open_camera(name):
    cfg = CAMERAS[name]
    kwargs = dict(index_or_path=cfg["index"], width=cfg["width"], height=cfg["height"], fps=cfg["fps"])
    if cfg["fourcc"]:
        kwargs["fourcc"] = cfg["fourcc"]
    cam = OpenCVCamera(OpenCVCameraConfig(**kwargs))
    cam.connect()
    return cam


def capture_snapshot(cam, warmup=WARMUP_FRAMES):
    for _ in range(warmup):
        cam.read()
    return cam.read()  # RGB


def wait_for_click(frame_rgb, camera_name):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)
    ax.set_title(f"'{camera_name}' -- click the target point, then close/confirm", fontsize=12)
    ax.axis("off")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise KeyboardInterrupt(f"No click recorded for camera '{camera_name}'.")
    return pts[0]


def undistort_point(u, v, camera_matrix, dist_coeffs):
    """Pixel -> normalized camera-frame ray direction (x, y, 1)."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x, y = undistorted[0, 0]
    return np.array([x, y, 1.0])


def triangulate_dlt(rays_and_origins):
    """Least-squares intersection of N rays in a common (robot) frame.
    Each entry: (origin (3,), direction (3,) -- not required unit length)."""
    A = []
    b = []
    for origin, direction in rays_and_origins:
        d = direction / np.linalg.norm(direction)
        # Projector onto the plane perpendicular to d: (I - d d^T)
        P = np.eye(3) - np.outer(d, d)
        A.append(P)
        b.append(P @ origin)
    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)
    point, *_ = np.linalg.lstsq(A, b, rcond=None)
    return point


def main():
    print("GB02 -- Triangulation test (no robot, no YOLO)")
    print("=" * 50)

    names_raw = input(f"Cameras to use (comma-separated, >=2, from {list(CAMERAS.keys())}): ").strip()
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if len(names) < 2:
        print("Need at least 2 cameras.")
        return

    transforms = {}
    intrinsics = {}
    for name in names:
        try:
            transforms[name] = np.load(f"camera_{name}_to_robot.npy")
            data = np.load(f"camera_{name}_intrinsics.npy", allow_pickle=True).item()
            intrinsics[name] = (data["camera_matrix"], data["dist_coeffs"])
        except FileNotFoundError as e:
            print(f"Missing calibration file for '{name}': {e}. Run GB01 for this camera first.")
            return

    cams = {name: open_camera(name) for name in names}

    try:
        while True:
            cmd = input("\nPress Enter to triangulate a point ('q' to quit): ").strip().lower()
            if cmd == "q":
                break

            rays = []
            for name in names:
                frame_rgb = capture_snapshot(cams[name])
                u, v = wait_for_click(frame_rgb, name)
                camera_matrix, dist_coeffs = intrinsics[name]
                ray_cam = undistort_point(u, v, camera_matrix, dist_coeffs)

                cam_to_robot = transforms[name]
                origin_robot = cam_to_robot[:3, 3]
                direction_robot = cam_to_robot[:3, :3] @ ray_cam
                rays.append((origin_robot, direction_robot))
                print(f"  '{name}': clicked pixel ({u:.0f}, {v:.0f})")

            point_robot = triangulate_dlt(rays)
            print(f"\n  Triangulated point in robot base frame (meters): "
                  f"x={point_robot[0]:.4f}  y={point_robot[1]:.4f}  z={point_robot[2]:.4f}")
            print("  Compare against a ruler/known reference before trusting this.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        for cam in cams.values():
            cam.disconnect()
        plt.close("all")
        print("Done.")


if __name__ == "__main__":
    main()
