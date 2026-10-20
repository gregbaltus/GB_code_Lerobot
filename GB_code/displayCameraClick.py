#!/usr/bin/env python3
"""
displayCameraClick.py
Captures a snapshot from the camera, displays it with matplotlib,
and lets you click to get pixel coordinates printed in the terminal.
Click on the image, then close the window to record and continue.
Press Ctrl+C or type 'n' when asked to stop.
"""

import cv2
import matplotlib
matplotlib.use("TkAgg")   # reliable backend for interactive clicks
import matplotlib.pyplot as plt


def capture_frame(cap, warmup=5):
    """Read a fresh frame, discarding a few stale ones first."""
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    # List available cameras
    available = []
    for i in range(5):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            available.append(i)
            test.release()

    if not available:
        print("No cameras found.")
        return

    print(f"Available cameras: {available}")
    cam_idx = int(input("Select camera index: ").strip())

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("\nFor each round: a snapshot opens, click on your point, then close the window.")
    print("Type 'n' when asked to stop recording.\n")

    clicked_points = []

    while True:
        frame_rgb = capture_frame(cap)
        if frame_rgb is None:
            print("Failed to read frame from camera.")
            break

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(frame_rgb)

        # Draw already recorded points
        for i, (u, v) in enumerate(clicked_points):
            ax.plot(u, v, 'go', markersize=8)
            ax.annotate(str(i + 1), (u + 6, v - 6),
                        color='lime', fontsize=11, fontweight='bold')

        n = len(clicked_points) + 1
        ax.set_title(f"Point {n} — click on your target, then close this window",
                     fontsize=13)
        ax.axis('off')

        # ginput waits for exactly 1 click, timeout=0 means wait forever
        pts = plt.ginput(1, timeout=0)
        plt.close(fig)

        if not pts:
            print("No click recorded (window closed without clicking).")
        else:
            u, v = int(round(pts[0][0])), int(round(pts[0][1]))
            clicked_points.append((u, v))
            print(f"  Recorded point {n}: u={u}  v={v}")

        again = input("Record another point? (y/n): ").strip().lower()
        if again != 'y':
            break

    cap.release()
    plt.close('all')

    print(f"\nAll recorded points ({len(clicked_points)}):")
    for i, (u, v) in enumerate(clicked_points):
        print(f"  {i+1}: u={u}  v={v}")


if __name__ == "__main__":
    main()
