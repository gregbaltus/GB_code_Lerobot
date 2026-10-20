#!/usr/bin/env python3
"""
GB01_capture_images.py

First step of the YOLO fine-tuning pipeline: capture raw training images of
the soil bed from the "top" camera (/dev/video2) -- the same camera
GB_code_permabot_labo/GB02_see_weeds.py already uses for weed/cultivable
detection, and the only camera the fine-tuned YOLO model will ever actually
run on (see that discussion: training data should match the deployment
distribution -- left/right aren't used here since they're a very different
scale/framing and the model will never see them at inference time).

No robot connection -- pure vision tool, matching GB02_see_weeds.py's camera
setup. The arm is deliberately kept out of frame for all these captures,
since in the current pipeline YOLO only ever runs in the pre-ACT detection
window, before the arm moves -- see that discussion for why training
exclusively on arm-free frames is the correct match, not a compromise.

Since the goal is robustness to lighting/arrangement changes (not just raw
volume), run this across multiple separate sessions, deliberately varying
lighting and plant positions between sessions rather than capturing many
near-identical frames in one sitting. Image numbering resumes automatically
across sessions, so you can run this script as many times as you like.

Usage: press Enter to capture (shows a preview you can accept or retake),
       'r' to retake the last preview, 'q' to quit.
"""

import glob
import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Camera -- identical setup to GB02_see_weeds.py
# ---------------------------------------------------------------------------
CAMERA_INDEX  = "/dev/video2"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
WARMUP_FRAMES = 5

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_images")
FILENAME_STEM  = "img"
FILENAME_DIGITS = 4


def capture_snapshot(cap, warmup=WARMUP_FRAMES):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def next_index(output_dir, stem):
    """Scan existing files to resume numbering across sessions instead of
    overwriting previous captures."""
    existing = glob.glob(os.path.join(output_dir, f"{stem}_*.jpg"))
    if not existing:
        return 1
    indices = []
    for path in existing:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            indices.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return (max(indices) + 1) if indices else 1


def show_preview(frame_bgr, index, total_saved):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(frame_rgb)
    ax.set_title(
        f"Preview: {FILENAME_STEM}_{index:0{FILENAME_DIGITS}d}.jpg  "
        f"({total_saved} saved so far)\n"
        f"Close this window, then: Enter=keep, r=retake, q=quit",
        fontsize=11,
    )
    ax.axis('off')
    plt.show()
    plt.close(fig)


def main():
    print("GB01 -- Capture Images (YOLO fine-tuning dataset)")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}.")
        return

    index = next_index(OUTPUT_DIR, FILENAME_STEM)
    total_saved = index - 1
    print(f"Saving to: {OUTPUT_DIR}")
    print(f"Resuming at {FILENAME_STEM}_{index:0{FILENAME_DIGITS}d}.jpg "
          f"({total_saved} image(s) already captured in previous sessions)")
    print("Press Enter to capture. Type 'q' to quit.\n")
    print("Reminder: vary lighting/plant arrangement between sessions, "
          "keep the arm out of frame.\n")

    try:
        while True:
            cmd = input(f"Capture #{index}> ").strip()
            if cmd.lower() == 'q':
                break

            frame = capture_snapshot(cap)

            while True:
                show_preview(frame, index, total_saved)
                decision = input("  Keep this one? [Enter=keep, r=retake, q=quit]: ").strip().lower()
                if decision == 'q':
                    print("Done.")
                    cap.release()
                    plt.close('all')
                    return
                elif decision == 'r':
                    print("  Retaking...")
                    frame = capture_snapshot(cap)
                    continue
                else:
                    break

            filename = f"{FILENAME_STEM}_{index:0{FILENAME_DIGITS}d}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(path, frame)
            total_saved += 1
            index += 1
            print(f"  Saved {filename} ({total_saved} total)\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
