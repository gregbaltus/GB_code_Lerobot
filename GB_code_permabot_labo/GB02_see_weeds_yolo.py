#!/usr/bin/env python3
"""
GB02_see_weeds_yolo.py

Same role as GB02_see_weeds.py (vision-only: snapshot the soil bed, detect
weeds vs cultivable plants, display boxed results) but using the fine-tuned
YOLO model (GB_YOLO_finetuning/GB04_finetune_yolo.py) instead of the
classical HSV + blob-area detector -- which was shown not to generalize
across lighting conditions, the whole reason this exists.

MODEL_DIR points at the model's FOLDER (e.g. a git-cloned Hub repo, like
"yolo_PermaBotLaboWeedDetection_pretest"), not the .pt file directly -- the
script looks for weights/best.pt inside it. Change this constant whenever
you have a newer/better fine-tuned model to swap in.

Class names (e.g. "weed"/"crop") come directly from the model itself
(model.names), not hardcoded, so this keeps working if you retrain with
different class names later.

Frames are passed to the model as raw BGR (straight from cv2, no RGB
conversion) -- same as GB_YOLO_finetuning/GB05_test_yolo.py, which is
already confirmed to give good results this way. RGB conversion only
happens for the matplotlib display.

No robot, no homography, no movement -- just detection + display.

Usage: press Enter to take a snapshot and detect, 'q' to quit.
"""

import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model -- points at the model's FOLDER (not the .pt file directly); looks
# for weights/best.pt inside it. Change this to swap in a newer model.
# ---------------------------------------------------------------------------
MODEL_DIR = "yolo_PermaBotLaboWeedDetection_pretest"

CONFIDENCE_THRESHOLD = 0.25  # Ultralytics' own default -- lower = more (possibly false) detections

# ---------------------------------------------------------------------------
# Camera -- identical setup to GB02_see_weeds.py
# ---------------------------------------------------------------------------
CAMERA_INDEX  = "/dev/video2"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
WARMUP_FRAMES = 5


def color_for_class(name):
    return "red" if "weed" in name.lower() else "deepskyblue"


def find_plant_blobs_yolo(frame_bgr, model):
    """Run YOLO on frame_bgr, return detections in the same
    {"center", "bbox", "classification"} format GB02_see_weeds.py's
    find_plant_blobs() uses."""
    results = model.predict(source=frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        classification = model.names[class_id]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        detections.append({
            "center": (cx, cy),
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "classification": classification,
            "confidence": confidence,
        })

    return detections


def capture_snapshot(cap, warmup=WARMUP_FRAMES):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


def show_detections(frame_bgr, detections):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    for d in detections:
        x, y, w, h = d["bbox"]
        color = color_for_class(d["classification"])
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=2, edgecolor=color, facecolor='none'))
        label = f"{d['classification']} {d['confidence']:.2f}"
        ax.text(x, y - 6, label, color=color, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))

    n_weed = sum(1 for d in detections if "weed" in d["classification"].lower())
    n_other = len(detections) - n_weed
    ax.set_title(f"Found {n_weed} weed(s) [red], {n_other} other [blue]", fontsize=13)
    ax.axis('off')

    plt.show()
    plt.close(fig)


def main():
    print("GB02 -- See Weeds (YOLO)")
    print("=" * 50)

    checkpoint_path = os.path.join(MODEL_DIR, "weights", "best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path} -- check MODEL_DIR.")
        return

    print(f"Loading model: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    print(f"Classes: {model.names}\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}.")
        return

    print("Press Enter to take a snapshot and detect. Type 'q' to quit.\n")

    try:
        while True:
            cmd = input("Detect> ").strip()
            if cmd.lower() == 'q':
                break

            frame_bgr = capture_snapshot(cap)
            detections = find_plant_blobs_yolo(frame_bgr, model)

            if not detections:
                print("  Nothing detected. Try again.\n")
                continue

            print(f"  Found {len(detections)} detection(s)")
            for d in sorted(detections, key=lambda d: d["classification"]):
                print(f"    {d['classification']:<11} conf={d['confidence']:.2f} center={d['center']}")

            show_detections(frame_bgr, detections)
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
