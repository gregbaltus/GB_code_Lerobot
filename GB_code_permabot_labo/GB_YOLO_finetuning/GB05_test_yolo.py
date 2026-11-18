#!/usr/bin/env python3
"""
GB05_test_yolo.py

Tests the fine-tuned YOLO checkpoint (from GB04_finetune_yolo.py) against
the held-out TEST split of the dataset -- never touched during training or
validation, so this is the real, honest check of how well the model
generalizes.

Two things:
1. Quantitative stats (precision/recall/mAP per class) via Ultralytics'
   built-in model.val(split="test") -- no custom metric code needed.
   (Metric accessor names below -- class_result(i), mp/mr/map50/map --
   were verified directly against the installed ultralytics source before
   writing this, not guessed.)
2. Qualitative step-through viewer: for each test image, shows ground-truth
   boxes side-by-side with the model's own predictions, so you can see
   directly which weeds it missed/found, which crops it confused, etc.

MODEL_PATH expects a LOCAL checkpoint file -- same git-clone-from-the-Hub
workflow you already use for ACT checkpoints (GB04 pushes the trained run
to MODEL_REPO_ID at the end of training; git clone that here, then point
MODEL_PATH at the resulting weights/best.pt).

Usage: python3 GB05_test_yolo.py -- press Enter to advance through the
       qualitative viewer, 'q' to quit early.
"""

import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from huggingface_hub import snapshot_download
from ultralytics import YOLO

from GB02_visualize_labels import color_for_class, load_dataset_config, load_yolo_labels

# ---------------------------------------------------------------------------
# Model -- a LOCAL path, same git-clone-from-Hub workflow as your ACT
# checkpoints. Edit this once you've cloned GB04's pushed model repo
# (git clone https://huggingface.co/Greynar/yolo_PermaBotLaboWeedDetection).
# ---------------------------------------------------------------------------
MODEL_PATH = "yolo_PermaBotLaboWeedDetection_pretest/weights/best.pt"

# ---------------------------------------------------------------------------
# Dataset -- same Hub repo GB04 trained from; downloaded fresh here too
# (cheap/cached after the first call) so this test is self-contained.
# ---------------------------------------------------------------------------
DATASET_REPO_ID = "Greynar/PermaBotLaboWeedDetectionYOLO"

CONFIDENCE_THRESHOLD = 0.25  # Ultralytics' own default -- lower = more (possibly false) detections


def predictions_to_boxes(result, class_names):
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        boxes.append({
            "classification": class_names[class_id],
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "confidence": conf,
        })
    return boxes


def show_comparison(frame_bgr, gt_boxes, pred_boxes, filename, index, total):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, boxes, title in [(ax_gt, gt_boxes, "Ground truth"), (ax_pred, pred_boxes, "Predictions")]:
        ax.imshow(frame_rgb)
        for b in boxes:
            x, y, w, h = b["bbox"]
            color = color_for_class(b["classification"])
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))
            label = b["classification"]
            if "confidence" in b:
                label += f" {b['confidence']:.2f}"
            ax.text(x, y - 4, label, color=color, fontsize=7, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))
        ax.set_title(f"{title} ({len(boxes)})", fontsize=11)
        ax.axis('off')

    fig.suptitle(f"[{index}/{total}] {filename}", fontsize=12)
    plt.show()
    plt.close(fig)


def main():
    print("GB05 -- Test YOLO (held-out test split)")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"No checkpoint found at {MODEL_PATH} -- edit MODEL_PATH to wherever you "
              f"cloned the trained model from the Hub "
              f"(git clone https://huggingface.co/Greynar/yolo_PermaBotLaboWeedDetection).")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Downloading dataset '{DATASET_REPO_ID}' from the Hub ...")
    dataset_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset")
    class_names, splits = load_dataset_config(dataset_dir)
    if "test" not in splits:
        print("No 'test' split found in this dataset.")
        return
    images_dir, labels_dir = splits["test"]
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    print(f"Dataset ready at: {dataset_dir}\n")

    # 1. Quantitative stats on the test set
    print("Running quantitative evaluation on the test split ...")
    metrics = model.val(data=data_yaml, split="test")
    print("\nPer-class results:")
    for i, name in enumerate(class_names):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"  {name:<10} precision={p:.3f}  recall={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}")
    print(f"  Overall   precision={metrics.box.mp:.3f}  recall={metrics.box.mr:.3f}  "
          f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}\n")

    # 2. Qualitative step-through viewer
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(image_files)} test image(s). Press Enter to view, 'q' to quit.\n")

    for i, filename in enumerate(image_files, start=1):
        frame = cv2.imread(os.path.join(images_dir, filename))
        img_h, img_w = frame.shape[:2]
        stem = os.path.splitext(filename)[0]
        gt_boxes = load_yolo_labels(os.path.join(labels_dir, f"{stem}.txt"), img_w, img_h, class_names)

        result = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        pred_boxes = predictions_to_boxes(result, class_names)

        cmd = input(f"[{i}/{len(image_files)}] {filename} "
                    f"(gt={len(gt_boxes)}, pred={len(pred_boxes)}) -- Enter to view, 'q' to quit: ").strip().lower()
        if cmd == 'q':
            break

        show_comparison(frame, gt_boxes, pred_boxes, filename, i, len(image_files))

    print("\nDone.")


if __name__ == "__main__":
    main()
