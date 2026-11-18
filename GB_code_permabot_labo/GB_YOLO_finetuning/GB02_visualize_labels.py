#!/usr/bin/env python3
"""
GB02_visualize_labels.py

Workflow this fits into: GB02_see_weeds.py (classical detector) turned out
not to generalize across lighting conditions, so labeling is done by hand
instead, directly in Roboflow -- upload data/raw_images/, draw boxes for
every weed/crop yourself, then generate + download a dataset version from
Roboflow (YOLO11 or YOLOv8 export format) into this folder's data/
directory. ONLY ONCE that's downloaded does this script become useful.

Read-only viewer for that downloaded Roboflow-exported YOLO dataset
(DATASET_FOLDER_NAME below, e.g.
"Crop and Weed Detection.v1-crop-weed-baseline.yolov11"). Lets you pick
which split to look at (train/valid/test), then steps through every image
in that split showing its ground-truth boxes -- same red=weed,
other-colour=everything-else convention as GB02_see_weeds.py.

Class names and which splits exist are read directly from data.yaml and the
export's folder structure -- nothing hardcoded, so this keeps working
regardless of what you named your classes in Roboflow.

Usage: press Enter to advance to the next image, 'q' to quit early.
"""

import os

import cv2
import yaml
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Which downloaded Roboflow export to view -- the folder name under data/
# (exactly as Roboflow names it on download, e.g. "<project>.v<N>-<version-name>.<format>").
# Change this whenever you want to look at a different dataset version/export.
DATASET_FOLDER_NAME = "Crop and Weed Detection.v1-crop-weed-baseline.yolov11"


def find_dataset_root():
    """Resolve DATASET_FOLDER_NAME to a real path and confirm it has a data.yaml."""
    dataset_root = os.path.join(DATA_DIR, DATASET_FOLDER_NAME)
    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"No data.yaml found at {yaml_path} -- check DATASET_FOLDER_NAME matches "
            f"a Roboflow export actually downloaded into {DATA_DIR}."
        )
    return dataset_root


def load_dataset_config(dataset_root):
    with open(os.path.join(dataset_root, "data.yaml")) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    # Roboflow's data.yaml writes train/val/test as paths like "../train/images",
    # relative to a base directory that isn't simply the yaml's own folder (resolving
    # that literally lands one level too high). Since the export's actual on-disk
    # layout is always <dataset_root>/<split>/images + /labels as direct
    # subfolders, check those directly instead of trusting the yaml's path strings.
    splits = {}
    for split_key in ["train", "valid", "test"]:
        images_dir = os.path.join(dataset_root, split_key, "images")
        labels_dir = os.path.join(dataset_root, split_key, "labels")
        if os.path.isdir(images_dir):
            splits[split_key] = (images_dir, labels_dir)
    return class_names, splits


def color_for_class(name):
    return "red" if "weed" in name.lower() else "deepskyblue"


def load_yolo_labels(label_path, img_w, img_h, class_names):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, xc, yc, w, h = line.split()
            class_id = int(class_id)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
            box_w = w * img_w
            box_h = h * img_h
            x = xc * img_w - box_w / 2
            y = yc * img_h - box_h / 2
            boxes.append({
                "classification": class_names[class_id],
                "bbox": (x, y, box_w, box_h),
            })
    return boxes


def show_image_with_boxes(frame_bgr, boxes, filename, index, total):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(frame_rgb)

    for b in boxes:
        x, y, w, h = b["bbox"]
        color = color_for_class(b["classification"])
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=2, edgecolor=color, facecolor='none'))
        ax.text(x, y - 4, b["classification"], color=color, fontsize=7,
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))

    ax.set_title(f"[{index}/{total}] {filename}  --  {len(boxes)} box(es)", fontsize=11)
    ax.axis('off')
    plt.show()
    plt.close(fig)


def main():
    print("GB02 -- Visualize Labels (Roboflow export, read-only review)")
    print("=" * 50)

    dataset_root = find_dataset_root()
    class_names, splits = load_dataset_config(dataset_root)
    print(f"Dataset: {dataset_root}")
    print(f"Classes: {class_names}")
    print(f"Available splits: {list(splits.keys())}\n")

    while True:
        choice = input(f"Which split do you want to view? [{'/'.join(splits.keys())}] (q to quit): ").strip().lower()
        if choice == 'q':
            return
        if choice in splits:
            break
        print(f"  Unknown split '{choice}'. Choose one of: {list(splits.keys())}")

    images_dir, labels_dir = splits[choice]
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        print(f"No images found in {images_dir}.")
        return

    print(f"\n{len(image_files)} image(s) in '{choice}'. Press Enter to advance, 'q' to quit early.\n")

    totals = {name: 0 for name in class_names}
    zero_box_images = []

    for i, filename in enumerate(image_files, start=1):
        frame = cv2.imread(os.path.join(images_dir, filename))
        img_h, img_w = frame.shape[:2]
        stem = os.path.splitext(filename)[0]
        boxes = load_yolo_labels(os.path.join(labels_dir, f"{stem}.txt"), img_w, img_h, class_names)

        if not boxes:
            zero_box_images.append(filename)
        for b in boxes:
            totals[b["classification"]] += 1

        cmd = input(f"[{i}/{len(image_files)}] {filename} "
                    f"({len(boxes)} box(es)) -- Enter to view, 'q' to quit: ").strip().lower()
        if cmd == 'q':
            break

        show_image_with_boxes(frame, boxes, filename, i, len(image_files))

    print("\n" + "=" * 50)
    print(f"Split '{choice}' summary:")
    print(f"  Total boxes: {totals}")
    print(f"  Images with ZERO boxes ({len(zero_box_images)}): {zero_box_images}")


if __name__ == "__main__":
    main()
