#!/usr/bin/env python3
"""
GB03_push_dataset_to_hub.py

Pushes the downloaded Roboflow-exported YOLO dataset (DATASET_FOLDER_NAME
under data/, e.g. "Crop and Weed Detection.v1-crop-weed-baseline.yolov11")
to a Hugging Face Hub dataset repo, as-is -- same images/ + labels/ +
data.yaml structure, no reformatting. This mirrors how every other dataset
in this project lives on the Hub (just a generic file store here, not a
LeRobot-format dataset), so the fine-tuning script (run on a separate GPU
machine) can pull it down via DATASET_REPO_ID instead of needing the raw
files copied over by hand.

Usage: python3 GB03_push_dataset_to_hub.py
"""

import os

from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Which local export to push -- same DATASET_FOLDER_NAME convention as
# GB02_visualize_labels.py, so this stays consistent if you have multiple
# Roboflow export versions later.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_FOLDER_NAME = "Crop and Weed Detection.v1-crop-weed-baseline.yolov11"

# ---------------------------------------------------------------------------
# Hub destination -- edit before running if you want a different name/visibility
# ---------------------------------------------------------------------------
DATASET_REPO_ID = "Greynar/PermaBotLaboWeedDetectionYOLO"
PRIVATE = True


def main():
    print("GB03 -- Push YOLO Dataset to Hub")
    print("=" * 50)

    dataset_root = os.path.join(DATA_DIR, DATASET_FOLDER_NAME)
    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"No data.yaml found at {yaml_path} -- check DATASET_FOLDER_NAME "
              f"matches a Roboflow export actually downloaded into {DATA_DIR}.")
        return

    print(f"Local dataset: {dataset_root}")
    print(f"Hub destination: {DATASET_REPO_ID} (private={PRIVATE})\n")

    api = HfApi()
    api.create_repo(DATASET_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    print("Uploading (this may take a little while for all the images)...")
    api.upload_folder(
        folder_path=dataset_root,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )

    print(f"\nDone. Dataset is at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
