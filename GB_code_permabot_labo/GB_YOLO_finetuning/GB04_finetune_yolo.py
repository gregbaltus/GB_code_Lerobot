#!/usr/bin/env python3
"""
GB04_finetune_yolo.py

Fine-tunes a small YOLO model (base checkpoint auto-downloaded by
Ultralytics) on the labeled weed/crop dataset pushed to the Hub by
GB03_push_dataset_to_hub.py.

Meant to run on a separate GPU machine -- copy just this one script over
(and make sure `ultralytics` + `huggingface_hub` are installed there), no
need to manually transfer the dataset: this script downloads it from the
Hub itself via DATASET_REPO_ID, same pattern as every LeRobotDataset(...)
call elsewhere in this project. Ultralytics correctly resolves Roboflow's
data.yaml path convention on its own (verified directly against the real
downloaded export before writing this), so no path patching needed here.

At the end, pushes the training run's full output folder (weights +
results.csv/plots/args.yaml) to a Hub model repo, so the trained checkpoint
can be retrieved on your local machine the same way you already do for ACT
checkpoints (git clone) -- no separate download script needed.

Current defaults are a FAST SMOKE TEST, not a real training run -- this
first dataset is only ~30 images (21 train), so the point right now is
confirming the whole pipeline (Hub download -> train -> push) works
end-to-end before investing in the full 100-300 image dataset. Scale up
EPOCHS/BATCH_SIZE once you have more data.

Usage: python3 GB04_finetune_yolo.py
"""

from huggingface_hub import HfApi, snapshot_download
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Dataset -- downloaded from the Hub at runtime, not manually copied over
# ---------------------------------------------------------------------------
DATASET_REPO_ID = "Greynar/PermaBotLaboWeedDetectionYOLO_oakd_v1"

# ---------------------------------------------------------------------------
# Base checkpoint -- "nano", smallest/fastest YOLO11 variant, a good fit for
# a small 2-class detector in a controlled setting. Auto-downloaded by
# Ultralytics on first use if not already cached locally.
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"

# ---------------------------------------------------------------------------
# Training -- fast smoke-test defaults for this small ~30-image dataset.
# Scale up (more epochs, bigger batch) once trained on the full dataset.
# ---------------------------------------------------------------------------
EPOCHS     = 50
BATCH_SIZE = 8
IMAGE_SIZE = 640
RUN_NAME   = "weed_detection_oakd_v1"  # <- bump this each run so Ultralytics doesn't overwrite past runs

# ---------------------------------------------------------------------------
# Hub destination for the trained checkpoint -- edit before running if you
# want a different name/visibility
# ---------------------------------------------------------------------------
MODEL_REPO_ID = "Greynar/yolo_PermaBotLaboWeedDetection_oakd_v1"
PRIVATE       = True


def main():
    print("GB04 -- Fine-tune YOLO")
    print("=" * 50)

    print(f"Downloading dataset '{DATASET_REPO_ID}' from the Hub ...")
    dataset_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset")
    data_yaml = f"{dataset_dir}/data.yaml"
    print(f"Dataset ready at: {dataset_dir}\n")

    print(f"Loading base model '{BASE_MODEL}' ...")
    model = YOLO(BASE_MODEL)

    print(f"Training for {EPOCHS} epochs (batch={BATCH_SIZE}, imgsz={IMAGE_SIZE}) ...\n")
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        name=RUN_NAME,
    )

    save_dir = model.trainer.save_dir
    best_ckpt = model.trainer.best
    print(f"\nTraining finished. Run output: {save_dir}")
    print(f"Best checkpoint: {best_ckpt}")

    print(f"\nPushing results to Hub: {MODEL_REPO_ID} (private={PRIVATE}) ...")
    api = HfApi()
    api.create_repo(MODEL_REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"\nDone. Model is at: https://huggingface.co/{MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
