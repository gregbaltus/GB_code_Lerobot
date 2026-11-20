# YOLO fine-tuning pipeline

| Step | Script | Where |
|------|--------|-------|
| 1. Capture raw images | `GB01_capture_images.py` | local (lerobot env) |
| 2. Label + export | Roboflow (YOLOv11 format), unzip into `data/` | browser |
| 3. Check labels | `GB02_visualize_labels.py` | local |
| 4. Push dataset to Hub | `GB03_push_dataset_to_hub.py` | local |
| 5. Fine-tune + push model | `GB04_finetune_yolo.py` | GPU machine |
| 6. Test on held-out split | `GB05_test_yolo.py` | local (after `git clone` of the model repo) |

Run local scripts with `/home/gregory/miniconda3/envs/lerobot/bin/python` (the base `python3` has no `cv2`).

## Checklist: new dataset + new YOLO

Change these variables **before** running each script, otherwise you mix or overwrite a previous dataset/model.

| Script | Variable | Set to |
|--------|----------|--------|
| GB01 | `CAMERA_OAK` | `True` = OAK-D Pro (1280x800 RGB), `False` = webcam `/dev/video2` (640x480) |
| GB01 | `RAW_DATASET_NAME` | a **new** folder name, e.g. `raw_oakd_v2` → images go to `data/<name>/` |
| GB02, GB03 | `DATASET_FOLDER_NAME` | the new Roboflow export folder name in `data/` |
| GB03, GB04, GB05 | `DATASET_REPO_ID` | a **new** HF dataset repo (same value in all three) |
| GB04 | `MODEL_REPO_ID` | a **new** HF model repo |
| GB04 | `RUN_NAME` | a new run name |
| GB05 | `MODEL_PATH` | `weights/best.pt` inside the new cloned model repo |

Notes:
- Re-using an existing `RAW_DATASET_NAME` *adds* images to that dataset (numbering resumes) — that's the way to grow a dataset, not to start a new one.
- Inference must use the same camera as the training data (a webcam YOLO for the webcam, an OAK-D YOLO for the OAK-D).

## Existing datasets / models

| Camera | Raw images | Roboflow export | HF dataset | HF model |
|--------|-----------|-----------------|------------|----------|
| Webcam `/dev/video2` | `data/raw_webcam_v1/` | `Crop and Weed Detection.v1-crop-weed-baseline.yolov11` | `Greynar/PermaBotLaboWeedDetectionYOLO` | `Greynar/yolo_PermaBotLaboWeedDetection` (local clone: `yolo_PermaBotLaboWeedDetection_pretest/`) |
| OAK-D Pro | `data/raw_oakd_v1/` | `Crop and Weed Detecction OAK-D.v1i.yolov11` | `Greynar/PermaBotLaboWeedDetectionYOLO_oakd_v1` | `Greynar/yolo_PermaBotLaboWeedDetection_oakd_v1` (local clone: `yolo_PermaBotLaboWeedDetection_oakd_v1/`) |
