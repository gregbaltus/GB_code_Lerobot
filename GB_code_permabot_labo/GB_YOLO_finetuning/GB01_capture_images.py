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

CAMERA SWITCH: set CAMERA_OAK below.
  - False -> plain webcam (/dev/video2, 640x480) via OpenCV
  - True  -> OAK-D Pro RGB stream (CAM_A, 1280x800, RGB only -- no depth) via
             depthai (same API usage as GB_code_RGBD/GB01_oakd_see_streams.py)

OUTPUT FOLDER: set RAW_DATASET_NAME below -- images go to data/<RAW_DATASET_NAME>/.
Use a NEW name for every new dataset / new YOLO (see README.md). Re-using an
existing name appends to that dataset (numbering resumes where it stopped).
Existing raw datasets:
  - data/raw_webcam_v1/  -> webcam images behind Roboflow export
                            "Crop and Weed Detection.v1-crop-weed-baseline.yolov11"

Usage: press Enter to capture (shows a preview you can accept or retake),
       'r' to retake the last preview, 'q' to quit.
"""

import glob
import os
import time

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Camera switch -- True = OAK-D Pro (depthai), False = standard webcam (OpenCV)
# ---------------------------------------------------------------------------
CAMERA_OAK = True

# Webcam -- identical setup to GB02_see_weeds.py
CAMERA_INDEX  = "/dev/video2"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
WARMUP_FRAMES = 5

# OAK-D -- RGB only, same resolution as GB_code_RGBD/IL_pipeline/GB02_oakd_two_camera_teleop.py
OAK_RGB_SIZE      = (1280, 800)
OAK_FPS           = 15
OAK_WARMUP_FRAMES = 30   # auto-exposure/focus need longer to converge on the OAK-D
OAK_TIMEOUT_S     = 10.0

# ---------------------------------------------------------------------------
# Output -- one folder per dataset: data/<RAW_DATASET_NAME>/
# Pick a NEW name for each new dataset so previous captures are never mixed in.
# ---------------------------------------------------------------------------
RAW_DATASET_NAME = "raw_oakd_v1"

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR      = os.path.join(BASE_DIR, "data", RAW_DATASET_NAME)
FILENAME_STEM   = "img"
FILENAME_DIGITS = 4


class WebcamCamera:
    name = f"webcam {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})"

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}.")

    def snapshot(self):
        for _ in range(WARMUP_FRAMES):
            self.cap.read()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def close(self):
        self.cap.release()


class OakCamera:
    name = f"OAK-D RGB CAM_A ({OAK_RGB_SIZE[0]}x{OAK_RGB_SIZE[1]})"

    def __init__(self):
        import depthai as dai  # imported lazily so webcam mode doesn't need depthai

        self.pipeline = dai.Pipeline(dai.Device())
        device = self.pipeline.getDefaultDevice()
        print(f"OAK-D connected: {device.getDeviceName()}  USB speed: {device.getUsbSpeed().name}")

        rgb_cam = self.pipeline.create(dai.node.Camera)
        rgb_cam.setSensorType(dai.CameraSensorType.COLOR)
        rgb_cam.build(dai.CameraBoardSocket.CAM_A, sensorFps=OAK_FPS)
        cap = dai.ImgFrameCapability()
        cap.size.fixed(OAK_RGB_SIZE)
        cap.fps.fixed(OAK_FPS)
        self.queue = rgb_cam.requestOutput(cap, True).createOutputQueue(maxSize=4, blocking=False)
        self.pipeline.start()

        print(f"Discarding {OAK_WARMUP_FRAMES} warmup frames (auto-exposure/focus) ...")
        for _ in range(OAK_WARMUP_FRAMES):
            self._get_frame()

    def _get_frame(self):
        start = time.time()
        while time.time() - start < OAK_TIMEOUT_S:
            pkt = self.queue.tryGet()
            if pkt is not None:
                return pkt.getCvFrame()  # BGR, same convention as cv2.VideoCapture
            time.sleep(0.01)
        raise RuntimeError("Timed out waiting for an OAK-D RGB frame.")

    def snapshot(self):
        # The pipeline keeps streaming while you look at the preview, so drop
        # the stale buffered frames and return a fresh one.
        self.queue.tryGetAll()
        self._get_frame()
        return self._get_frame()

    def close(self):
        self.pipeline.stop()


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

    try:
        camera = OakCamera() if CAMERA_OAK else WebcamCamera()
    except Exception as e:
        print(f"Could not open camera: {e}")
        return
    print(f"Camera: {camera.name}")

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

            frame = camera.snapshot()

            while True:
                show_preview(frame, index, total_saved)
                decision = input("  Keep this one? [Enter=keep, r=retake, q=quit]: ").strip().lower()
                if decision == 'q':
                    return
                elif decision == 'r':
                    print("  Retaking...")
                    frame = camera.snapshot()
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
        camera.close()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
