#!/usr/bin/env python3
"""
GB01_oakd_see_streams.py

First test of the OAK-D RGB-D camera: confirms it's reachable and pulls one
frame from each of its two relevant output streams --

  - RGB: the 12MP color sensor (CAM_A / IMX378)
  - DEPTH: metric depth in millimeters, computed on-device by the
    StereoDepth node from the two mono cameras (CAM_B/CAM_C, OV9282 stereo
    pair) -- this is what "the distance" actually is on this camera: not a
    separate dedicated depth sensor, but a depth MAP computed from stereo
    matching between the two mono cameras, output already rectified/aligned
    and in real-world units (mm) by the device itself.

Not a /dev/videoN device like the other cameras in this project -- OAK-D
talks over USB via the DepthAI SDK/API, not plain UVC, so it needs the
`depthai` package (confirmed installed in the `oak` conda env, NOT the
`lerobot` env) and doesn't show up with plain v4l2 tools.

This confirms the device + both streams work before anything about
integrating it into lerobot/a robot pipeline is attempted.

Run with: /home/gregory/miniconda3/envs/oak/bin/python GB01_oakd_see_streams.py
"""

import time

import cv2
import depthai as dai
import numpy as np

OUT_RGB = "oakd_rgb.jpg"
OUT_DEPTH_VIS = "oakd_depth_colorized.jpg"
OUT_DEPTH_RAW = "oakd_depth_mm.npy"

RGB_SIZE = (1280, 800)   # downscaled from the sensor's native 4056x3040 -- plenty for inspection
FPS = 15
TIMEOUT_S = 20.0
WARMUP_FRAMES = 30  # discard this many of EACH stream first -- same convention as
                     # every other camera script in this project (auto-exposure/focus
                     # need time to converge; the very first frame is typically dark/blurry)


def main():
    print("GB01 -- OAK-D stream test")
    print("=" * 40)

    with dai.Pipeline(dai.Device()) as pipeline:
        device = pipeline.getDefaultDevice()
        print(f"Connected: {device.getDeviceName()}  USB speed: {device.getUsbSpeed().name}")

        cams = {c.socket: c for c in device.getConnectedCameraFeatures()}
        print("Cameras found:")
        for socket, c in cams.items():
            print(f"  {socket}: {c.sensorName} {c.width}x{c.height} {[t.name for t in c.supportedTypes]}")

        # --- RGB ---
        rgb_cam = pipeline.create(dai.node.Camera)
        rgb_cam.setSensorType(dai.CameraSensorType.COLOR)
        rgb_cam.build(dai.CameraBoardSocket.CAM_A, sensorFps=FPS)
        cap = dai.ImgFrameCapability()
        cap.size.fixed(RGB_SIZE)
        cap.fps.fixed(FPS)
        rgb_out = rgb_cam.requestOutput(cap, True)
        rgb_queue = rgb_out.createOutputQueue(maxSize=4, blocking=False)

        # --- Stereo depth (metric, mm) from the two mono cameras ---
        left_cam = pipeline.create(dai.node.Camera)
        left_cam.setSensorType(dai.CameraSensorType.MONO)
        left_cam.build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)

        right_cam = pipeline.create(dai.node.Camera)
        right_cam.setSensorType(dai.CameraSensorType.MONO)
        right_cam.build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        left_cam.requestFullResolutionOutput().link(stereo.left)
        right_cam.requestFullResolutionOutput().link(stereo.right)
        depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print(f"\nPipeline running, discarding {WARMUP_FRAMES} warmup frames per stream "
              f"(auto-exposure/focus need time to converge) ...")

        rgb_seen = depth_seen = 0
        rgb_frame = None
        depth_frame = None
        start = time.time()
        while time.time() - start < TIMEOUT_S and (rgb_frame is None or depth_frame is None):
            if rgb_frame is None:
                pkt = rgb_queue.tryGet()
                if pkt is not None:
                    rgb_seen += 1
                    if rgb_seen > WARMUP_FRAMES:
                        rgb_frame = pkt.getCvFrame()
                        print(f"  RGB frame received (after {rgb_seen} frames): {rgb_frame.shape}")
            if depth_frame is None:
                pkt = depth_queue.tryGet()
                if pkt is not None:
                    depth_seen += 1
                    if depth_seen > WARMUP_FRAMES:
                        depth_frame = pkt.getCvFrame()  # uint16, millimeters
                        print(f"  Depth frame received (after {depth_seen} frames): "
                              f"{depth_frame.shape} dtype={depth_frame.dtype}")
            time.sleep(0.05)

        pipeline.stop()

        if rgb_frame is None:
            print("Timed out waiting for an RGB frame.")
            return
        if depth_frame is None:
            print("Timed out waiting for a depth frame.")
            return

        cv2.imwrite(OUT_RGB, rgb_frame)
        print(f"\nSaved {OUT_RGB}")

        # depthai uses 0 AND 65535 as "invalid/no reading" sentinels, not just 0
        valid = depth_frame[(depth_frame > 0) & (depth_frame < 65535)]
        if valid.size == 0:
            print("Depth frame has no valid pixels -- check scene/lighting/distance.")
        else:
            print(f"Depth stats (mm, valid pixels only): "
                  f"min={valid.min()} max={valid.max()} mean={valid.mean():.0f} "
                  f"valid_fraction={valid.size / depth_frame.size:.2%}")

        np.save(OUT_DEPTH_RAW, depth_frame)
        print(f"Saved raw depth (mm, uint16) to {OUT_DEPTH_RAW}")

        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(OUT_DEPTH_VIS, depth_vis)
        print(f"Saved colorized depth visualization to {OUT_DEPTH_VIS}")

    print("\nDone.")


if __name__ == "__main__":
    main()
