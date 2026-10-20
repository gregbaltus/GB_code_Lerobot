#!/usr/bin/env python3
"""
displaySeeForm.py

Computer-vision-only detection of the four coloured 3D-printed pieces:
  - orange circle
  - green star
  - red triangle
  - yellow hexagon

Strategy: HSV color segmentation → contour → shape validation.
No robot, no homography — just camera and OpenCV.

Run this to tune the detection before integrating into GB05.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Piece definitions
# ---------------------------------------------------------------------------
# Each entry:
#   label       : display name
#   hsv_ranges  : list of (lower, upper) HSV tuples — red needs two ranges
#   draw_color  : BGR color used to draw annotations on the frame
#   shape_check : callable(contour) -> bool  (optional shape validation)

def _is_circle(cnt):
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return False
    return (4 * np.pi * area / perim ** 2) > 0.70

def _is_star(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return False
    return (area / hull_area) < 0.82   # convex shapes sit near 1.0; chubby stars ~0.65–0.80

def _is_triangle(cnt):
    peri  = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    return len(approx) == 3

def _is_hexagon(cnt):
    peri  = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    return 5 <= len(approx) <= 8


PIECES = [
    {
        "label":       "orange circle",
        # Measured: H≈10, S≈217, V≈206  (HSV 20°, 85%, 81%)
        "hsv_ranges":  [((3, 130, 130), (22, 255, 255))],
        "draw_color":  (0, 140, 255),    # BGR orange
        "shape_check": _is_circle,
    },
    {
        "label":       "green star",
        # Measured: H≈83, S≈255, V≈112  (HSV 166°, 100%, 44%)
        "hsv_ranges":  [((72, 120, 60), (97, 255, 190))],
        "draw_color":  (0, 180, 0),      # BGR green
        #"hsv_ranges":  [((158, 120, 80), (179, 255, 230))], # Test with red star
        #"draw_color":  (0, 0, 220),      # BGR red # Test with red star
        "shape_check": _is_star,
    },
    {
        "label":       "red triangle",
        # Measured: H≈171, S≈230, V≈171  (HSV 343°, 90%, 67%) — raspberry red
        # Lives near the 180° wrap, NOT near 0°
        "hsv_ranges":  [((158, 120, 80), (179, 255, 230))],
        "draw_color":  (0, 0, 220),      # BGR red
        "shape_check": _is_triangle,
    },
    {
        "label":       "yellow hexagon",
        "hsv_ranges":  [((15, 60, 80), (38, 255, 255))],
        "draw_color":  (0, 210, 230),    # BGR yellow-ish
        "shape_check": _is_hexagon,
    },
]

MIN_AREA = 400   # pixels² — ignore tiny noise blobs


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def build_mask(hsv, hsv_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def find_piece(hsv, piece, min_area=MIN_AREA):
    """
    Returns {"center": (cx, cy), "bbox": (x,y,w,h), "contour": cnt}
    or None if the piece is not found / shape validation fails.
    """
    mask = build_mask(hsv, piece["hsv_ranges"])

    # Morphological cleanup: close small holes, remove tiny blobs
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None

    # Optional shape validation
    if piece["shape_check"] and not piece["shape_check"](cnt):
        return None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(cnt)
    return {"center": (cx, cy), "bbox": (x, y, w, h), "contour": cnt}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def capture_snapshot(cap, warmup=5):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame  # BGR


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def show_detections(frame_bgr, detections):
    """
    Draw bounding boxes and labels on a copy of the frame,
    display with matplotlib (blocks until window is closed).
    """
    vis = frame_bgr.copy()

    for piece, result in detections:
        color = piece["draw_color"]
        label = piece["label"]

        if result is None:
            continue

        cx, cy = result["center"]
        x, y, w, h = result["bbox"]

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(vis, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Count found / not found
    found     = [p["label"] for p, r in detections if r is not None]
    not_found = [p["label"] for p, r in detections if r is None]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    title = f"Found: {', '.join(found) if found else 'none'}"
    if not_found:
        title += f"   |   Not found: {', '.join(not_found)}"
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.show(block=True)
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("displaySeeForm — colour+shape detection for 3D-printed pieces")
    print("=" * 60)

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

    print("\nClose the detection window to continue. Type 'n' to stop.\n")

    while True:
        frame_bgr = capture_snapshot(cap)
        hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        detections = []
        for piece in PIECES:
            result = find_piece(hsv, piece)
            detections.append((piece, result))
            if result:
                cx, cy = result["center"]
                print(f"  {piece['label']:20s}  center=({cx:4d}, {cy:4d})")
            else:
                print(f"  {piece['label']:20s}  NOT FOUND")
        print()

        show_detections(frame_bgr, detections)

        again = input("Try again? (y/n): ").strip().lower()
        if again != 'y':
            break

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
