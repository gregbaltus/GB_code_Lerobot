#!/usr/bin/env python3
"""
GB02_see_shape.py

Vision-only test: take a snapshot from the angled camera and detect the
4 coloured 3D-printed pieces, using the same HSV colour segmentation +
shape validation as GB_code/GB05_so101_see_shape_and_move.py.

No robot, no homography, no movement -- just detection + display.

Pieces recognised:
  orange circle  |  green star  |  red triangle  |  yellow hexagon

Usage: type a piece name (partial match accepted, e.g. "star", "orange").
       A fresh snapshot is taken for every query. Type 'q' to quit.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX  = "/dev/video2"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
WARMUP_FRAMES = 5

MIN_AREA = 400  # pixels^2

# ---------------------------------------------------------------------------
# Shape checks (identical to GB05 -- reused as-is, tune here if the angled
# camera's distortion causes misses)
# ---------------------------------------------------------------------------
def _is_circle(cnt):
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    return perim > 0 and (4 * np.pi * area / perim ** 2) > 0.70

def _is_star(cnt):
    area      = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    # 0.90 (not GB05's 0.82): at this angle/distance the star is small enough
    # that morphological closing smooths its points, raising solidity
    return hull_area > 0 and (area / hull_area) < 0.90

def _is_triangle(cnt):
    # larger epsilon than the other shapes: this angled camera's perspective
    # distortion on the triangle needs more aggressive polygon simplification
    # to collapse down to 3 vertices (0.04 leaves a spurious 4th vertex on
    # some instances -- 0.10 was the smallest value that worked for all
    # tested triangle instances, with margin)
    approx = cv2.approxPolyDP(cnt, 0.10 * cv2.arcLength(cnt, True), True)
    return len(approx) == 3

def _is_hexagon(cnt):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    return 5 <= len(approx) <= 8


# ---------------------------------------------------------------------------
# Colour + shape definitions, decoupled -- HSV ranges retuned from GB05's for
# this angled camera's lighting/white balance (sampled directly from a real
# snapshot; see notes):
#   - orange/red hues sit much closer together here than on GB05's camera
#     (orange ~5-13, red right at the 0/179 wrap, ~0-2 and ~178-179), so
#     orange's upper bound was narrowed and red gets two ranges to cover
#     the hue wraparound.
#   - orange's old upper bound (22) overlapped yellow hexagon's hue (~19-22),
#     causing the hexagon to be misdetected as an orange circle.
# COLORS/SHAPES are independent (any colour can be combined with any shape,
# for GB05_see_shape_preprocess_drop_unconform.py's conform/unconform pieces).
# PIECES below is regenerated from them for the 4 "conform" combinations --
# kept as the same list-of-dicts shape so GB02/GB03/GB04 need no changes.
# ---------------------------------------------------------------------------
COLORS = {
    "orange": {"hsv_ranges": [((5, 130, 120), (16, 255, 255))], "draw_color": (0, 140, 255)},
    "green":  {"hsv_ranges": [((72, 120, 60), (97, 255, 190))], "draw_color": (0, 180, 0)},
    "red":    {"hsv_ranges": [((0, 100, 90), (4, 255, 230)), ((174, 100, 90), (179, 255, 230))], "draw_color": (0, 0, 220)},
    "yellow": {"hsv_ranges": [((17, 60, 80), (38, 255, 255))], "draw_color": (0, 210, 230)},
}

SHAPES = {
    "circle":   _is_circle,
    "star":     _is_star,
    "triangle": _is_triangle,
    "hexagon":  _is_hexagon,
}

# shape -> its "conform" colour (the 4 pieces GB01-GB04 were built/trained around)
CONFORM_PAIRS = {
    "circle":   "orange",
    "star":     "green",
    "triangle": "red",
    "hexagon":  "yellow",
}


def make_piece(color_name, shape_name):
    """Build a PIECES-style dict for any colour+shape combination (conform or
    not) -- reuses find_all_instances()/build_mask() unchanged."""
    color = COLORS[color_name]
    return {
        "label":       f"{color_name} {shape_name}",
        "hsv_ranges":  color["hsv_ranges"],
        "draw_color":  color["draw_color"],
        "shape_check": SHAPES[shape_name],
    }


def match_color(name):
    """Return the first COLORS key containing `name` (case-insensitive), or None."""
    name_lower = name.strip().lower()
    for c in COLORS:
        if name_lower in c:
            return c
    return None


def match_shape(name):
    """Return the first SHAPES key containing `name` (case-insensitive), or None."""
    name_lower = name.strip().lower()
    for s in SHAPES:
        if name_lower in s:
            return s
    return None


PIECES = [make_piece(color, shape) for shape, color in CONFORM_PAIRS.items()]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def build_mask(hsv, hsv_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def find_all_instances(hsv, piece):
    """Return a list of all detections for this piece (colour + shape match)."""
    mask = build_mask(hsv, piece["hsv_ranges"])

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # smaller than GB05's 7x7: pieces are smaller in this view
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        if piece["shape_check"] and not piece["shape_check"](cnt):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        results.append({"center": (cx, cy), "bbox": (x, y, w, h)})

    return results


def match_piece(name):
    """Return the first PIECES entry whose label contains `name` (case-insensitive)."""
    name_lower = name.strip().lower()
    for p in PIECES:
        if name_lower in p["label"].lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def capture_snapshot(cap, warmup=WARMUP_FRAMES):
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def show_detections(frame_bgr, piece, instances):
    """Show every detected instance of `piece`, boxed and labelled. Blocks
    until the window is closed."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    color_hex = "#{:02x}{:02x}{:02x}".format(*piece["draw_color"][::-1])  # BGR->RGB
    for inst in instances:
        cx, cy     = inst["center"]
        x, y, w, h = inst["bbox"]
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=2, edgecolor=color_hex, facecolor='none'))
        ax.plot(cx, cy, '+', color=color_hex, markersize=14, markeredgewidth=2)

    n = len(instances)
    ax.set_title(f"Found {n}x {piece['label']}", fontsize=13)
    ax.axis('off')

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GB02 -- See Shape (vision only, angled camera)")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}.")
        return

    known = "  |  ".join(p["label"] for p in PIECES)
    print(f"\nKnown pieces: {known}")
    print("Partial names accepted (e.g. 'star', 'orange'). Type 'q' to quit.\n")

    try:
        while True:
            name = input("Piece to find: ").strip()
            if not name or name.lower() == 'q':
                break

            piece = match_piece(name)
            if piece is None:
                print(f"  Unknown piece '{name}'. Known: {known}\n")
                continue

            print(f"  Looking for '{piece['label']}' …")
            frame_bgr = capture_snapshot(cap)
            hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            instances = find_all_instances(hsv, piece)

            if not instances:
                print(f"  '{piece['label']}' not detected. Try again.\n")
                continue

            print(f"  Found {len(instances)}x '{piece['label']}'")
            for inst in instances:
                print(f"    center={inst['center']}  bbox={inst['bbox']}")

            show_detections(frame_bgr, piece, instances)
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
