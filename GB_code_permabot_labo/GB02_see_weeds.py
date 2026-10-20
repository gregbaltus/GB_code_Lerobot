#!/usr/bin/env python3
"""
GB02_see_weeds.py

Vision-only test: take a snapshot of the soil bed and classify every green
plant blob as either a "weed" (small, to be removed) or "cultivable" (big,
leave alone), using HSV hue segmentation + connected-component area --
analogous to GB_code_camera/GB02_see_shape.py's role in that pipeline, but
classifying by blob AREA instead of by shape (circle/star/triangle/hexagon).

No robot, no homography, no movement -- just detection + display.

Background: a real pixel analysis of an actual lab screenshot (soil bed with
4 big grayish-green cultivable plants + small weed sprigs of the same
material) found HUE is the reliable separator from soil (plants ~H65-67,
soil ~H15-17 -- a huge gap), NOT brightness (soil's own texture/crumbs vary
enough in brightness to cause false positives on a pure V threshold). Once
masked by hue, blob AREA cleanly separated the two big cultivable plants
(tens of thousands of px) from every weed candidate (under ~1700px) -- a
large safety margin. A reddish-brown plant variety was also present in that
lab setup but deliberately NOT targeted here: its hue is too close to the
soil's own hue to separate reliably, and per the lab setup all weed props
are confirmed to be the grayish-green material only, so a green-hue-only
mask safely never flags the reddish plant as anything (not weed, not
cultivable) -- it's simply invisible to this detector, which is fine since
we only need to find weeds.

The HSV/area constants below are a starting point from that one screenshot,
at a different resolution/crop than your live camera -- expect to retune
them (same iterative process used for GB_code_camera/GB02_see_shape.py's
colour ranges) once you run this against the real feed.

Usage: press Enter to take a snapshot and detect, 'q' to quit.
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
CAMERA_INDEX  = "/dev/video2"   # left camera -- confirmed as the marking/detection camera
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
WARMUP_FRAMES = 5

# ---------------------------------------------------------------------------
# Region of interest -- the soil bed is a rectangle, but rotated relative to
# the camera, so a simple axis-aligned (x, y, w, h) box can't tightly match
# it: a box big enough to contain the whole bed still leaks some background
# in at the corners. Instead this is a 4-corner polygon (clockwise from
# top-left), masked with cv2.fillPoly -- everything outside it is excluded
# from detection, following the bed's true tilted shape.
#
# These corners are a PROPORTIONAL ESTIMATE eyeballed from the screenshot you
# shared, scaled to CAMERA_WIDTH x CAMERA_HEIGHT -- the screenshot was a
# different resolution/crop than your live feed, so treat these as a rough
# starting point, not a calibrated value. Tune live: run the script, look at
# the yellow polygon outline show_detections() draws, and nudge the 4 points
# until it tightly traces the soil bed's true edges.
# ---------------------------------------------------------------------------
ROI_POLYGON = [
    (54, 14),    # top-left corner of the soil bed
    (623, 3),    # top-right corner
    (640, 345),  # bottom-right corner
    (52, 269),   # bottom-left corner
]


def build_roi_mask(frame_shape, polygon):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

# ---------------------------------------------------------------------------
# Plant colour -- green hue only (confirmed: all weed props are the
# grayish-green material, not the reddish-brown variety). Loose S/V bounds
# since leaf brightness/saturation varied a lot across the test image
# (shadowed vs. lit leaves); hue is what does the real separating work.
# ---------------------------------------------------------------------------
PLANT_HSV_RANGES = [((40, 20, 30), (100, 255, 255))]

MIN_AREA      = 50     # pixels^2 -- floor to exclude single stray fallen leaves/noise; tune live
WEED_MAX_AREA = 3000    # pixels^2 -- below this = weed candidate; found a huge gap (~1700 vs ~40000+) in testing


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def build_mask(hsv, hsv_ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def find_plant_blobs(hsv, roi_mask):
    """Return every green plant blob inside roi_mask, classified by area.
    Operates on the full frame (masked, not cropped), so centers/bboxes come
    out already in full-frame coordinates -- no offset bookkeeping needed."""
    mask = build_mask(hsv, PLANT_HSV_RANGES)
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        classification = "weed" if area < WEED_MAX_AREA else "cultivable"
        results.append({
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
            "classification": classification,
        })

    return results


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
def show_detections(frame_bgr, blobs):
    """Weeds boxed in red, cultivable plants boxed in blue, each labelled
    with its area (for threshold tuning). Blocks until the window is closed."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(10, 7))
    ax.imshow(frame_rgb)

    ax.add_patch(patches.Polygon(ROI_POLYGON, closed=True,
                 linewidth=1, edgecolor='yellow', facecolor='none', linestyle=':'))

    for blob in blobs:
        x, y, w, h = blob["bbox"]
        cx, cy     = blob["center"]
        is_weed    = blob["classification"] == "weed"
        color      = "red" if is_weed else "deepskyblue"
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     linewidth=2, edgecolor=color, facecolor='none'))
        if is_weed:
            ax.plot(cx, cy, '+', color=color, markersize=14, markeredgewidth=2)
        ax.text(x, y - 6, f"{blob['classification']} ({blob['area']:.0f}px)",
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))

    n_weeds = sum(1 for b in blobs if b["classification"] == "weed")
    n_cult  = len(blobs) - n_weeds
    ax.set_title(f"Found {n_weeds} weed(s) [red], {n_cult} cultivable [blue]", fontsize=13)
    ax.axis('off')

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GB02 -- See Weeds (vision only)")
    print("=" * 50)

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
            hsv       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            roi_mask  = build_roi_mask(frame_bgr.shape, ROI_POLYGON)
            blobs     = find_plant_blobs(hsv, roi_mask)

            if not blobs:
                print("  Nothing detected. Try again.\n")
                continue

            weeds = [b for b in blobs if b["classification"] == "weed"]
            cult  = [b for b in blobs if b["classification"] == "cultivable"]
            print(f"  Found {len(weeds)} weed(s), {len(cult)} cultivable plant(s)")
            for b in sorted(blobs, key=lambda b: b["area"]):
                print(f"    {b['classification']:<11} center={b['center']} area={b['area']:.0f}")

            show_detections(frame_bgr, blobs)
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        plt.close('all')
        print("Done.")


if __name__ == "__main__":
    main()
