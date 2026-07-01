import os
import re
import argparse
from glob import glob

import cv2
import numpy as np


RE_OVERLAY = re.compile(r"_tile_(\d+)_(\d+)__overlay\.png$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-dir", required=True, help="folder containing tile __overlay.png and __mask.png")
    ap.add_argument("--orig", required=True, help="original full image path")
    ap.add_argument("--out", required=True, help="output stitched overlay image path")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    ap.add_argument("--color", default="0,255,255", help="BGR color like '0,255,255'")
    ap.add_argument("--tile-h", type=int, default=None, help="tile height; default: read from first overlay")
    ap.add_argument("--tile-w", type=int, default=None, help="tile width; default: read from first overlay")
    return ap.parse_args()


def main():
    args = parse_args()

    # =========================
    # Read original image with alpha channel preserved
    # =========================
    orig = cv2.imread(args.orig, cv2.IMREAD_UNCHANGED)

    if orig is None:
        raise FileNotFoundError(f"Cannot read original image: {args.orig}")

    H, W = orig.shape[:2]

    has_alpha = (orig.ndim == 3 and orig.shape[2] == 4)

    print(f"[INFO] Original image size: {W} x {H}")
    print(f"[INFO] Original image shape: {orig.shape}")

    if has_alpha:
        print("[INFO] Alpha channel detected. Transparent background will be preserved.")
    else:
        print("[INFO] No alpha channel detected.")

    overlay_paths = sorted(glob(os.path.join(args.tile_dir, "*__overlay.png")))

    if not overlay_paths:
        raise RuntimeError(f"No overlay tiles found in {args.tile_dir}")

    # =========================
    # Infer tile size
    # =========================
    if args.tile_h is None or args.tile_w is None:
        sample = cv2.imread(overlay_paths[0], cv2.IMREAD_UNCHANGED)

        if sample is None:
            raise RuntimeError(f"Cannot read sample tile: {overlay_paths[0]}")

        th, tw = sample.shape[:2]
    else:
        th, tw = args.tile_h, args.tile_w

    print(f"[INFO] Tile size: {tw} x {th}")

    canvas = orig.copy()

    alpha = args.alpha

    color_bgr = np.array(
        [int(x) for x in args.color.split(",")],
        dtype=np.float32
    )

    # =========================
    # Stitch masks back to original image
    # =========================
    for op in overlay_paths:
        name = os.path.basename(op)
        m = RE_OVERLAY.search(name)

        if m is None:
            continue

        row, col = int(m.group(1)), int(m.group(2))

        mask_path = op.replace("__overlay.png", "__mask.png")

        if not os.path.exists(mask_path):
            print(f"[WARN] missing mask: {mask_path}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[WARN] cannot read mask: {mask_path}")
            continue

        y0 = row * th
        x0 = col * tw
        y1 = min(y0 + th, H)
        x1 = min(x0 + tw, W)

        if y0 >= H or x0 >= W:
            continue

        m_tile = mask[:(y1 - y0), :(x1 - x0)]
        crack = (m_tile > 127)

        if not crack.any():
            continue

        roi = canvas[y0:y1, x0:x1]

        if has_alpha:
            # BGRA image:
            # only blend BGR channels, keep alpha channel
            roi_bgr = roi[:, :, :3]
            roi_alpha = roi[:, :, 3]

            roi_crack = roi_bgr[crack].astype(np.float32)

            blended = (
                roi_crack * (1 - alpha)
                + color_bgr * alpha
            )

            roi_bgr[crack] = blended.astype(np.uint8)

            # Ensure crack overlay is visible even if the original pixel was transparent
            roi_alpha[crack] = 255

            roi[:, :, :3] = roi_bgr
            roi[:, :, 3] = roi_alpha

        else:
            # Normal BGR image
            roi_crack = roi[crack].astype(np.float32)

            blended = (
                roi_crack * (1 - alpha)
                + color_bgr * alpha
            )

            roi[crack] = blended.astype(np.uint8)

        canvas[y0:y1, x0:x1] = roi

    # =========================
    # Save result
    # =========================
    out_dir = os.path.dirname(args.out)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Important:
    # save as PNG if you want to preserve transparency
    cv2.imwrite(args.out, canvas)

    print(f"[DONE] Saved: {args.out}")


if __name__ == "__main__":
    main()