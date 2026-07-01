import os
import re
import argparse
from glob import glob

import cv2
import numpy as np


RE_MASK = re.compile(r"_tile_(\d+)_(\d+)__mask\.png$")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stitch *_mask.png tiles into one transparent red mask PNG."
    )
    ap.add_argument(
        "--tile-dir",
        required=True,
        help="folder containing *_mask.png tiles"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output stitched PNG path (should end with .png)"
    )
    ap.add_argument(
        "--tile-h",
        type=int,
        default=None,
        help="tile height; default: infer from first mask"
    )
    ap.add_argument(
        "--tile-w",
        type=int,
        default=None,
        help="tile width; default: infer from first mask"
    )
    ap.add_argument(
        "--full-h",
        type=int,
        default=None,
        help="full stitched image height; default: infer from max row"
    )
    ap.add_argument(
        "--full-w",
        type=int,
        default=None,
        help="full stitched image width; default: infer from max col"
    )
    ap.add_argument(
        "--color",
        default="255,0,0",
        help="output color in RGB, default red: '255,0,0'"
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="binarization threshold for white mask area"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    mask_paths = sorted(glob(os.path.join(args.tile_dir, "*_mask.png")))
    if not mask_paths:
        raise RuntimeError(f"No *_mask.png files found in: {args.tile_dir}")

    print(f"[INFO] Found {len(mask_paths)} mask tiles")

    parsed_tiles = []
    max_row = -1
    max_col = -1

    for mp in mask_paths:
        name = os.path.basename(mp)
        m = RE_MASK.search(name)
        if m is None:
            print(f"[WARN] Skip unmatched filename: {name}")
            continue

        row, col = int(m.group(1)), int(m.group(2))
        parsed_tiles.append((mp, row, col))
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    if not parsed_tiles:
        raise RuntimeError("No valid mask tile filenames matched the pattern *_tile_row_col__mask.png")

    # infer tile size
    if args.tile_h is None or args.tile_w is None:
        sample = cv2.imread(parsed_tiles[0][0], cv2.IMREAD_GRAYSCALE)
        if sample is None:
            raise RuntimeError(f"Cannot read sample tile: {parsed_tiles[0][0]}")
        th, tw = sample.shape[:2]
    else:
        th, tw = args.tile_h, args.tile_w

    print(f"[INFO] Tile size: {tw} x {th}")

    # infer final canvas size
    if args.full_h is None:
        H = (max_row + 1) * th
    else:
        H = args.full_h

    if args.full_w is None:
        W = (max_col + 1) * tw
    else:
        W = args.full_w

    print(f"[INFO] Output canvas size: {W} x {H}")

    # RGBA canvas: default fully transparent
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    rgb = [int(x) for x in args.color.split(",")]
    if len(rgb) != 3:
        raise ValueError("--color must be in RGB format like 255,0,0")
    r, g, b = rgb

    for mp, row, col in parsed_tiles:
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Cannot read: {mp}")
            continue

        y0 = row * th
        x0 = col * tw
        y1 = min(y0 + mask.shape[0], H)
        x1 = min(x0 + mask.shape[1], W)

        valid_h = y1 - y0
        valid_w = x1 - x0
        if valid_h <= 0 or valid_w <= 0:
            print(f"[WARN] Tile out of range, skip: {mp}")
            continue

        m_tile = mask[:valid_h, :valid_w]
        white_region = m_tile > args.threshold

        if not np.any(white_region):
            continue

        roi = canvas[y0:y1, x0:x1]

        # 白色区域 -> 红色 + 不透明
        roi[white_region, 0] = r
        roi[white_region, 1] = g
        roi[white_region, 2] = b
        roi[white_region, 3] = 255

        canvas[y0:y1, x0:x1] = roi

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(args.out, canvas)
    if not success:
        raise RuntimeError(f"Failed to save output: {args.out}")

    print(f"[DONE] Saved stitched transparent mask to: {args.out}")


if __name__ == "__main__":
    main()