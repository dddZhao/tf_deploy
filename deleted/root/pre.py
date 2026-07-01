import os
import re
import glob
import argparse
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmseg.apis import init_model, inference_model
from mmengine.utils import mkdir_or_exist


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_best_miou_ckpt(work_dir: str) -> str:
    cand = sorted(glob.glob(os.path.join(work_dir, "best_mIoU_iter_*.pth")))
    if not cand:
        raise FileNotFoundError(f"No best_mIoU_iter_*.pth found in: {work_dir}")

    def iter_from_name(p):
        m = re.search(r"best_mIoU_iter_(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    cand.sort(key=iter_from_name)
    return cand[-1]


def list_images(img_dir: str):
    img_dir = Path(img_dir)
    files = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))
    files.sort()
    return files


def overlay_mask_on_image(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                          color_bgr=(0, 255, 255)):
    overlay = img_bgr.copy()
    crack = (mask == 1)
    if crack.any():
        color = np.array(color_bgr, dtype=np.uint8)
        overlay[crack] = (overlay[crack].astype(np.float32) * (1 - alpha) +
                          color.astype(np.float32) * alpha).astype(np.uint8)
    return overlay


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="mmseg config path")
    ap.add_argument("--work-dir", required=True, help="work_dir containing checkpoints")
    ap.add_argument("--img-dir", required=True, help="folder containing images to infer")
    ap.add_argument("--out-dir", default=None, help="output folder (default: <work_dir>/infer_results)")
    ap.add_argument("--ckpt", default=None, help="checkpoint path; if not set, auto-pick best_mIoU")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu. default: auto")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay alpha")
    ap.add_argument("--color", default="0,255,255", help="BGR color like '0,255,255'")
    ap.add_argument("--save-mask", action="store_true", help="also save __mask.png (0/255)")
    ap.add_argument("--interval", type=int, default=50, help="print progress every N images")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---- device auto ----
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ---- output dir ----
    out_dir = args.out_dir or os.path.join(args.work_dir, "infer_results")
    mkdir_or_exist(out_dir)

    # ---- ckpt pick ----
    ckpt_path = args.ckpt or find_best_miou_ckpt(args.work_dir)
    print(f"[INFO] Using ckpt: {ckpt_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output dir: {out_dir}")

    # ---- parse color ----
    color_bgr = tuple(int(x) for x in args.color.split(","))

    model = init_model(args.config, ckpt_path, device=device)
    model.cfg.visualizer = None

    imgs = list_images(args.img_dir)
    print(f"[INFO] Found {len(imgs)} images under: {args.img_dir}")
    if not imgs:
        return

    for i, img_path in enumerate(imgs, 1):
        result = inference_model(model, img_path)

        pred = result.pred_sem_seg.data
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        pred = pred.cpu().numpy().astype(np.int32)

        img = mmcv.imread(img_path, channel_order="bgr")
        vis = overlay_mask_on_image(img, pred, alpha=args.alpha, color_bgr=color_bgr)

        rel = os.path.relpath(img_path, args.img_dir)
        rel_stem = os.path.splitext(rel)[0].replace(os.sep, "__")

        out_vis = os.path.join(out_dir, f"{rel_stem}__overlay.png")
        mmcv.imwrite(vis, out_vis)

        if args.save_mask:
            out_mask = os.path.join(out_dir, f"{rel_stem}__mask.png")
            mask_u8 = (pred == 1).astype(np.uint8) * 255
            mmcv.imwrite(mask_u8, out_mask)

        if i % args.interval == 0 or i == 1:
            print(f"[{i:>5}/{len(imgs)}] saved -> {out_vis}")

    print(f"[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
