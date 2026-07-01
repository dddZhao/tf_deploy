import logging
import os
import re
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .cn_cv import cv_imread_cn, cv_imwrite_cn
from .paths import get_image_output_root, get_output_dir, get_project_root
from .runtime import infer_image, load_engine, preprocess


CRACK_RE_MASK = re.compile(r"_tile_(\d+)_(\d+)__mask\.png$")
CRACK_RE_OVERLAY = re.compile(r"_tile_(\d+)_(\d+)__overlay\.png$")
CRACK_MODEL_PATH = os.path.join(
    get_project_root(), "model", "crackseg_segformer.engine"
)


def _clear_transformed_path(transformed_img_path: str) -> Path:
    path = Path(transformed_img_path)
    name = path.name
    if name.endswith("_black_transformed.png"):
        return path.with_name(name.replace("_black_transformed.png", "_clear_transformed.png"))
    return path


def _parse_color_bgr(color_bgr: Tuple[int, int, int]) -> np.ndarray:
    if len(color_bgr) != 3:
        raise ValueError("color_bgr must contain exactly three values")
    return np.array([int(x) for x in color_bgr], dtype=np.float32)


def _crack_mask_from_output(output: np.ndarray) -> np.ndarray:
    pred = np.asarray(output)
    pred = np.squeeze(pred)

    if pred.ndim == 3:
        if pred.shape[0] <= 8:
            pred = np.argmax(pred, axis=0)
        elif pred.shape[-1] <= 8:
            pred = np.argmax(pred, axis=-1)
        else:
            pred = np.squeeze(pred)

    if pred.dtype.kind == "f":
        crack = pred > 0.5
    else:
        crack = pred == 1
    return crack.astype(np.uint8) * 255


def overlay_mask_on_image(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.9,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    overlay = img_bgr.copy()
    crack = mask > 127
    if crack.any():
        color = _parse_color_bgr(color_bgr)
        overlay[crack] = (
            overlay[crack].astype(np.float32) * (1 - alpha)
            + color * alpha
        ).astype(np.uint8)
    return overlay


def predict_crack_tiles(
    transformed_img_path: str,
    tile_size: Tuple[int, int] = (512, 512),
    alpha: float = 0.9,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
) -> tuple[Path, int]:
    selected_dir = Path(f"{get_output_dir(transformed_img_path)}_select")
    output_dir = (
        get_image_output_root(transformed_img_path)
        / f"{Path(transformed_img_path).stem}_crack_tiles"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("*__mask.png"):
        old_file.unlink()
    for old_file in output_dir.glob("*__overlay.png"):
        old_file.unlink()

    selected_tiles = sorted(selected_dir.glob("*.jpg"))
    if not selected_tiles:
        logging.warning(f"没有可进行裂隙识别的筛选切片: {selected_dir}")
        return output_dir, 0

    engine = load_engine(CRACK_MODEL_PATH)
    processed = 0

    for tile_path in selected_tiles:
        output = infer_image(
            engine,
            str(tile_path),
            tile_size,
            binding_name="input",
            input_shape_format="CHW",
            preprocess_func=preprocess,
        )
        mask = _crack_mask_from_output(output)

        try:
            tile_img = cv_imread_cn(str(tile_path))
        except IOError:
            logging.warning(f"无法读取裂隙切片: {tile_path}")
            continue
        if mask.shape[:2] != tile_img.shape[:2]:
            mask = cv2.resize(mask, (tile_img.shape[1], tile_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = overlay_mask_on_image(tile_img, mask, alpha=alpha, color_bgr=color_bgr)
        cv_imwrite_cn(str(output_dir / f"{tile_path.stem}__mask.png"), mask)
        cv_imwrite_cn(str(output_dir / f"{tile_path.stem}__overlay.png"), overlay)
        processed += 1

    logging.info(f"裂隙识别完成: {processed}/{len(selected_tiles)} 个筛选切片")
    return output_dir, processed


def stitch_crack_overlay(
    transformed_img_path: str,
    crack_tile_dir: Path,
    tile_size: Tuple[int, int] = (512, 512),
    alpha: float = 0.9,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
) -> Path | None:
    orig_path = _clear_transformed_path(transformed_img_path)
    orig = cv2.imread(str(orig_path), cv2.IMREAD_UNCHANGED)
    if orig is None:
        raise FileNotFoundError(f"无法读取裂隙叠加底图: {orig_path}")

    mask_paths = sorted(Path(crack_tile_dir).glob("*__mask.png"))
    if not mask_paths:
        logging.warning(f"没有可拼接的裂隙 mask: {crack_tile_dir}")
        return None

    height, width = orig.shape[:2]
    tile_width, tile_height = tile_size
    color = _parse_color_bgr(color_bgr)
    has_alpha = orig.ndim == 3 and orig.shape[2] == 4
    canvas = orig.copy()
    stitched_tiles = 0

    for mask_path in mask_paths:
        match = CRACK_RE_MASK.search(mask_path.name)
        if match is None:
            logging.warning(f"跳过无法解析行列号的裂隙 mask: {mask_path.name}")
            continue

        row, col = int(match.group(1)), int(match.group(2))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"无法读取裂隙 mask: {mask_path}")
            continue

        y0 = row * tile_height
        x0 = col * tile_width
        y1 = min(y0 + mask.shape[0], height)
        x1 = min(x0 + mask.shape[1], width)
        if y0 >= height or x0 >= width or y1 <= y0 or x1 <= x0:
            continue

        mask_tile = mask[:(y1 - y0), :(x1 - x0)]
        crack = mask_tile > 127
        if not crack.any():
            continue

        roi = canvas[y0:y1, x0:x1]
        if has_alpha:
            roi_bgr = roi[:, :, :3]
            roi_alpha = roi[:, :, 3]
            roi_bgr[crack] = (
                roi_bgr[crack].astype(np.float32) * (1 - alpha)
                + color * alpha
            ).astype(np.uint8)
            roi_alpha[crack] = 255
            roi[:, :, :3] = roi_bgr
            roi[:, :, 3] = roi_alpha
        else:
            roi[crack] = (
                roi[crack].astype(np.float32) * (1 - alpha)
                + color * alpha
            ).astype(np.uint8)

        canvas[y0:y1, x0:x1] = roi
        stitched_tiles += 1

    output_path = (
        get_image_output_root(transformed_img_path)
        / f"{Path(orig_path).stem}_crack_overlay.png"
    )
    cv_imwrite_cn(str(output_path), canvas)
    logging.info(f"裂隙叠加图已保存: {output_path}，有效裂隙切片 {stitched_tiles}")
    return output_path


def detect_cracks(
    transformed_img_path: str,
    tile_size: Tuple[int, int] = (512, 512),
    alpha: float = 0.9,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
) -> dict:
    crack_tile_dir, tile_count = predict_crack_tiles(
        transformed_img_path,
        tile_size=tile_size,
        alpha=alpha,
        color_bgr=color_bgr,
    )
    overlay_path = None
    if tile_count > 0:
        overlay_path = stitch_crack_overlay(
            transformed_img_path,
            crack_tile_dir,
            tile_size=tile_size,
            alpha=alpha,
            color_bgr=color_bgr,
        )
    return {
        "tile_dir": str(crack_tile_dir),
        "tile_count": tile_count,
        "overlay_path": str(overlay_path) if overlay_path else None,
    }
