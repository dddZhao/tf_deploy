import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from utils.utils import *
from utils.crack import detect_cracks
from arc_matching.transform import transform_image

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def setup_logging(log_level='INFO'):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def process_image(img_path, tile_size=(512, 512), full_pipeline=True):
    img_path = str(img_path)
    if not Path(img_path).exists():
        raise FileNotFoundError(f"输入文件不存在: {img_path}")

    start_time = time.time()
    logging.info(f"开始处理图像: {img_path}")

    output_json_path = tunnelface_segmentation(img_path)
    output_image_path = str(get_output_path(img_path, suffix="_transformed"))

    transformed_img_path = transform_image(
        input_image_path=img_path,
        input_json_path=output_json_path,
        output_image_path=output_image_path
    )

    show_seg(img_path)

    split_images(transformed_img_path, tile_size)
    selected_count = select_images(transformed_img_path, tile_size)

    result = {
        "image": img_path,
        "output_dir": str(get_image_output_root(img_path)),
        "transformed_image": transformed_img_path,
        "selected_tiles": selected_count,
        "crack_detection": None,
    }

    if full_pipeline:
        crack_result = detect_cracks(transformed_img_path, tile_size=tile_size)

        classified_count = classify_posui(transformed_img_path, tile_size)
        show_result_posui(transformed_img_path)
        show_result_water(transformed_img_path)

        YTLX = classify_YTLX(transformed_img_path)
        FHCD = classify_FHCD(transformed_img_path)

        YTLX_MAPPING = {0: "岩浆岩", 1: "沉积岩", 2: "变质岩"}
        FHCD_MAPPING = {0: "未风化", 1: "弱风化", 2: "强风化"}

        result_msg = (f"岩体类型: {YTLX_MAPPING.get(YTLX, '未知')}\n"
                      f"风化程度: {FHCD_MAPPING.get(FHCD, '未知')}")
        logging.info(result_msg)
        result.update({
            "YTLX": int(YTLX),
            "FHCD": int(FHCD),
            "YTLX_name": YTLX_MAPPING.get(YTLX, "未知"),
            "FHCD_name": FHCD_MAPPING.get(FHCD, "未知"),
            "classified_tiles": classified_count,
            "crack_detection": crack_result,
        })

    elapsed = time.time() - start_time
    logging.info(f"处理完成! 耗时: {elapsed:.2f}秒")
    return result

def iter_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path
        return

    seen = set()
    for ext in IMAGE_EXTENSIONS:
        for candidate in sorted(input_path.glob(f"*{ext}")) + sorted(input_path.glob(f"*{ext.upper()}")):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

def process_images(input_path, tile_size=(512, 512), full_pipeline=True):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if input_path.is_dir():
        logging.info(f"开始批量处理文件夹: {input_path}")

    image_files = list(iter_images(input_path))
    if not image_files:
        logging.warning(f"未找到可处理图片: {input_path}")
        return 0, 0

    success = 0
    failed = 0
    for img_path in image_files:
        try:
            process_image(img_path, tile_size=tile_size, full_pipeline=full_pipeline)
            success += 1
        except Exception as e:
            failed += 1
            logging.error(f"处理失败 {img_path}: {str(e)}", exc_info=True)
            if input_path.is_file():
                raise

    if input_path.is_dir():
        logging.info(f"批量处理完成: 成功 {success}, 失败 {failed}")
    return success, failed

def main(input_path, tile_size=(512, 512), log_level='INFO', full_pipeline=True):
    setup_logging(log_level)
    try:
        process_images(input_path, tile_size=tile_size, full_pipeline=full_pipeline)
    except Exception as e:
        logging.error(f"处理失败: {str(e)}", exc_info=True)
        sys.exit(1)

# 设置命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隧道面图像处理流程")
    parser.add_argument('input_path', help="输入图像或图片文件夹路径")

    parser.add_argument('--tile_size', type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=[512, 512],
                        help="切片尺寸 (默认: %(default)s)")

    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="日志级别")
    parser.add_argument('--screen_only', action='store_true',
                        help="仅执行分割、变换、切片和筛选")

    args = parser.parse_args()

    # 转换tile_size为元组
    tile_size = tuple(args.tile_size)

    main(
        args.input_path,
        tile_size=tile_size,
        log_level=args.log_level,
        full_pipeline=not args.screen_only
    )
