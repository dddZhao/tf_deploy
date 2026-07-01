import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from utils.utils import *
from arc_matching.transform import transform_image
from utils.split_forlabel import clean_tiles

# 配置项目根路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def process_image(img_path, tile_size, log_level='INFO'):
    """处理单个图像直到select_images步骤"""
    logging.info(f"开始处理图像: {img_path}")
    start_time = time.time()

    # 执行核心处理流程
    output_json_path = tunnelface_segmentation(img_path)
    stem = Path(img_path).stem
    output_image_path = str(PROJECT_ROOT / "data" / "output" / f"{stem}_transformed.png")

    transformed_img_path = transform_image(
        input_image_path=img_path,
        input_json_path=output_json_path,
        output_image_path=output_image_path
    )

    output_dir = split_images(transformed_img_path, tile_size)
    #select_images(transformed_img_path, tile_size)
    clean_tiles(output_dir, 0.15)

    # 记录处理耗时
    elapsed = time.time() - start_time
    logging.info(f"图像处理完成! 耗时: {elapsed:.2f}秒")
    return img_path


def main(folder_path, tile_size=(512, 512), log_level='INFO'):
    """批量处理文件夹中的所有图像"""
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # 验证输入路径
    if not Path(folder_path).is_dir():
        logging.error(f"输入路径不是文件夹: {folder_path}")
        return

    # 获取所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in Path(folder_path).iterdir()
                   if f.suffix.lower() in img_extensions]

    if not image_files:
        logging.warning(f"文件夹中未找到图像文件: {folder_path}")
        return

    # 批量处理图像
    for img_file in image_files:
        try:
            process_image(str(img_file), tile_size, log_level)
        except Exception as e:
            logging.error(f"处理失败 {img_file.name}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隧道面图像批量处理")
    parser.add_argument('folder_path', help="输入文件夹路径")
    parser.add_argument('--tile_size', type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=[512, 512],
                        help="切片尺寸 (默认: %(default)s)")
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="日志级别")

    args = parser.parse_args()
    main(
        folder_path=args.folder_path,
        tile_size=tuple(args.tile_size),
        log_level=args.log_level
    )