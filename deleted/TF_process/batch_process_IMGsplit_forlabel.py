import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
from utils.utils import *
from arc_matching.transform import transform_image
from utils.split_forlabel import *

# 配置项目根路径
PROJECT_ROOT = Path(r"E:\crack_dataset")

#PROJECT_ROOT = Path(__file__).parent.parent
#sys.path.append(str(PROJECT_ROOT))

def process_image(img_path: str, tile_size: Tuple[int, int] = (512, 512)) -> None:
    """
    根据图像文件大小决定处理流程，严格保持宽高比

    参数:
        img_path: 输入图像路径
        tile_size: 切分瓦片尺寸 (宽度, 高度)
    """
    output_json_path = tunnelface_segmentation(img_path)

    # 获取图像文件大小（单位：KB）
    file_size_kb = os.path.getsize(img_path) / 1024

    # 定义目标尺寸
    TARGET_WIDTH = 2400
    TARGET_HEIGHT = 2000
    SIZE_THRESHOLD = 1500  # KB

    if file_size_kb < SIZE_THRESHOLD:
        masked_path = mask_seg(img_path)


        # 打开图像并获取原始尺寸
        with Image.open(masked_path) as img:

            original_width, original_height = img.size
            aspect_ratio = original_width / original_height

            # 计算保持宽高比的目标尺寸
            if aspect_ratio > TARGET_WIDTH / TARGET_HEIGHT:
                # 宽图：以宽度为基准
                new_width = TARGET_WIDTH
                new_height = int(TARGET_WIDTH / aspect_ratio)
            else:
                # 高图：以高度为基准
                new_height = TARGET_HEIGHT
                new_width = int(TARGET_HEIGHT * aspect_ratio)

            # 调整图像大小（使用高质量Lanczos算法）
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            # 保存调整后的图像
            resized_img.save(masked_path, quality=95)

            # 使用调整后的图像进行切分
            output_dir = split_images_forlabel(str(masked_path), tile_size)
            clean_tiles(output_dir, 0.15)
    else:
        # 执行核心处理流程
        output_image_path = get_output_path(img_path, suffix="_transformed")

        transformed_img_path = transform_image(
            input_image_path=img_path,
            input_json_path=output_json_path,
            output_image_path=str(output_image_path)
        )
        output_dir = split_images_forlabel(transformed_img_path, tile_size)
        clean_tiles(output_dir, 0.15)

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
            process_image(str(img_file), tile_size)
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