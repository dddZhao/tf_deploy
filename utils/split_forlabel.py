import math
from PIL import Image
from pathlib import Path
from typing import Tuple
from .paths import get_output_dir
import os
import numpy as np

def split_images_forlabel(
        img_path: str,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: float = 0.2
) -> None:
    """
    将图像分割为带重叠区域的瓦片

    参数:
        img_path: 输入图像路径
        tile_size: 瓦片尺寸 (宽度, 高度)
        overlap: 瓦片之间的重叠比例 (0.0-1.0)
    """
    image = Image.open(img_path)
    original_width, original_height = image.size
    tile_width, tile_height = tile_size

    # 计算步长（考虑重叠）
    step_x = int(tile_width * (1 - overlap))
    step_y = int(tile_height * (1 - overlap))

    # 计算瓦片数量（考虑重叠）
    num_columns = math.ceil((original_width - tile_width) / step_x) + 1
    num_rows = math.ceil((original_height - tile_height) / step_y) + 1

    output_dir = get_output_dir(img_path)  # 假设这个函数已定义
    base_name = Path(img_path).stem

    for row in range(num_rows):
        for col in range(num_columns):
            # 计算瓦片位置（考虑边界情况）
            left = col * step_x
            upper = row * step_y
            right = min(left + tile_width, original_width)
            lower = min(upper + tile_height, original_height)

            # 如果瓦片超出边界，调整位置使其保持正确大小
            if right - left < tile_width:
                left = max(0, original_width - tile_width)
            if lower - upper < tile_height:
                upper = max(0, original_height - tile_height)

            # 裁剪并保存瓦片
            cropped_image = image.crop((left, upper, right, lower))
            filename = f"{base_name}_tile_{row:04d}_{col:04d}.jpg"
            save_path = output_dir / filename
            cropped_image = cropped_image.convert('RGB')
            cropped_image.save(save_path)
    return output_dir


def clean_tiles(folder_path: str, threshold: float = 0.2) -> None:
    """
    清理切分后的图像，删除背景黑色像素占比超过阈值的图像

    参数:
        folder_path: 包含切分图像的文件夹路径
        threshold: 黑色像素占比阈值（默认0.2，即20%）
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"无效的文件夹路径: {folder_path}")

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    deleted_count = 0

    # 遍历文件夹中的所有图像文件
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                # 打开图像并转换为NumPy数组
                with Image.open(file_path) as img:
                    # 确保图像是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)

                # 计算黑色像素数量（RGB所有通道都为0）
                # 更精确的检测方法：使用位运算减少内存占用
                black_mask = (img_array[..., 0] == 0) & (img_array[..., 1] == 0) & (img_array[..., 2] == 0)
                black_count = np.sum(black_mask)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                black_ratio = black_count / total_pixels

                # 如果黑色像素占比超过阈值，删除图像
                if black_ratio > threshold:
                    os.remove(file_path)
                    deleted_count += 1

            except Exception as e:
                print(f"处理 {file_path.name} 时出错: {str(e)}")
