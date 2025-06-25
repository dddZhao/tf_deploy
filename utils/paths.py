import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def get_project_root():
    """返回项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_input_path(image_name):
    return PROJECT_ROOT / "data" / "input" / image_name

def get_output_json_path(image_name, suffix="_seg"):
    stem = Path(image_name).stem
    return PROJECT_ROOT / "data" / "output" / f"{stem}{suffix}.json"
def get_output_path(image_name, suffix="_seg"):
    stem = Path(image_name).stem
    return PROJECT_ROOT / "data" / "output" / f"{stem}{suffix}.png"

def get_output_dir(
        image_name: str,
        parent_dir: str = "output",
        create_dir: bool = True) -> Path:

    stem = Path(image_name).stem
    output_dir = PROJECT_ROOT / "data" / parent_dir / stem

    if create_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_split_image_path(
        image_name: str,
        row: int,
        col: int,
        suffix: str = "",
        ext: str = "jpg",
        **kwargs  # 传递给 get_output_dir 的参数
) -> Path:
    """
    生成单张切分图片的完整路径（如 output/DyK1218+71.6/DyK1218+71.6_2_4.jpg）

    Args:
        image_name: 原始图片名
        row: 行号（从1开始）
        col: 列号（从1开始）
        suffix: 文件名后缀（如 "_mask"）
        ext: 文件扩展名（默认 "jpg"）
        **kwargs: 传递给 get_output_dir() 的参数

    Returns:
        Path: 完整文件路径对象
    """
    stem = Path(image_name).stem
    filename = f"{stem}_{row}_{col}{suffix}.{ext}"
    output_dir = get_output_dir(image_name, **kwargs)

    return output_dir / filename