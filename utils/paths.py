import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
#PROJECT_ROOT = Path(r"E:\crack_dataset")

DERIVED_IMAGE_SUFFIXES = (
    "_black_transformed",
    "_clear_transformed",
    "_transformed",
)

def get_project_root():
    """返回项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_input_path(image_name):
    return PROJECT_ROOT / "data" / "input" / image_name

def get_output_stem(image_name):
    stem = Path(image_name).stem
    changed = True
    while changed:
        changed = False
        for suffix in DERIVED_IMAGE_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True
                break
    return stem

def get_image_output_root(image_name, parent_dir="output", create_dir=True):
    output_dir = PROJECT_ROOT / "data" / parent_dir / get_output_stem(image_name)
    if create_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_output_json_path(image_name, suffix="_seg"):
    stem = Path(image_name).stem
    if suffix:
        filename = f"{stem}{suffix}.json"
    else:
        filename = f"{stem}.json"
    return get_image_output_root(image_name) / filename

def get_output_path(image_name, suffix="_seg"):
    stem = Path(image_name).stem
    if suffix:
        filename = f"{stem}{suffix}.png"
    else:
        filename = f"{stem}.png"
    return get_image_output_root(image_name) / filename

def get_output_dir(
        image_name: str,
        parent_dir: str = "output",
        create_dir: bool = True) -> Path:

    stem = Path(image_name).stem
    output_dir = get_image_output_root(image_name, parent_dir, create_dir=False) / stem

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
