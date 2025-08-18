import argparse
from pathlib import Path
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from arc_matching.transform import transform_image

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def process_images(folder_path):
    """仅执行segmentation和visualization的批量处理"""
    folder = Path(folder_path)
    image_extensions = ('*.jpg', '*.jpeg', '*.png')

    for ext in image_extensions:
        for img_path in folder.glob(ext):
            try:
                logging.info(f"Processing: {img_path.name}")
                output_json_path = tunnelface_segmentation(str(img_path))
                stem = Path(img_path).stem
                output_image_path = str(PROJECT_ROOT / "data" / "output" / f"{stem}_transformed.png")
                transformed_img_path = transform_image(
                    input_image_path=img_path,
                    input_json_path=output_json_path,
                    output_image_path=output_image_path
                )
                tile_size = (512, 512)
                show_seg(str(img_path))
                split_images(transformed_img_path, tile_size)
                select_images(transformed_img_path, tile_size)
            except Exception as e:
                logging.error(f"Failed {img_path}: {str(e)}")
                continue


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help="包含图片的文件夹路径")
    args = parser.parse_args()

    if not Path(args.folder).is_dir():
        logging.error("必须提供有效文件夹路径")
        exit(1)

    process_images(args.folder)