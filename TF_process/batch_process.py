import argparse
from pathlib import Path
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

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
                tunnelface_segmentation(str(img_path))
                show_seg(str(img_path))
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