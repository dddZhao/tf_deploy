import argparse
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TF_process.run import process_images, setup_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help="包含图片的文件夹路径")
    parser.add_argument('--tile_size', type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=[512, 512],
                        help="切片尺寸 (默认: %(default)s)")
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="日志级别")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if not Path(args.folder).is_dir():
        print("必须提供有效文件夹路径", file=sys.stderr)
        exit(1)

    process_images(
        args.folder,
        tile_size=tuple(args.tile_size),
        full_pipeline=False
    )
