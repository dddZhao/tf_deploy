import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import argparse
from utils.utils import *

def main(img_path):
    """
    输入图片，output文件夹中输出_seg.png,_posui.png,_water.png
    :param img_path: 路径中不能有“_”
    :return:
    """
    tunnelface_segmentation(img_path)
    show_seg(img_path)

    tile_size = (256, 256)
    split_images(img_path, tile_size)
    select_images(img_path, tile_size)
    classify_posui(img_path, tile_size)
    show_result_posui(img_path)

    show_result_water(img_path)

    YTLX = classify_YTLX(img_path)
    FHCD = classify_FHCD(img_path)

    YTLX_MAPPING = {
        0: "岩浆岩",
        1: "沉积岩",
        2: "变质岩"
    }

    FHCD_MAPPING = {
        0: "未风化",
        1: "弱风化",
        2: "强风化"
    }

    print(f"岩体类型: {YTLX_MAPPING.get(YTLX, '未知')}")
    print(f"风化程度: {FHCD_MAPPING.get(FHCD, '未知')}")

# 设置命令行参数解析
if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Image processing pipeline for tunnel face analysis.")
    
    # 添加 img_path 参数
    parser.add_argument('img_path', type=str, help="Path to the image file to process.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行主函数
    main(args.img_path)
