import os
import cv2
import shutil
import argparse
from PIL import Image
from utils import *


def delete_json(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error: {e}")

def main(img_path):
    tunnelface_segmentation(img_path)
    calculate_and_save_images(img_path)
    process_folder(img_path)
    classify_posui(img_path)
    show_result_posui(img_path)
    show_result_water(img_path)
    
    # 定义分类结果的映射关系
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
    
    YTLX = classify_YTLX(img_path)
    FHCD = classify_FHCD(img_path)
    
    # 打印分类结果
    print(f"岩石类型 (YTLX): {YTLX_MAPPING.get(YTLX, '未知')}")
    print(f"风化程度 (FHCD): {FHCD_MAPPING.get(FHCD, '未知')}")
    
    folder_path = img_path.split('.')[0]
    folder_path2 = img_path.split('.')[0] + '_select'
    
    try:
        shutil.rmtree(folder_path)
        shutil.rmtree(folder_path2)
    except Exception as e:
        print(f"Error: {e}")
    
    folder_path = img_path.split('/')[0]
    delete_json(folder_path)

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
