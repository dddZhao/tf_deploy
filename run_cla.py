from my_utils.utils_forcla import *
import argparse

def main(img_path):
    calculate_and_save_images(img_path)
    process_folder(img_path)
    classify_posui(img_path)
    show_result_posui(img_path)

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