from cut_utils import *

import traceback

def remove_filename_spaces(path):
    """如果路径中包含空格，移除空格并重命名文件"""
    if ' ' in path:
        new_path = path.replace(' ', '')
        try:
            os.rename(path, new_path)
            print(f"重命名文件: '{path}' -> '{new_path}'")
            return new_path
        except Exception as e:
            print(f"重命名失败 [{path}]: {str(e)}")
            return path
    return path

if __name__ == "__main__":
    folder_path = r'E:\crack_dataset\tf\part\b'

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        original_path = os.path.join(folder_path, file_name)
        img_path = remove_filename_spaces(original_path)

        try:
            tunnelface_segmentation(img_path)
            calculate_and_save_images(img_path)
            process_folder(img_path)
            show_seg(img_path)

        except Exception as e:
            error_msg = f"处理失败 [{file_name}]: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
