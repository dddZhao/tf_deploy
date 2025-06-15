from utils import *

if __name__ == "__main__":

    folder_path = r'E:\ContinuousTunnelface\test\zhuziqing\transform1'

    for file_name in os.listdir(folder_path):

        if file_name.endswith("227_transformed.png"):
            img_path = os.path.join(folder_path, file_name)
            calculate_and_save_images(img_path)

