import os
import cv2
import shutil

from PIL import Image
from utils import *


if __name__ == "__main__":
    folder_path = 'test'
    
    for file_name in os.listdir(folder_path):
        
        if file_name.endswith(".jpg"):
            img_path = os.path.join(folder_path, file_name)

            tunnelface_segmentation(img_path)
            show_seg(img_path)


