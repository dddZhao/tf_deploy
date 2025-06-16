from PIL import Image
import numpy as np
import cv2
import json
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='SimHei.ttf')

from trt import *
from utils_Selectimg import *


def calculate_and_save_images(img_path):
    file_path = img_path
    file_name = (img_path.split('/')[-1]).split('.')[0]
    output_folder = os.path.join(os.path.dirname(img_path), file_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = Image.open(file_path)
    original_width, original_height = image.size
    num_columns = math.ceil(original_width / 500)
    num_rows = math.ceil(original_height / 500)

    for row in range(num_rows):
        for column in range(num_columns):
            left = column * 500
            upper = row * 500
            right = min(left + 500, original_width)
            lower = min(upper + 500, original_height)
            cropped_image = image.crop((left, upper, right, lower))

            new_file_name = f"{file_name.split('_')[0]}_{column}_{row}_{num_columns}x{num_rows}_IMAGES.jpg"
            new_file_path = os.path.join(output_folder, new_file_name)
            cropped_image.save(new_file_path)

    # 创建并保存包含MxN信息和文件名的txt文件
    txt_file_path = os.path.join(output_folder, 'mxn_info.txt')
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"File Name: {file_name}\n")
        txt_file.write(f"M: {num_columns}\n")
        txt_file.write(f"N: {num_rows}\n")
        txt_file.write("\n")

def load_target_point(path):
    '''加载目标轮廓点'''
    points = np.load(path)
    #points[:, 0, 1] -= 100
    return(points)

# filter the images in the TF
def process_folder(img_path):
    file_name = (img_path.split('/')[-1]).split('.')[0]
    folder_path = os.path.join(os.path.dirname(img_path), file_name)
    output_folder_name = (img_path.split('/')[-1]).split('.')[0] + '_select'
    output_folder_path = os.path.join(os.path.dirname(img_path), output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    target_point = r"D:\pycharm\correction\pythonProject\arc_matching\points3_scaled.npy"
    polygon_points = load_target_point(target_point)

    image_files = []
    picnum = 0
    piecenum = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".jpg"):
            image_files.append(file_path)

    picnum = 0
    for image_file in image_files:
        picnum = picnum + 1
        image_name = os.path.basename(image_file)
        num, column, row, m, n = re.findall(r"\d+", image_name)
        img = Image.open(image_file)

        column = int(column)
        row = int(row)
        left = column * 500
        upper = row * 500
        right = left + 500
        lower = upper + 500
        img_points = [(left, lower), (left, upper), (right, upper), (right, lower)]

        if row<6:
            if is_square_inside_polygon(img_points, polygon_points):
                output_file_path = os.path.join(output_folder_path, image_name)
                img.save(output_file_path)
                piecenum = piecenum + 1
            elif intersect_square_polygon(img_points, polygon_points):
                output_file_path = os.path.join(output_folder_path, image_name)
                img.save(output_file_path)
                piecenum = piecenum + 1
    print(f"{piecenum} out of {picnum} images were screened")


def classify_posui(img_path):
    engine = load_engine('model/posui.engine')
    dir_name = (img_path.split('/')[-1]).split('.')[0] + '_select'
    img_dir = os.path.join(os.path.dirname(img_path), dir_name)

    original_image = cv2.imread(img_path)
    original_height, original_width = original_image.shape[:2]

    shapes = []
    for filename in os.listdir(img_dir):
        parts = filename.split("_")
        row = int(parts[2])
        col = int(parts[1])
        x = col * 500
        y = row * 500
        output = infer_cla(engine, os.path.join(img_dir, filename))
        pred = np.argmax(output)
        shape = {
            "label": pred,
            "text": "",
            "points": [[x, y], [x, y + 500], [x + 500, y + 500], [x + 500, y]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    json_data = {
        "version": "0.2.23",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageHeight": original_height,
        "imageWidth": original_width,
        "text": ""
    }
    json_filepath = img_path.split('.')[0] + "_structure.json"
    json_str = json.dumps(json_data, default=int)
    with open(json_filepath, 'w') as f:
        f.write(json_str)


# display classification results
def show_result_posui(img_path):
    # 读取JSON文件
    json_filepath = img_path.split('.')[0] + "_structure.json"
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image = cv2.imread(img_path)
    output_path = img_path.split('.')[0] + "_Posui.png"
    label_map2 = {
        0: ((139, 250, 146), 'Block', '完整结构'),
        1: ((201, 212, 74), 'Layered', '层状结构'),
        2: ((235, 197, 94), 'Mosaic', '镶嵌结构'),
        3: ((214, 135, 86), 'Fractured', '碎裂结构'),
        4: ((245, 95, 115), 'Granular', '散体结构')
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 遍历shapes列表，绘制多边形形状并填充内部
    for shape in data["shapes"]:
        label = shape["label"]
        color, en_name, cn_name = label_map2[label]
        # 提取多边形顶点
        points = shape['points']
        points = np.array(points, dtype=np.int32)

        color = [c / 255.0 for c in color]
        # 绘制多边形形状
        polygon = mpatches.Polygon(points, closed=True, edgecolor=color, facecolor=color, alpha=0.4)  # 设置填充颜色和透明度
        ax.add_patch(polygon)

    # 添加图例
    patches = []
    for i in range(len(label_map2)):
        color, en_name, cn_name = label_map2[i]
        color = [c / 255.0 for c in color]
        patches.append(mpatches.Patch(color=color, label=cn_name))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax.legend(loc='upper right', handles=patches, prop=font)

    # 设置坐标轴隐藏
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
