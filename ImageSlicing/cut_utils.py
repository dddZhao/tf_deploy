import cv2
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='SimHei.ttf')

import os
import json
from shapely.geometry import Point, Polygon
from PIL import Image

from trt import *
import shutil

def tunnelface_segmentation(input_file):
    engine = load_engine('model/seg.engine')
    img, orig_size = infer_seg(engine, input_file)

    long_edge = max(orig_size[0], orig_size[1])
    long_edge_size = 2496
    scale = long_edge_size / long_edge

    # 计算缩放后的尺寸
    new_w = round(orig_size[0] * scale)
    new_h = round(orig_size[1] * scale)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    array = np.array(img)

    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt,True), True) for cnt in contours]

    # 处理为json格式
    result = {
      "SegDate": "2025-5-27",
      "shapes": [],
      "imageHeight": new_w,
      "imageWidth": new_h
    }
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (orig_size[1]*orig_size[0] // 20):
            shape = {"label": "TunnelFace",
                     "points": cnt.reshape(-1,2).tolist(),
                     "group_id": {},
                     "shape_type": "polygon",
                     "flags": {}
                    }
            result["shapes"].append(shape)
    json_path = os.path.splitext(input_file)[0] + '_seg.json'
    with open(json_path, 'w') as f:
        json.dump(result, f)


def calculate_and_save_images(img_path):
    output_folder = os.path.splitext(img_path)[0]

    filename = os.path.basename(img_path)
    filename = os.path.splitext(filename)[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = Image.open(img_path)
    orig_size = image.size

    long_edge = max(orig_size)
    long_edge_size = 2496
    scale = long_edge_size / long_edge
    new_w = round(orig_size[0] * scale)
    new_h = round(orig_size[1] * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)  # 高质量缩放

    num_columns = math.ceil(new_w / 416)
    num_rows = math.ceil(new_h / 416)

    for row in range(num_rows):
        for column in range(num_columns):
            left = column * 416
            upper = row * 416
            right = min(left + 416, new_w)
            lower = min(upper + 416, new_h)
            cropped_image = image.crop((left, upper, right, lower)).convert('RGB')

            new_file_name = f"{filename}_{column}_{row}.jpg"
            new_file_path = os.path.join(output_folder, new_file_name)
            cropped_image.save(new_file_path)


def process_folder(img_path):
    folder_path = os.path.splitext(img_path)[0]  # 原文件夹路径
    output_folder_path = folder_path + '_select'  # 输出文件夹路径
    json_path = folder_path + '_seg.json'  # JSON标注文件路径

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    json_data = parse_json(json_path)

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".jpg")
    ]

    total_images = 0
    selected_images = 0

    for shape in json_data["shapes"]:
        polygon_points = shape["points"]

        for image_file in image_files:
            # ---------- 1 检查图片尺寸 ----------
            if is_image_416x416(image_file):

                total_images += 1
                image_name = os.path.basename(image_file)

                # ---------- 2 解析行列号 ----------
                try:
                    name_without_ext = os.path.splitext(image_name)[0]
                    *_, column, row = name_without_ext.rsplit("_", 2)
                    column, row = int(column), int(row)
                except (ValueError, IndexError):
                    print(f"跳过文件名格式错误的图片: {image_name}")
                    continue

                # ---------- 3 计算图片位置 ----------
                left = column * 416
                upper = row * 416
                img_points = [
                    (left, upper + 416),  # 左下
                    (left, upper),  # 左上
                    (left + 416, upper),  # 右上
                    (left + 416, upper + 416)  # 右下
                ]

                # ---------- 4 检查是否在多边形内或相交 ----------
                with Image.open(image_file) as img:  # 使用 with 自动关闭句柄
                    if (is_square_inside_polygon(img_points, polygon_points) or
                            intersect_square_polygon(img_points, polygon_points)):
                        output_path = os.path.join(output_folder_path, image_name)
                        img.save(output_path)
                        selected_images += 1

    # ========== 5. 输出结果并清理 ==========
    print(f"{selected_images} out of {total_images} images were screened")

    try:
        shutil.rmtree(folder_path)  # 删除原文件夹
    except PermissionError as e:
        print(f"无法删除文件夹 {folder_path}，错误: {e}")


def show_seg(img_path):

    try:
        image = Image.open(img_path)
        orig_size = image.size  # 原始尺寸 (width, height)

        long_edge = max(orig_size)
        long_edge_size = 2496
        scale = long_edge_size / long_edge
        new_w = round(orig_size[0] * scale)
        new_h = round(orig_size[1] * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)  # 高质量缩放

        # 使用 PIL 读取图片
        image = np.array(image)      # 转换为 NumPy 数组供 Matplotlib 使用
        if image.size == 0:
            raise ValueError("图片内容为空")
    except Exception as e:
        print(f"图片加载失败: {e}")
        return

    output_path = os.path.splitext(img_path)[0] + "_seg.png"
    TF_filepath = os.path.splitext(img_path)[0] + "_seg.json"
    with open(TF_filepath, 'r', encoding='utf-8') as f:
        data_tf = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    # 绘制掌子面轮廓
    tf = data_tf["shapes"]
    for i in range(len(tf)):
        tf_points = tf[i]['points']
        # Create a polygon patch
        polygon = mpatches.Polygon(tf_points, closed=True, facecolor='none', edgecolor=(0, 1, 0))
        ax.add_patch(polygon)

    # 设置坐标轴隐藏
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    os.remove(TF_filepath)


# 解析JSON文件&检查 "shapes"
def parse_json(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def json_check(json_data):
    if "shapes" in json_data:
        shapes = json_data["shapes"]
        if len(shapes) > 0:
            return True

# 判断图像尺寸是否为 512x512
def is_image_416x416(image_path):
    with Image.open(image_path) as img:
        return img.size == (416, 416)

#判断是否在内部
def is_square_inside_polygon(square_vertices, polygon_vertices):
    square = Polygon(square_vertices)
    polygon = Polygon(polygon_vertices)
    return polygon.contains(square)


# 判断是否相交
def intersect_square_polygon(square_vertices, polygon_vertices):
    square = Polygon(square_vertices)
    polygon = Polygon(polygon_vertices)
    intersection = square.intersection(polygon)
    return intersection.area > 416 * 416 * 0.85
