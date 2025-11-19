from PIL import Image
import numpy as np
import cv2
import os
from datetime import date
import math
from typing import Optional, Tuple
import shutil
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='SimHei.ttf')

from .trt import *
from .utils_Selectimg import *
from .paths import *
from .cn_cv import *

def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def tunnelface_segmentation(input_file):
    # 路径设置
    model_path = os.path.join(get_project_root(), "model", "seg_tf.engine")
    output_json_path = get_output_json_path(input_file, suffix="_seg")

    engine = load_engine(model_path)
    img, orig_size = infer_seg(engine, input_file)
    img = img.resize(orig_size, Image.Resampling.LANCZOS)
    array = np.array(img)

    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt,True), True) for cnt in contours]

    # 处理为json格式
    result = {
      "SegDate": str(date.today()),
      "shapes": [],
      "imageHeight": orig_size[1],
      "imageWidth": orig_size[0]
    }
    min_contour_area = orig_size[0] * orig_size[1] * 0.05
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            shape = {
                "label": "TunnelFace",
                "points": cnt.squeeze().tolist(),
                "shape_type": "polygon"
            }
            result["shapes"].append(shape)
    with open(output_json_path, 'w') as f:
        json.dump(result, f)
    return output_json_path

def show_seg(img_path):
    image = cv2_imread(img_path)
    output_path = get_output_path(img_path, suffix="_seg")
    TF_filepath = get_output_json_path(img_path, suffix="_seg")

    with open(TF_filepath, 'r', encoding='utf-8') as f:
        data_tf = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

    return output_path

def mask_seg(img_path, suffix="_seg"):
    # 读取原始图像
    image = cv2_imread(img_path)  # 假设返回BGR格式
    output_path = get_output_path(img_path, suffix=suffix)
    TF_filepath = get_output_json_path(img_path, suffix=suffix)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 解析JSON标注数据
    with open(TF_filepath, 'r', encoding='utf-8') as f:
        data_tf = json.load(f)

    # 填充掌子面区域为白色（255）
    for shape in data_tf["shapes"]:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=255)

    result = cv2.bitwise_and(image, image, mask=mask)

    # 保存结果图像
    cv2.imwrite(output_path, result)
    return output_path

def split_images(
    img_path: str,
    tile_size: Tuple[int, int] = (512, 512)
) -> None:

    image = Image.open(img_path)
    original_width, original_height = image.size
    tile_width, tile_height = tile_size

    num_columns = math.ceil(original_width / tile_width)
    num_rows = math.ceil(original_height / tile_height)

    tile_metadata = {}
    output_dir = get_output_dir(img_path)
    base_name = Path(img_path).stem

    for row in range(num_rows):
        for col in range(num_columns):
            left = col * tile_width
            upper = row * tile_height
            right = min(left + tile_width, original_width)
            lower = min(upper + tile_height, original_height)

            cropped_image = image.crop((left, upper, right, lower))
            filename = f"{base_name}_tile_{row:04d}_{col:04d}.jpg"
            save_path = output_dir / filename
            cropped_image.save(save_path)
            tile_metadata[filename] = (row, col)

    # 保存元数据到JSON
    #metadata_path = output_dir / f"{base_name}_tile_metadata.json"
    #with open(metadata_path, 'w') as f:
    #    json.dump(tile_metadata, f)
    return output_dir

# filter the images in the TF
def select_images(
        img_path: str,
        tile_size: Tuple[int, int] = (512, 512)
) -> None:
    """基于元数据JSON而非文件名解析来选择图像"""
    input_folder = get_output_dir(img_path)
    output_folder = Path(f"{input_folder}_select")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. 加载切片元数据
    base_name = Path(img_path).stem
    metadata_path = input_folder / f"{base_name}_tile_metadata.json"

    try:
        with open(metadata_path) as f:
            tile_metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"切片元数据未找到: {metadata_path}")

    # 2. 加载JSON标注
    json_path = img_path.replace("_black_transformed.png", ".png")
    json_path = get_output_json_path(json_path, suffix="_seg")
    #json_path = img_path.replace(".png", "_seg.json")
    try:
        json_data = parse_json(json_path)
    except Exception as e:
        raise ValueError(f"JSON解析失败: {str(e)}")

    if not json_check(json_data):
        raise ValueError("JSON标注中没有有效多边形")

    selected_count = 0
    total_valid_images = 0
    tile_width, tile_height = tile_size

    # 3. 处理每个切片
    for image_file in input_folder.glob("*.jpg"):
        total_valid_images += 1

        if not is_image_correct_size(image_file, tile_size):
            continue

        filename = image_file.name

        # 从元数据获取行列信息
        if filename not in tile_metadata:
            logging.warning(f"文件中缺少元数据: {filename}")
            continue

        row, col = tile_metadata[filename]
        img_bbox = calculate_bbox(col, row, tile_width, tile_height)

        for shape in json_data["shapes"]:
            polygon = shape["points"]
            if (is_square_inside_polygon(img_bbox, polygon) or
                    intersects_polygon(img_bbox, polygon, min_overlap_ratio=0.75)):
                shutil.copy2(image_file, output_folder / filename)
                selected_count += 1
                break

    logging.info(f"筛选完成: {selected_count}/{total_valid_images} 图片被选中")

def classify_posui(
        img_path: str,
        tile_size: Tuple[int, int] = (512, 512)
) -> None:
    """使用元数据JSON替代文件名解析"""
    model_path = os.path.join(get_project_root(), "model", "cla_posui.engine")
    img_dir = Path(f"{get_output_dir(img_path)}_select")

    # 1. 加载切片元数据
    base_name = Path(img_path).stem
    metadata_path = get_output_dir(img_path) / f"{base_name}_tile_metadata.json"

    try:
        with open(metadata_path) as f:
            tile_metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"切片元数据未找到: {metadata_path}")

    engine = load_engine(model_path)
    original_image = cv2_imread(img_path)
    original_height, original_width = original_image.shape[:2]

    shapes = []
    tile_width, tile_height = tile_size
    missing_metadata = 0

    # 2. 处理每个图像文件
    for img_file in img_dir.glob("*.jpg"):
        # 获取文件名（不含路径）
        filename = img_file.name

        # 从元数据中获取行列信息
        if filename not in tile_metadata:
            logging.warning(f"缺少元数据条目: {filename}")
            missing_metadata += 1
            continue

        row, col = tile_metadata[filename]
        x = col * tile_width
        y = row * tile_height

        # 执行模型推理
        output = infer_cla(engine, str(img_file))
        pred = np.argmax(output)

        # 构建形状数据
        shape = {
            "label": int(pred),
            "points": [
                [x, y],
                [x, y + tile_height],
                [x + tile_width, y + tile_height],
                [x + tile_width, y]
            ],
            "shape_type": "polygon"
        }
        shapes.append(shape)

    if missing_metadata > 0:
        logging.warning(f"警告: {missing_metadata}个切片缺少元数据")

    # 3. 保存结构JSON
    json_data = {
        "shapes": shapes,
        "imagePath": img_path,
        "imageHeight": original_height,
        "imageWidth": original_width,
        "metadata": f"基于元数据: {metadata_path}"  # 添加可追溯性
    }

    json_path = get_output_json_path(img_path, "_structure")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    logging.info(f"结构分类完成，共处理{len(shapes)}个切片")

LABEL_MAP = {
    0: ((139, 250, 146), 'Block', '完整结构'),
    1: ((201, 212, 74), 'Layered', '层状结构'),
    2: ((235, 197, 94), 'Mosaic', '镶嵌结构'),
    3: ((214, 135, 86), 'Fractured', '碎裂结构'),
    4: ((245, 95, 115), 'Granular', '散体结构')
}

def show_result_posui(
        img_path: str
) -> None:
    """可视化结构分类结果"""
    # 添加错误处理以防文件丢失
    try:
        json_path = get_output_json_path(img_path, "_structure")
        tf_json_path = get_output_json_path(img_path, "_seg")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(tf_json_path, 'r', encoding='utf-8') as f:
            data_tf = json.load(f)

    except FileNotFoundError as e:
        logging.error(f"关键文件缺失: {str(e)}")
        return

    image = cv2_imread(str(img_path))
    if image is None:
        logging.error(f"无法读取图像: {img_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 1. 绘制结构多边形
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in LABEL_MAP:
            continue

        color, _, cn_name = LABEL_MAP[label]
        norm_color = tuple(c / 255.0 for c in color)
        points = np.array(shape['points'], dtype=np.int32)

        # 添加多边形到图表
        polygon = mpatches.Polygon(
            points,
            closed=True,
            edgecolor=norm_color,
            facecolor=norm_color,
            alpha=0.4
        )
        ax.add_patch(polygon)

    # 2. 绘制掌子面轮廓
    for shape in data_tf.get("shapes", []):
        points = np.array(shape['points'], dtype=np.int32)
        polygon = mpatches.Polygon(
            points,
            closed=True,
            facecolor='none',
            edgecolor=(0, 1, 0),
            linewidth=2
        )
        ax.add_patch(polygon)

    # 3. 创建图例
    patches = []
    for label_id, (color, _, cn_name) in LABEL_MAP.items():
        norm_color = tuple(c / 255.0 for c in color)
        patches.append(mpatches.Patch(color=norm_color, label=cn_name))

    patches.append(mpatches.Patch(facecolor='none', edgecolor=(0, 1, 0), label='隧道面轮廓'))

    # 4. 设置字体和图例
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    ax.legend(loc='upper right', handles=patches)
    ax.axis('off')

    # 5. 保存结果
    output_path = get_output_path(img_path, "_posui")
    fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
    plt.close(fig)
    logging.info(f"可视化结果已保存至: {output_path}")

WATER_COLOR_MAP = {
    0: [0, 0, 0],        # 背景
    1: [245, 95, 115],   # 渗水区域1
    2: [0, 255, 0]       # 渗水区域2
}

WATER_LABEL_MAP = {
    1: ("渗水", (245, 95, 115)),
    2: ("涌水", (0, 255, 0))
}

def show_result_water(
    img_path: str
) -> None:

    model_path = os.path.join(get_project_root(), "model", "seg_water.engine")
    json_path = get_output_json_path(img_path, "_seg")
    output_path = get_output_path(img_path, "_water")

    engine = load_engine(model_path)
    img_seg, orig_size = infer_seg(engine, img_path)

    img_seg = img_seg.resize(orig_size,Image.Resampling.LANCZOS)
    array_seg = np.array(img_seg)

    # 创建彩色分割图
    height, width = array_seg.shape
    seg_color = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in WATER_COLOR_MAP.items():
        seg_color[array_seg == value] = color

    img = cv2_imread(img_path)
    img_array = np.array(img)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygon_points = data["shapes"][0]["points"]
    polygon = np.array(polygon_points, dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    # 创建半透明效果
    alpha = 0.5  # 半透明强度
    overlay = cv2.addWeighted(seg_color, alpha, img, 1 - alpha, 0)
    result = img_array.copy()
    result[mask == 255] = overlay[mask == 255]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    # 绘制掌子面轮廓
    for shape in data["shapes"]:
        points = np.array(shape['points'], dtype=np.int32)
        polygon_patch = mpatches.Polygon(
            points,
            closed=True,
            facecolor='none',
            edgecolor=(0, 1, 0),
            linewidth=2
        )
        ax.add_patch(polygon_patch)
    # 设置坐标轴隐藏
    patches = [
        mpatches.Patch(color=np.array(color) / 255, label=label)
        for label, (label_name, color) in WATER_LABEL_MAP.items()
    ]
    patches.append(mpatches.Patch(facecolor='none', edgecolor=(0, 1, 0), label='隧道面轮廓'))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax.legend(loc='upper right', handles=patches)
    ax.axis('off')

    # 10. 保存结果
    fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
    plt.close(fig)

def classify_YTLX(img_path):
    model_path = os.path.join(get_project_root(), "model", "cla_ytlx.engine")
    engine = load_engine(model_path)
    output = infer_cla2(engine, img_path)
    pred = np.argmax(output)
    return(pred)

def classify_FHCD(img_path):
    model_path = os.path.join(get_project_root(), "model", "cla_fhcd.engine")
    engine = load_engine(model_path)
    output = infer_cla2(engine, img_path)
    pred = np.argmax(output)
    return(pred)



            
