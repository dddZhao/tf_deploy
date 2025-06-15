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

def tunnelface_segmentation(input_file):
    engine = load_engine('model/seg_swin.engine')
    img,orig_size = infer_seg(engine, input_file)
    
    img = img.resize((orig_size[0],orig_size[1]),Image.Resampling.LANCZOS)
    array = np.array(img)
    
    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt,True), True) for cnt in contours]
    
    # 处理为json格式
    result = {
      "version": "0.2.23",
      "flags": {},
      "shapes": [],  
      "imageData": {},
      "imageHeight": orig_size[1],
      "imageWidth": orig_size[0]
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
    json_path = input_file.split('.')[0] + '_seg.json' 
    with open(json_path, 'w') as f:
        json.dump(result, f)

# clip image for one pic
def calculate_and_save_images(img_path):
    
    file_path = img_path
    file_name = (img_path.split('/')[-1]).split('.')[0]
    output_folder = os.path.join(os.path.dirname(img_path),file_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = Image.open(file_path)
    original_width, original_height = image.size
    num_columns = math.ceil(original_width / 512)
    num_rows = math.ceil(original_height / 512)

    for row in range(num_rows):
        for column in range(num_columns):
            left = column * 512
            upper = row * 512
            right = min(left + 512, original_width)
            lower = min(upper + 512, original_height)
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


# filter the images in the TF
def process_folder(img_path):
    
    file_name = (img_path.split('/')[-1]).split('.')[0]
    folder_path = os.path.join(os.path.dirname(img_path),file_name)
    output_folder_name = (img_path.split('/')[-1]).split('.')[0]+ '_select' 
    output_folder_path = os.path.join(os.path.dirname(img_path),output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    json_path = img_path.split('.')[0] + '_seg.json' 
    json_data = parse_json(json_path)
    
    image_files = []
    picnum = 0
    piecenum = 0
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".jpg"):
            image_files.append(file_path)   
            
    if json_check(json_data):
        for i in range(len(json_data["shapes"])):
            polygon_points = json_data["shapes"][i]["points"]
            picnum = 0
            for image_file in image_files:
                if is_image_512x512(image_file):
                    picnum = picnum + 1
                    image_name = os.path.basename(image_file)
                    num, column, row, m, n = re.findall(r"\d+", image_name)
                    img = Image.open(image_file)

                    column = int(column)
                    row = int(row)
                    left = column * 512
                    upper = row * 512
                    right = left + 512
                    lower = upper + 512
                    img_points = [(left,lower),(left,upper),(right,upper),(right,lower)]

                    if is_square_inside_polygon(img_points,polygon_points):
                        output_file_path = os.path.join(output_folder_path, image_name)
                        img.save(output_file_path)
                        piecenum = piecenum + 1
                    elif intersect_square_polygon(img_points,polygon_points):
                        output_file_path = os.path.join(output_folder_path, image_name)
                        img.save(output_file_path)
                        piecenum = piecenum + 1
        print(f"{piecenum} out of {picnum} images were screened")


def classify_posui(img_path):  
    engine = load_engine('model/cla_posui.engine')
    dir_name = (img_path.split('/')[-1]).split('.')[0]+ '_select'
    img_dir = os.path.join(os.path.dirname(img_path),dir_name)

    original_image = cv2.imread(img_path)
    original_height, original_width = original_image.shape[:2]

    shapes = []
    for filename in os.listdir(img_dir):
        parts = filename.split("_")
        row = int(parts[2])
        col = int(parts[1])
        x = col * 512
        y = row * 512
        output = infer_cla(engine, os.path.join(img_dir,filename))
        pred = np.argmax(output)
        shape = {
            "label": pred,
            "text": "",
            "points": [[x, y], [x, y + 512], [x + 512, y + 512], [x + 512, y]],
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
    json_str = json.dumps(json_data,default=int)
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
      0: ((139, 250, 146),'Block', '完整结构'),
      1: ((201, 212, 74),'Layered', '层状结构'),
      2: ((235, 197, 94),'Mosaic', '镶嵌结构'), 
      3: ((214, 135, 86),'Fractured', '碎裂结构'),
      4: ((245, 95, 115),'Granular', '散体结构')
    }
    TF_filepath = img_path.split('.')[0] + "_seg.json"
    with open(TF_filepath, 'r', encoding='utf-8') as f:
        data_tf = json.load(f)
    
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
        polygon = mpatches.Polygon(points, closed=True,edgecolor=color, facecolor=color, alpha=0.4)  # 设置填充颜色和透明度
        ax.add_patch(polygon)

    # 绘制掌子面轮廓
    tf = data_tf["shapes"]
    for i in range(len(tf)):
        tf_points = tf[i]['points']
        # Create a polygon patch
        polygon = mpatches.Polygon(tf_points, closed=True, facecolor='none', edgecolor=(0, 1, 0))
        ax.add_patch(polygon)

    # 添加图例
    patches = []
    for i in range(len(label_map2)):
        color, en_name, cn_name = label_map2[i]
        color = [c / 255.0 for c in color]
        patches.append(mpatches.Patch(color=color, label=cn_name))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax.legend(loc='upper right',handles=patches,prop=font)

    # 设置坐标轴隐藏
    ax.axis('off')
    fig.savefig(output_path,bbox_inches='tight')
    plt.close(fig)

def classify_water(img_path):  
    engine = load_engine('model/cla_posui.engine')
    dir_name = (img_path.split('/')[-1]).split('.')[0]+ '_select'
    img_dir = os.path.join(os.path.dirname(img_path),dir_name)

    original_image = cv2.imread(img_path)
    original_height, original_width = original_image.shape[:2]

    shapes = []
    for filename in os.listdir(img_dir):
        parts = filename.split("_")
        row = int(parts[2])
        col = int(parts[1])
        x = col * 512
        y = row * 512
        output = infer_cla(engine, os.path.join(img_dir,filename))
        pred = np.argmax(output)
        shape = {
            "label": pred,
            "text": "",
            "points": [[x, y], [x, y + 512], [x + 512, y + 512], [x + 512, y]],
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
    json_filepath = img_path.split('.')[0] + "_water.json"
    json_str = json.dumps(json_data,default=int)
    with open(json_filepath, 'w') as f:
        f.write(json_str)

# display classification results
def show_result_water(img_path):

    engine = load_engine('model/water.engine')
    img_seg, orig_size = infer_seg(engine, img_path)

    img_seg = img_seg.resize((orig_size[0],orig_size[1]),Image.Resampling.LANCZOS)
    array_seg = np.array(img_seg)

    # 创建一个空的 RGB 图像，和 array_seg 的形状一样
    height, width = array_seg.shape
    waterseg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个值赋颜色
    waterseg[array_seg == 0] = [0, 0, 0]     
    waterseg[array_seg == 1] = [245, 95, 115]   
    waterseg[array_seg == 2] = [0, 255, 0]  

    img = cv2.imread(img_path)
    img_array = np.array(img)
    json_path = img_path.split('.')[0] + '_seg.json' 
    with open(json_path, "r", encoding="utf-8") as f:  # 替换为JSON文件路径
        data = json.load(f)
    output_path = img_path.split('.')[0] + "_water.png"    

    # 提取多边形点
    polygon_points = data["shapes"][0]["points"]
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # 创建掩码
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    # 创建半透明效果
    alpha = 0.5  # 半透明强度
    segmented_overlay = cv2.addWeighted(waterseg, alpha, img_array, 1 - alpha, 0)
    
    # 将多边形区域显示为半透明覆盖
    result = img_array.copy()
    result[mask == 255] = segmented_overlay[mask == 255]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    # 绘制掌子面轮廓
    tf = data["shapes"]
    for i in range(len(tf)):
        tf_points = tf[i]['points']
        # Create a polygon patch
        polygon = mpatches.Polygon(tf_points, closed=True, facecolor='none', edgecolor=(0, 1, 0))
        ax.add_patch(polygon)
    
    # 设置坐标轴隐藏
    ax.axis('off')
    fig.savefig(output_path,bbox_inches='tight')
    plt.close(fig)


def classify_YTLX(img_path):  
    engine = load_engine('model/cla_YTLX.engine')
    output = infer_cla2(engine, img_path)
    pred = np.argmax(output)
    return(pred)


def classify_FHCD(img_path):  
    engine = load_engine('model/cla_FHCD.engine')
    output = infer_cla2(engine, img_path)
    pred = np.argmax(output)
    return(pred)




# display classification results
def show_seg(img_path):
    image = cv2.imread(img_path)
    output_path = img_path.split('.')[0] + "_seg.png"

    TF_filepath = img_path.split('.')[0] + "_seg.json"
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
    fig.savefig(output_path,bbox_inches='tight')
    plt.close(fig)








            
