import os
import json
import re
from shapely.geometry import Point, Polygon
from PIL import Image

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
def is_image_512x512(image_path):
    image = Image.open(image_path)
    width, height = image.size

    if width == 512 and height == 512:
        return True
    else:
        return False

#判断是否在内部
def is_square_inside_polygon(square_vertices, polygon_vertices):
    square = Polygon(square_vertices)
    polygon = Polygon(polygon_vertices)

    if polygon.contains(square):
        return True
    else:
        return False


# 判断是否相交
def intersect_square_polygon(square_vertices, polygon_vertices):
    square = Polygon(square_vertices)
    polygon = Polygon(polygon_vertices)

    if square.intersects(polygon):
        # 计算相交面积
        intersection = square.intersection(polygon)
        intersection_area = intersection.area
        if intersection_area > 512*512*0.8:
            return True
        else:
            return False

