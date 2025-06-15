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
    with Image.open(image_path) as img:
        return img.size == (512, 512)

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
    return intersection.area > 512 * 512 * 0.75
