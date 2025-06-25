import os
import json
import re
from shapely.geometry import Point, Polygon
from PIL import Image
from typing import Tuple, List
from pathlib import Path

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

def is_image_correct_size(image_path: Path, tile_size: Tuple[int, int]) -> bool:
    """检查图片尺寸是否匹配切分尺寸"""
    with Image.open(image_path) as img:
        return img.size == tile_size

def calculate_bbox(col: int, row: int, width: int, height: int) -> List[Tuple[int, int]]:
    """计算切分图片的边界框坐标"""
    return [
        (col * width, (row + 1) * height),  # 左下
        (col * width, row * height),        # 左上
        ((col + 1) * width, row * height),  # 右上
        ((col + 1) * width, (row + 1) * height)  # 右下
    ]

def is_square_inside_polygon(square_vertices, polygon_vertices) -> bool:
    """判断方形是否完全在多边形内"""
    return Polygon(polygon_vertices).contains(Polygon(square_vertices))


# 判断是否相交

def intersects_polygon(square_vertices, polygon_vertices, min_overlap_ratio: float = 0.75) -> bool:
    """判断方形与多边形是否相交且重叠面积超过阈值"""
    intersection = Polygon(square_vertices).intersection(Polygon(polygon_vertices))
    return intersection.area > (Polygon(square_vertices).area * min_overlap_ratio)