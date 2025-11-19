import cv2
import numpy as np

import json
import matplotlib.pyplot as plt

def find_circle_points(center, radius, x_axis_slope):
    """
    计算圆上指定角度的4个点坐标

    参数:
        center: 圆心坐标 (cx, cy)
        radius: 圆的半径
        x_axis_slope: 基准轴的斜率
    返回:
        np.array: 4个点的坐标数组
    """
    # 计算基准轴的角度
    base_angle = np.arctan(x_axis_slope)

    # 定义4个角度（逆时针）
    angles = np.deg2rad([180, 240, 300, 360])

    # 计算4个点的坐标
    points = np.array([
        [center[0] + radius * np.cos(angle + base_angle),
         center[1] + radius * np.sin(angle + base_angle)]
        for angle in angles
    ])
    return points

def apply_perspective_transform(image, src_points, dst_points):
    """
    应用透视变换并返回完整显示的结果

    参数:
        image: 原始图像
        src_points: 原始图像上的4个点坐标 (np.array, shape=(4,2))
        dst_points: 目标图像上的4个点坐标 (np.array, shape=(4,2))

    返回:
        transformed_image: 变换后的图像
        H: 变换矩阵
    """
    # 确保点坐标是float32类型
    src_pts = src_points.astype(np.float32)
    dst_pts = dst_points.astype(np.float32)

    # 计算透视变换矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # 应用透视变换（带边缘填充）
    transformed_image = cv2.warpPerspective(
        image,
        H,
        (6000, 4000),
        #(4120,4310),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # 黑色填充
    )
    return transformed_image, H

def transform_line_parameters(m, c, src_points, dst_points, image_width):
    """ 变换直线参数 """
    # 将直线转换为齐次坐标表示 [A, B, C] (Ax + By + C = 0)
    line = np.array([m, -1, c])  # mx - y + c = 0

    # 计算透视变换矩阵
    H, _ = cv2.findHomography(np.array(src_points), np.array(dst_points))

    # 变换直线参数 (l' = H^{-T} * l)
    transformed_line = np.dot(np.linalg.inv(H).T, line)

    # 归一化直线方程
    transformed_line /= transformed_line[1]  # 使y系数为-1

    # 提取新的m和c (y = m'x + c')
    new_m = -transformed_line[0]
    new_c = -transformed_line[2]

    return new_m, new_c


def transform_line_params(m, c, H):
    """
    将直线参数(m,c)根据单应性矩阵H进行变换

    参数:
        m: 原始直线斜率
        c: 原始直线截距
        H: 3x3单应性矩阵
        image_width: 原始图像宽度（用于选择参考点）

    返回:
        (m_prime, c_prime): 变换后的直线参数
    """
    # 方法1：使用直线方程变换（更精确）
    # 将直线方程转换为一般式：mx - y + c = 0 → [m, -1, c]
    line_vector = np.array([m, -1, c])

    # 计算变换后的直线：l' = H^{-T} * l
    H_inv_T = np.linalg.inv(H).T
    transformed_line = H_inv_T @ line_vector

    # 从一般式转换回斜截式
    a, b, c_prime = transformed_line
    if abs(b) > 1e-6:  # 避免除以零
        m_prime = -a / b
        c_prime = -c_prime / b
    else:  # 垂直线情况
        m_prime = float('inf')
        c_prime = -c_prime / a  # 此时c_prime表示x截距

    return m_prime, c_prime
