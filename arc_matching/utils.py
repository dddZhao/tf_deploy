
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2
import os
import json
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def resize_image_and_points(image_path, json_path, target_width, target_height):
    """
    缩放图像和对应的JSON标注点

    参数:
        image_path (str): 原始图像路径
        json_path (str): JSON标注文件路径
        target_width (int): 目标宽度
        target_height (int): 目标高度

    返回:
        tuple: (缩放后的图像, 缩放后的点列表)
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    original_height, original_width = original_image.shape[:2]

    # 计算缩放比例
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 缩放图像
    resized_image = cv2.resize(
        original_image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    # 读取JSON文件并缩放点
    with open(json_path, 'r') as f:
        data = json.load(f)

    scaled_points = []
    for shape in data['shapes']:
        scaled_shape_points = []
        for point in shape['points']:
            scaled_x = point[0] * width_ratio
            scaled_y = point[1] * height_ratio
            scaled_shape_points.append([scaled_x, scaled_y])
        scaled_points.append(scaled_shape_points)

    return resized_image, scaled_points[0]


def resize_image_and_points_by_width(image_path, json_path, target_width):
    """
    根据目标宽度缩放图像和对应的JSON标注点，高度按原始比例自动计算

    参数:
        image_path (str): 原始图像路径
        json_path (str): JSON标注文件路径
        target_width (int): 目标宽度

    返回:
        tuple: (缩放后的图像, 缩放后的点列表)
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    original_height, original_width = original_image.shape[:2]

    # 计算按比例缩放后的高度
    height_ratio = target_width / original_width
    target_height = int(original_height * height_ratio)

    # 计算缩放比例
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # 缩放图像
    resized_image = cv2.resize(
        original_image,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    # 读取JSON文件并缩放点
    with open(json_path, 'r') as f:
        data = json.load(f)

    scaled_points = []
    for shape in data['shapes']:
        scaled_shape_points = []
        for point in shape['points']:
            scaled_x = point[0] * width_ratio
            scaled_y = point[1] * height_ratio
            scaled_shape_points.append([scaled_x, scaled_y])
        scaled_points.append(scaled_shape_points)

    return resized_image, scaled_points[0]


# 均匀分布点
def evenly_distribute_points(points, num_points=100):
    points = np.array(points)

    # 将首尾连接起来，形成闭合
    closed_points = np.vstack([points, points[0]])

    # 计算每两个点之间的距离（包括最后一段首尾连接）
    segment_lengths = np.linalg.norm(np.diff(closed_points, axis=0), axis=1)

    # 累积距离（包括首点）
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative_lengths[-1]

    # 归一化
    normalized_lengths = cumulative_lengths / total_length

    # 提取 x, y 坐标（注意是闭合的点）
    x_closed = closed_points[:, 0]
    y_closed = closed_points[:, 1]

    # 构造插值函数
    fx = interp1d(normalized_lengths, x_closed, kind='linear')
    fy = interp1d(normalized_lengths, y_closed, kind='linear')

    # 均匀采样的 alpha 值（0~1 之间均匀分布）
    alpha = np.linspace(0, 1, num_points, endpoint=False)

    x_new = fx(alpha)
    y_new = fy(alpha)

    return list(zip(x_new, y_new))


def find_bottom_edge(points, tolerance=5, bottom_fraction=0.2, max_slope=np.tan(np.radians(45))):
    """
    找出相对底边：从靠近图像底部的一批点中，拟合一条不超过 45° 倾斜的直线，
    然后找出距离这条线在 tolerance 范围内的所有点作为 bottom_edge。

    Args:
        points: 点集 (list of (x, y))
        tolerance: 与拟合线的距离容差
        bottom_fraction: 保留 y 最大的 bottom_fraction 作为拟合候选点（默认底部20%）
        max_slope: 最大允许的斜率（默认 45°）

    Returns:
        bottom_edge_points: 拟合底边上的点 (list of (x, y))
        m: 直线斜率
        c: 直线截距
    """
    points = np.array(points)
    y_values = points[:, 1]

    # 1. 找出 y 最大的 bottom_fraction 的点作为拟合候选
    y_threshold = np.percentile(y_values, 100 * (1 - bottom_fraction))
    candidate_points = points[y_values >= y_threshold]

    if len(candidate_points) < 2:
        return [], None, None

    # 2. 拟合直线 y = mx + c
    x_fit = candidate_points[:, 0]
    y_fit = candidate_points[:, 1]
    A = np.vstack([x_fit, np.ones(len(x_fit))]).T
    m, c = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    # 如果斜率太大，认为不是底边
    if abs(m) > max_slope:
        return [], None, None

    # 3. 找出所有点中，距离拟合直线小于 tolerance 的点
    distances = np.abs(m * points[:, 0] - points[:, 1] + c) / np.sqrt(m**2 + 1)
    bottom_edge_points = points[distances <= tolerance]

    # 按 x 排序
    bottom_edge_points = bottom_edge_points[bottom_edge_points[:, 0].argsort()]

    return bottom_edge_points.tolist(), m, c


# 找出相对底边
def find_bottom_edge_old(points, tolerance=5):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 找到最低的y坐标
    bottom_y = np.max(y)

    # 找到所有y坐标接近最低点的点
    bottom_candidates = points[np.abs(y - bottom_y) <= tolerance]

    # 如果候选点不足，直接返回空列表
    if len(bottom_candidates) < 2:
        return []

    # 拟合一条直线 y = mx + c
    x_fit = bottom_candidates[:, 0]
    y_fit = bottom_candidates[:, 1]
    A = np.vstack([x_fit, np.ones(len(x_fit))]).T
    m, c = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    # 计算所有点到拟合直线的垂直距离
    distances = np.abs(m * points[:, 0] - points[:, 1] + c) / np.sqrt(m ** 2 + 1)

    # 找到距离拟合直线在容差范围内的点
    bottom_edge_points = points[distances <= tolerance]

    # 按x坐标排序
    bottom_edge_points = bottom_edge_points[bottom_edge_points[:, 0].argsort()]

    return bottom_edge_points.tolist(), m, c


def detect_continuous_arc_points(points, curvature_threshold=0.1, min_arc_length=10):
    """
    检测点集中可能是圆弧的连续点。

    参数:
        points (list): 点列表，格式为 [(x1, y1), (x2, y2), ...]。
        curvature_threshold (float): 曲率阈值，用于判断是否为弧线点。
        min_arc_length (int): 最小弧线点数量，少于该数量则不认为是圆弧。

    返回:
        arc_points (list): 检测到的连续弧线点列表。
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 计算曲率（简化方法：使用相邻点的夹角）
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x * dy - dx * d2y) / (dx ** 2 + dy ** 2) ** 1.5

    # 找到曲率大于阈值的点
    high_curvature_indices = np.where(curvature > curvature_threshold)[0]

    # 找到连续的弧线点
    arc_points = []
    current_arc = []

    for i in range(len(high_curvature_indices)):
        if i == 0 or high_curvature_indices[i] == high_curvature_indices[i - 1] + 1:
            current_arc.append(points[high_curvature_indices[i]])
        else:
            if len(current_arc) >= min_arc_length:
                arc_points.extend(current_arc)
            current_arc = [points[high_curvature_indices[i]]]

    # 检查最后一段
    if len(current_arc) >= min_arc_length:
        arc_points.extend(current_arc)

    return arc_points


def fit_circle(points):
    """
    使用最小二乘法拟合圆。

    参数:
        points (list): 点列表，格式为 [(x1, y1), (x2, y2), ...]。

    返回:
        center (tuple): 圆心坐标，格式为 (a, b)。
        radius (float): 圆的半径。
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 构造线性方程组
    A = np.vstack([2 * x, 2 * y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2
    a, b, c = np.linalg.lstsq(A, b, rcond=None)[0]

    # 计算圆心和半径
    center = (a, b)
    radius = np.sqrt(a ** 2 + b ** 2 + c)

    return center, radius


def plot_symmetry_axis(image, uniform_points, arc_points, center, radius, m, c,
                       show_circle=True, fit_frame_to_image=True, ax=None):
    """
    绘制均匀分布点、底边拟合直线、对称轴，并可选显示拟合圆

    参数：
    - image: 需要绘制的背景图像
    - uniform_points: 均匀分布的轮廓点 (list of tuples)
    - arc_points: 用于拟合圆的弧线点 (list of tuples)
    - center: 拟合圆的圆心坐标 (tuple)
    - radius: 拟合圆的半径 (float)
    - m: 底边的拟合直线斜率 (float)
    - c: 底边的拟合直线截距 (float)
    - show_circle: 是否显示拟合圆 (bool)
    - fit_frame_to_image: 是否将相框大小按照图像大小设置 (bool)
    - ax: 可选的 matplotlib Axes 对象，如果为 None 则创建新的
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_aspect('equal')  # 保持长宽比

    # 设置坐标轴范围
    if fit_frame_to_image:
        plt.xlim(0, image.shape[1])  # 设置x轴范围
        plt.ylim(image.shape[0], 0)  # 设置y轴范围（注意y轴方向）
    else:
        # 计算能完整显示圆的范围
        x_min = min(center[0] - radius - 20, 0)
        x_max = max(center[0] + radius + 20, image.shape[1])
        y_min = min(center[1] - radius - 20, 0)
        y_max = max(center[1] + radius + 20, image.shape[0])
        plt.xlim(x_min, x_max)
        plt.ylim(y_max, y_min)  # 注意y轴方向

    # 绘制均匀分布的点
    ax.plot([p[0] for p in uniform_points], [p[1] for p in uniform_points], 'ro', label='轮廓点')

    # 拟合直线（底边）
    x_min_points = min(p[0] for p in uniform_points)
    x_max_points = max(p[0] for p in uniform_points)
    x_fit = np.linspace(x_min_points, x_max_points, 100)
    y_fit = m * x_fit + c
    ax.plot(x_fit, y_fit, 'g--', label='拟合底边')

    # 计算对称轴
    if m != 0:  # 斜率不为0，计算对称轴
        k_sym = -1 / m  # 计算对称轴的斜率
        x_fit_sym = np.linspace(center[0] - 100, center[0] + 100, 100)
        y_fit_sym = k_sym * (x_fit_sym - center[0]) + center[1]
        if fit_frame_to_image:
            y_fit_sym = np.clip(y_fit_sym, 0, image.shape[0])  # 限制y范围
    else:  # 底边水平，对称轴是垂直线
        x_fit_sym = np.array([center[0], center[0]])
        y_fit_sym = np.array([center[1] - 150, center[1] + 150])
        if fit_frame_to_image:
            y_fit_sym = np.clip(y_fit_sym, 0, image.shape[0])  # 限制y范围

    # 绘制拟合圆（可选）
    if show_circle and len(arc_points) >= 3:
        circle = plt.Circle(center, radius, color='blue', fill=False, label='拟合圆')
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'bx', markersize=10, label='圆心')  # 标记圆心

    # 绘制对称轴
    ax.plot(x_fit_sym, y_fit_sym, 'm--', linewidth=2, label='对称轴')

    ax.legend()
    plt.title("相对底边与垂直平分线")
    # 如果没有传入ax，则显示图形
    if ax is None:
        plt.show()


def visualize_points_and_circle(points, center=None, radius=None, ax=None, show_y_axis=True, show_x_axis=True):
    """
    可视化点数据、拟合圆及坐标轴

    参数:
        points (numpy.ndarray): 点数据，格式为 [[x1, y1], [x2, y2], ...]
        center (tuple): 拟合圆的圆心坐标，格式为 (a, b)
        radius (float): 拟合圆的半径
        ax (matplotlib.axes.Axes): 可选的Axes对象
        show_y_axis (bool): 是否显示经过圆心的垂直对称轴(y轴)
        show_x_axis (bool): 是否显示经过圆心平行于x轴的水平线
    """
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        show_plot = False

    # 翻转y轴以匹配图像坐标系
    ax.invert_yaxis()

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], 'ro-', markersize=2, label='目标点')

    # 绘制拟合圆和圆心
    if center is not None and radius is not None:
        circle = plt.Circle(center, radius, color='blue', fill=False, label='拟合圆')
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'bx', markersize=10, label='圆心')

        # 获取当前坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # 绘制经过圆心平行于x轴的线（水平线）
        if show_x_axis:
            ax.axhline(y=center[1], color='orange', linestyle=':', linewidth=1.5, label='水平基准线')

        # 绘制经过圆心的垂直对称轴
        if show_y_axis:
            ax.axvline(x=center[0], color='green', linestyle='--', linewidth=1.5, label='垂直对称轴')

        # 恢复原始坐标轴范围
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.set_aspect('equal')
    ax.legend()
    title = "点数据与拟合圆"
    if show_x_axis or show_y_axis:
        title += " (带基准线)" if (show_x_axis and show_y_axis) else (" (带垂直轴)" if show_y_axis else " (带水平线)")
    ax.set_title(title)

    # 如果没有传入ax，则显示图像
    if show_plot:
        plt.show()


def filter_edge_points(points):
    points = np.array(points)
    filtered = points.tolist()  # 先复制原始点

    # 边界值
    x_max = points[:, 0].max()

    # 条件1：x == 0
    x0_points = points[points[:, 0] == 0]
    if len(x0_points) > 3:
        # 找 y 最大和最小的点
        y_max_point = x0_points[np.argmax(x0_points[:, 1])]
        y_min_point = x0_points[np.argmin(x0_points[:, 1])]
        # 删除所有 x == 0 的点，再添加保留的两个
        filtered = [p for p in filtered if p[0] != 0]
        filtered += [y_min_point.tolist()]

    # 条件2：y == 0
    y0_points = points[points[:, 1] == 0]
    if len(y0_points) > 3:
        # 找 x 最大和最小的点
        x_max_point = y0_points[np.argmax(y0_points[:, 0])]
        x_min_point = y0_points[np.argmin(y0_points[:, 0])]
        filtered = [p for p in filtered if p[1] != 0]
        filtered += [x_max_point.tolist(), x_min_point.tolist()]

    # 条件3：x == x_max
    xmax_points = points[points[:, 0] == x_max]
    if len(xmax_points) > 3:
        # 找 y 最大和最小的点
        y_max_point = xmax_points[np.argmax(xmax_points[:, 1])]
        y_min_point = xmax_points[np.argmin(xmax_points[:, 1])]
        filtered = [p for p in filtered if p[0] != x_max]
        filtered += [y_min_point.tolist()]

    return filtered


def compute_fit_error(points, center, radius):
    """计算拟合圆的平均误差"""
    distances = [np.linalg.norm(np.array(p) - np.array(center)) for p in points]
    errors = [abs(d - radius) for d in distances]
    return np.mean(errors)


def auto_detect_best_arc(uniform_points, min_arc_length=20):
    """
    自动选择最佳 curvature_threshold，返回最佳拟合结果。

    Returns:
        arc_points: 连续弧线点
        center: 拟合圆圆心 (x, y)
        radius: 拟合圆半径
    """
    best_threshold = None
    lowest_error = float('inf')
    best_result = None
    best_score = -1

    # 候选 curvature 阈值
    thresholds = np.logspace(-50, -4, num=80)

    uniform_points = [pt for pt in uniform_points if pt[1] < 2500]

    for threshold in thresholds:
        arc_points = detect_continuous_arc_points(uniform_points,
                                                  curvature_threshold=threshold,
                                                  min_arc_length=min_arc_length)
        if len(arc_points) >= 3:
            center, radius = fit_circle(arc_points)
            error = compute_fit_error(arc_points, center, radius)

            score = len(arc_points) / (error + 100)
            if score >= best_score:
                best_score = score
                best_threshold = threshold
                best_result = (arc_points, center, radius)

    if best_result:
        return best_result  # arc_points, center, radius
    else:
        return [], None, None


def save_transformed_images(transformed_img, image_path, dir_name, target_points=None, m=None, c=None):
    """
    保存三种不同版本的变换后图像

    参数:
        image_path: 原始图像路径
        src_points: 源点(用于透视变换)
        dst_points: 目标点(用于透视变换)
        target_points: 目标区域点(用于创建mask)
    """

    # 获取基础文件名
    base_name = os.path.basename(image_path)
    base_name_without_ext = base_name.rsplit('.', 1)[0]

    # 1. 保存原始变换图像
    output_path1 = os.path.join(dir_name, f"{base_name_without_ext}_transformed.png")
    cv2.imwrite(output_path1, transformed_img)

    if target_points is not None:
        # 创建mask
        height, width = transformed_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        target_points_array = np.array([target_points], dtype=np.int32)
        cv2.fillPoly(mask, target_points_array, 255)

        # 如果有直线参数，添加到mask
        if m is not None and c is not None:
            # 创建直线以下的mask
            line_mask = np.zeros((height, width), dtype=np.uint8)

            y_left = int(np.clip(m * 0 + c, 0, height - 1))
            y_right = int(np.clip(m * (width - 1) + c, 0, height - 1))

            poly_points = np.array([[
                [0, y_left],  # 左上（直线）
                [width - 1, y_right],  # 右上（直线）
                [width - 1, height - 1],  # 右下
                [0, height - 1],  # 左下
            ]], dtype=np.int32)
            cv2.fillPoly(line_mask, poly_points, 255)

            mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))

        # 2. 保存透明背景版本
        # 创建BGRA图像(带alpha通道)
        transparent_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2BGRA)
        # 设置mask区域外的部分为透明
        transparent_img[:, :, 3] = mask
        output_path2 = os.path.join(dir_name, f"clear_{base_name_without_ext}_transformed.png")
        cv2.imwrite(output_path2, transparent_img)

        # 3. 保存白色背景版本
        white_img = transformed_img.copy()
        # 反转mask(背景区域为255)
        inverted_mask = cv2.bitwise_not(mask)
        # 将背景设置为白色
        white_img[inverted_mask == 255] = [255, 255, 255]
        output_path3 = os.path.join(dir_name, f"white_{base_name_without_ext}_transformed.png")
        cv2.imwrite(output_path3, white_img)


