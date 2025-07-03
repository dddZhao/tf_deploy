
from ImageTransform import *
from utils import *

def transform_image(input_image_path, input_json_path, output_image_path, target_points_path=None, width=4200):
    """
    执行图像透视变换并保存结果

    参数:
    input_image_path (str): 输入图像路径
    input_json_path (str): 输入JSON标注文件路径
    output_image_path (str): 输出图像路径
    target_points_path (str, optional): 目标轮廓点文件路径
    width (int): 调整图像宽度（默认4200）
    """
    # 如果没有提供目标点文件，使用默认的
    if target_points_path is None:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_points_path = os.path.join(current_dir, "points_scaled.npy")

    # 处理图像和点
    image, points = resize_image_and_points_by_width(input_image_path, input_json_path, width)

    # 均匀分布点
    uniform_points = evenly_distribute_points(points, num_points=100)

    # 找出并延长相对底边
    bottom_edge, m, c = find_bottom_edge(uniform_points, tolerance=100)

    # 对点统一进行四舍五入
    rounded_uniform_points = [round_point(pt) for pt in uniform_points]
    rounded_bottom_edge = set(round_point(pt) for pt in bottom_edge)

    # 排除 bottom_edge 中的点
    uniform_points_without_bottom = [
        pt for pt, rpt in zip(uniform_points, rounded_uniform_points)
        if rpt not in rounded_bottom_edge
    ]
    uniform_points_without_bottom = filter_edge_points(uniform_points_without_bottom)

    # 检测最佳圆弧
    arc_points, center, radius = auto_detect_best_arc(uniform_points_without_bottom, min_arc_length=10)

    # 加载目标轮廓点
    target_points = np.load(target_points_path)
    target_points = np.squeeze(target_points)
    center_target, radius_target = fit_circle(target_points)

    # 寻找源点和目标点
    src_points = find_circle_points(center, radius, m)
    dst_points = find_circle_points(center_target, radius_target, 0)

    # 应用透视变换
    transformed_img = apply_perspective_transform(image, src_points, dst_points)

    # 保存输出图像
    cv2.imwrite(output_image_path, transformed_img)

    # 返回变换后的图像路径
    return output_image_path