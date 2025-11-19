from .utils import *
from .ImageTransform import apply_perspective_transform, find_circle_points, transform_line_params
from .cn_cv import *

def round_point(pt, digits=6):
    return tuple(round(coord, digits) for coord in pt)

def transform_image(input_image_path, input_json_path, output_image_path,
                    target_points_path=None, width=4200):
    """
    执行图像和标注的透视变换，用于整体变换

    参数:
        input_image_path: 输入图像路径
        input_json_path: 输入JSON标注路径
        output_image_path: 输出图像路径
        target_points_path: 目标点文件路径(可选)
        width: 图像调整宽度(默认4200)

    返回:
        tuple: (输出图像路径, 输出JSON路径)
    """
    # 1. 处理目标点路径
    if target_points_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_points_path = os.path.join(current_dir, "points_scaled.npy")

    # 2. 处理图像和点
    image, points = resize_image_and_points_by_width(input_image_path, input_json_path, width)

    # 3. 加载原始标注
    with open(input_json_path, 'r') as f:
        annotation = json.load(f)

    # 提取所有形状的points
    shape_points = []
    for shape in annotation["shapes"]:
        if "points" in shape:
            shape_points.append(np.array(shape["points"], dtype=np.float32))

    # 4. 执行处理流程（均匀分布点、检测底边等）
    uniform_points = evenly_distribute_points(points, num_points=100)
    uniform_points = np.array(uniform_points)
    bottom_edge, m, c = find_bottom_edge(uniform_points, tolerance=200)

    rounded_uniform_points = [round_point(pt) for pt in uniform_points]
    rounded_bottom_edge = set(round_point(pt) for pt in bottom_edge)

    uniform_points_without_bottom = [
        pt for pt, rpt in zip(uniform_points, rounded_uniform_points)
        if rpt not in rounded_bottom_edge
    ]
    uniform_points_without_bottom = filter_edge_points(uniform_points_without_bottom)

    arc_points, center, radius = auto_detect_best_arc(uniform_points_without_bottom, min_arc_length=10)

    # 5. 加载目标轮廓点
    target_points = np.load(target_points_path)
    target_points = np.squeeze(target_points)
    center_target, radius_target = fit_circle(target_points)

    # 6. 寻找源点和目标点
    src_points = find_circle_points(center, radius, m)
    dst_points = find_circle_points(center_target, radius_target, 0)

    # 7. 应用透视变换到图像
    transformed_img, H = apply_perspective_transform(image, src_points, dst_points)

    # 8. 应用相同变换到轮廓点
    for i, points in enumerate(shape_points):
        # 将点转换为3D格式 [n_points, 1, 2]
        points_3d = points.reshape(-1, 1, 2)

        # 应用透视变换
        transformed_pts = cv2.perspectiveTransform(points_3d, H)

        # 转换回2D格式并更新到原annotation中
        annotation["shapes"][i]["points"] = transformed_pts.reshape(-1, 2).tolist()

    # 9. 更新并保存变换后的标注
    annotation["imagePath"] = os.path.basename(output_image_path)
    annotation["imageHeight"] = transformed_img.shape[0]
    annotation["imageWidth"] = transformed_img.shape[1]

    # 生成输出JSON路径
    output_json_path = output_image_path.replace(".png", "_seg.json")
    with open(output_json_path, 'w') as f:
        json.dump(annotation, f, indent=4)

    # 10. 保存变换后的图像
    output_path1 = output_image_path
    cv_imwrite_cn(output_path1, transformed_img)

    if target_points is not None:
        # 创建mask
        height, width = transformed_img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        target_points_array = np.array([target_points], dtype=np.int32)
        cv2.fillPoly(mask, target_points_array, 255)

        if m is not None and c is not None:
            m, c = transform_line_params(m, c, H)

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
        output_path2 = output_image_path.replace("_transformed.png", "_clear_transformed.png")
        cv_imwrite_cn(output_path2, transparent_img)

        black_img = transformed_img.copy()
        inverted_mask = cv2.bitwise_not(mask)
        black_img[inverted_mask == 255] = 0  # 使用标量0广播赋值，效率更高
        output_path3 = output_image_path.replace("_transformed.png", "_black_transformed.png")
        cv_imwrite_cn(output_path3, black_img)

    return output_path3