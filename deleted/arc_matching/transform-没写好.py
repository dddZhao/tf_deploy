from .utils import *
from utils.utils import mask_seg
from .ImageTransform import apply_perspective_transform, find_circle_points, transform_line_params
import copy

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

    # 2. 处理图像和标注轮廓点
    image, shape_points = resize_image_and_points_by_width(input_image_path, input_json_path, width)

    # 3. 加载原始标注
    with open(input_json_path, 'r') as f:
        annotation = json.load(f)

    # 4. 执行处理流程（均匀分布点、检测底边等）
    uniform_points = evenly_distribute_points(shape_points, num_points=100)
    uniform_points = np.array(uniform_points)
    bottom_edge, m, c = find_bottom_edge(uniform_points, tolerance=100)

    rounded_uniform_points = [round_point(pt) for pt in uniform_points]
    rounded_bottom_edge = set(round_point(pt) for pt in bottom_edge)

    uniform_points_without_bottom = [
        pt for pt, rpt in zip(uniform_points, rounded_uniform_points)
        if rpt not in rounded_bottom_edge
    ]
    uniform_points_without_bottom = filter_edge_points(uniform_points_without_bottom)

    arc_points, center, radius = auto_detect_best_arc(uniform_points_without_bottom, min_arc_length=15)

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
    # 逐个 shape 处理：仅对确有 points 的 shape 做缩放+透视，并按原索引写回
    for idx, shp in enumerate(annotation.get("shapes", [])):
        pts = shp.get("points")
        # 透视到输出坐标
        pts_np = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)   # [N,1,2]
        pts_warp = cv2.perspectiveTransform(pts_np, H).reshape(-1, 2)     # [N,2]
        # 按原索引写回
        annotation["shapes"][idx]["points"] = pts_warp.tolist()

    # 9. 更新并保存变换后的标注
    annotation["imagePath"] = os.path.basename(output_image_path)
    annotation["imageHeight"] = transformed_img.shape[0]
    annotation["imageWidth"] = transformed_img.shape[1]

    # 生成输出JSON路径
    output_json_path = output_image_path.replace(".png", ".json")
    with open(output_json_path, 'w') as f:
        json.dump(annotation, f, indent=4)

    # 保存变换后图像
    cv2.imwrite(output_image_path, transformed_img)
    masked_path = mask_seg(output_image_path, suffix="")
    img = cv2_imread(masked_path)
    if target_points is not None:
        # 创建mask
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        target_points_array = np.array([target_points], dtype=np.int32)
        cv2.fillPoly(mask, target_points_array, 255)
        # 设置mask区域外的部分为黑色
        black_img = cv2.bitwise_and(img, img, mask=mask)
        output_path2 = output_image_path.replace("_transformed.png", "_black_transformed.png")
        cv2.imwrite(output_path2, black_img)

    return output_path2