from utils import *
from ImageTransform import *
import matplotlib
import os
import glob
matplotlib.use('TkAgg')


def round_point(pt, digits=6):
    return tuple(round(coord, digits) for coord in pt)
def main():
    # 设置目标文件夹路径
    folder_path = r"E:\crack_dataset\tf\新建文件夹"

    # 获取所有 .jpg 文件路径
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

    # 遍历所有 .jpg 文件
    for image_path in jpg_files:
        try:
            # 构造对应的 json 路径（假设命名规则为 xxx.jpg → xxx_seg.json）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_name = base_name + "_seg.json"
            json_path = os.path.join(folder_path, json_name)
            # 图像路径和JSON路径
            image, points = resize_image_and_points_by_width(image_path, json_path, 4200)

            # 均匀分布点
            uniform_points = evenly_distribute_points(points, num_points=100)
            # 找出并延长相对底边
            bottom_edge, m, c = find_bottom_edge(uniform_points, tolerance=100)

            # 对两个点集中的点统一进行四舍五入
            rounded_uniform_points = [round_point(pt) for pt in uniform_points]
            rounded_bottom_edge = set(round_point(pt) for pt in bottom_edge)

            # 排除 bottom_edge 中的点
            uniform_points_without_bottom = [pt for pt, rpt in zip(uniform_points, rounded_uniform_points) if
                                             rpt not in rounded_bottom_edge]
            uniform_points_without_bottom = filter_edge_points(uniform_points_without_bottom)
            # 检测连续弧线点
            arc_points, center, radius = auto_detect_best_arc(uniform_points_without_bottom, min_arc_length=10)

            # 加载目标轮廓点
            target_points = np.load('points2_scaled.npy')  # 目标轮廓点
            target_points = np.squeeze(target_points)
            center_target, radius_target = fit_circle(target_points)

            src_points = find_circle_points(center, radius, m)
            dst_points = find_circle_points(center_target, radius_target, 0)

            transformed_img = apply_perspective_transform(image, src_points, dst_points)
            dir_name = os.path.join(os.path.dirname(image_path), "transform") # 获取目录路径
            os.makedirs(dir_name, exist_ok=True)
            m, c = transform_line_parameters(m, c, src_points, dst_points, 6000)
            #base_name = os.path.basename(image_path)  # 获取文件名
            #output_path = os.path.join(dir_name, f"{base_name.rsplit('.', 1)[0]}_transformed.png")
            #cv2.imwrite(output_path, transformed_img)
            save_transformed_images(transformed_img, image_path, dir_name, target_points, m, c)
        except Exception as e:
            print(f"处理出错: {image_path}")
            print(f"错误信息: {e}")

if __name__ == "__main__":
    main()
