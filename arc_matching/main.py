from utils import *
from ImageTransform import *
import json
import matplotlib.image as mpimg
import matplotlib
import os
import argparse  # 新增：用于命令行参数解析
import sys  # 新增：用于系统路径操作
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def round_point(pt, digits=6):
    return tuple(round(coord, digits) for coord in pt)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='隧道面图像处理工具')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像文件路径（相对deploy目录）')
    parser.add_argument('--json', type=str,
                        help='JSON标注文件路径（可选）')
    parser.add_argument('--output', type=str, default="../data/output/",
                        help='输出目录（默认../data/output/）')
    parser.add_argument('--target', type=str, default="points_scaled.npy",
                        help='目标点文件（默认points_scaled.npy）')
    parser.add_argument('--width', type=int, default=4200,
                        help='调整图像宽度（默认4200）')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示绘图结果')
    return parser.parse_args()

def resolve_paths(base_dir,args):
    """解析所有路径为绝对路径"""
    # 获取当前脚本所在目录（arc_matching）
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建deploy目录路径
    deploy_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # 解析输入路径
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(deploy_dir, input_path)

    # 解析JSON路径（如未提供则自动生成）
    if args.json:
        json_path = args.json
        if not os.path.isabs(json_path):
            json_path = os.path.join(deploy_dir, json_path)
    else:
        # 自动生成JSON路径：同名+_seg.json
        dir_name = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        json_path = os.path.join(dir_name, f"{name_without_ext}_seg.json")

    # 解析输出目录
    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(deploy_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件路径
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}_transformed.png")

    # 解析目标点文件路径
    target_path = args.target
    if not os.path.isabs(target_path):
        # 目标点文件默认在arc_matching目录
        target_path = os.path.join(current_dir, target_path)

    return input_path, json_path, output_path, target_path


def process_image(image_path, json_path, target_path, width=4200, no_plot=False):
    # 加载目标轮廓点
    target_points = np.load(target_path)
    target_points = np.squeeze(target_points)
    center_target, radius_target = fit_circle(target_points)

    # 处理图像
    image, points = resize_image_and_points_by_width(image_path, json_path, width)

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

    # 检测最佳圆弧
    arc_points, center, radius = auto_detect_best_arc(uniform_points_without_bottom, min_arc_length=10)

    # 寻找源点和目标点
    src_points = find_circle_points(center, radius, m)
    dst_points = find_circle_points(center_target, radius_target, 0)

    # 绘图（可选）
    if not no_plot:
        plot_1(image, uniform_points, arc_points, center, radius, m, c,
               src_points, target_points, center_target, radius_target, dst_points)

    # 应用透视变换
    transformed_img = apply_perspective_transform(image, src_points, dst_points)

    # 返回变换后的图像和路径信息
    return transformed_img, image, arc_points, center, radius, m, src_points


def save_and_show_result(transformed_img, original_img, output_path, arc_points, center, radius, m, src_points,
                         no_plot=False):
    # 保存输出图像
    cv2.imwrite(output_path, transformed_img)
    print(f"处理完成！输出文件已保存至: {output_path}")

    # 显示结果（可选）
    if not no_plot:
        # 创建简单的结果展示
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        # 原始图像
        ax[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        # 变换后的图像
        ax[1].imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Transformed Image")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    args = parse_arguments()

    # 解析所有路径
    input_path, json_path, output_path, target_path = resolve_paths(os.path.dirname(__file__),args)

    # 处理图像
    transformed_img, original_img, arc_points, center, radius, m, src_points = process_image(
        input_path, json_path, target_path, args.width, args.no_plot
    )

    # 保存并显示结果
    save_and_show_result(
        transformed_img, original_img, output_path,
        arc_points, center, radius, m, src_points, args.no_plot
    )
def plot_1(image, uniform_points, arc_points, center, radius, m, c,src_points,target_points, center_target, radius_target,dst_points):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    plt.sca(ax1)
    plot_symmetry_axis(image, uniform_points, arc_points, center, radius, m, c,
                       show_circle=True, fit_frame_to_image=False, ax=ax1)
    ax1.set_title("原始图像")

    ax1.scatter(src_points[:, 0], src_points[:, 1], s=80, marker='o',
                facecolors='none', edgecolors='yellow', linewidths=2)
    for i, (x, y) in enumerate(src_points):
        ax1.text(x, y, f'S{i + 1}', color='white', ha='center', va='center',
                 fontsize=10, bbox=dict(facecolor='red', alpha=0.7, boxstyle='circle'))

    plt.sca(ax2)
    visualize_points_and_circle(target_points, center_target, radius_target, ax=ax2)
    ax2.set_title("目标形状")
    ax2.scatter(dst_points[:, 0], dst_points[:, 1], s=80, marker='o',
                facecolors='none', edgecolors='yellow', linewidths=2)
    for i, (x, y) in enumerate(dst_points):
        ax2.text(x, y, f'D{i + 1}', color='white', ha='center', va='center',
                 fontsize=10, bbox=dict(facecolor='blue', alpha=0.7, boxstyle='circle'))

    plt.tight_layout()
    plt.show()

def plot_2(image,transformed_img,dst_points,target_points,center_target, radius_target):
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 原始图像和点
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 变换后的完整图像

    ax2.imshow(transformed_img)

    visualize_points_and_circle(target_points, center_target, radius_target, ax=ax2)
    ax2.scatter(dst_points[:, 0], dst_points[:, 1], s=80, marker='o',
                facecolors='none', edgecolors='yellow', linewidths=2)
    ax2.invert_yaxis()
    plt.title("Transformed Image")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
