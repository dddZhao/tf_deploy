import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from utils.utils import *
from arc_matching.transform import transform_image

PROJECT_ROOT = Path(__file__).parent.parent

def main(img_path, tile_size=(512, 512)):
    # 全局日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # 参数验证
    if not Path(img_path).exists():
        logging.error(f"输入文件不存在: {img_path}")
        return

    # 统一异常处理
    try:
        start_time = time.time()
        logging.info(f"开始处理图像: {img_path}")

        # 处理流程
        output_json_path = tunnelface_segmentation(img_path)

        stem = Path(img_path).stem
        output_image_path = str(PROJECT_ROOT / "data" / "output" / f"{stem}_transformed.png")

        transformed_img_path = transform_image(
            input_image_path=img_path,
            input_json_path=output_json_path,
            output_image_path=output_image_path
        )

        show_seg(img_path)

        split_images(transformed_img_path, tile_size)
        select_images(transformed_img_path, tile_size)
        classify_posui(transformed_img_path, tile_size)
        show_result_posui(transformed_img_path)
        show_result_water(transformed_img_path)

        # 分类结果
        YTLX = classify_YTLX(transformed_img_path)
        FHCD = classify_FHCD(transformed_img_path)

        # 分类映射
        YTLX_MAPPING = {0: "岩浆岩", 1: "沉积岩", 2: "变质岩"}
        FHCD_MAPPING = {0: "未风化", 1: "弱风化", 2: "强风化"}

        # 结果输出
        result_msg = (f"岩体类型: {YTLX_MAPPING.get(YTLX, '未知')}\n"
                      f"风化程度: {FHCD_MAPPING.get(FHCD, '未知')}")
        print(result_msg)
        logging.info(result_msg)

        # 性能统计
        elapsed = time.time() - start_time
        logging.info(f"处理完成! 耗时: {elapsed:.2f}秒")

    except Exception as e:
        logging.error(f"处理失败: {str(e)}", exc_info=True)
        sys.exit(1)

# 设置命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隧道面图像处理流程")
    parser.add_argument('img_path', help="输入图像路径")

    parser.add_argument('--tile_size', type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=[512, 512],
                        help="切片尺寸 (默认: %(default)s)")

    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="日志级别")

    args = parser.parse_args()

    # 转换tile_size为元组
    tile_size = tuple(args.tile_size)

    # 运行主函数并传递所有参数
    main(args.img_path, tile_size=tile_size)