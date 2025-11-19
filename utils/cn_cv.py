import os
import cv2
import numpy as np

def cv_imwrite_cn(path, img):
    """
    支持中文路径的 cv2.imwrite 封装
    """
    # 保证是字符串
    path = str(path)

    # 取扩展名（.png / .jpg 等）
    ext = os.path.splitext(path)[1]
    if ext == "":
        # 没有扩展名默认用 .png
        ext = ".png"
        path = path + ext

    # 用 imencode 把图像编码成二进制
    ok, encoded_img = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cv_imwrite_cn: 无法编码图像，路径: {path}")

    # 用 numpy 的 tofile 写入磁盘（支持中文路径）
    encoded_img.tofile(path)


def cv_imread_cn(path, flags=cv2.IMREAD_COLOR):
    """
    支持中文路径的 cv2.imread 封装（如果你有中文路径读图的需求）
    """
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"cv_imread_cn: 无法读取图像，路径: {path}")
    return img
