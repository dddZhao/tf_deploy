import os
import re
from PIL import Image

# 设置图像所在的文件夹路径
folder_path = r'D:\Users\Z\Downloads\predict'

# 正则表达式匹配文件名中的列行信息和总列行数（注意顺序）
pattern = re.compile(r'DK468\+227_(\d+)_(\d+)_(\d+)x(\d+)\.png')

# 读取所有符合规则的图像文件
tiles = []
max_row = max_col = 0
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        col, row, total_cols, total_rows = map(int, match.groups())
        filepath = os.path.join(folder_path, filename)
        tiles.append((row, col, filepath))  # 这里的 row 和 col 顺序调整了
        max_row = max(max_row, total_rows)
        max_col = max(max_col, total_cols)

# 如果找不到图像，给出提示
if not tiles:
    print("未找到符合命名规则的图像文件！")
    exit()

# 读取第一个图像，获取尺寸
sample_img = Image.open(tiles[0][2])
tile_width, tile_height = sample_img.size

# 创建一个空白图像用于拼接
full_image = Image.new('RGB', (tile_width * max_col, tile_height * max_row))

# 将每个图像粘贴到对应位置
for row, col, filepath in tiles:
    img = Image.open(filepath)
    full_image.paste(img, (col * tile_width, row * tile_height))

# 保存拼接后的图像
output_path = os.path.join(folder_path, 'merged.png')
full_image.save(output_path)
print(f"图像已成功拼接并保存至：{output_path}")
