import os
from PIL import Image

# 图片所在目录
input_dir = r"D:\doc\资料\论文-裂隙提取\论文图\掌子面图片"

# 支持的格式
valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}

# 遍历文件夹
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    # 跳过非文件
    if not os.path.isfile(file_path):
        continue

    # 获取扩展名
    name, ext = os.path.splitext(filename)

    # 跳过非图片
    if ext.lower() not in valid_ext:
        continue

    # 新文件名：原名 + "_seg" + 原扩展名
    new_name = f"{name}_seg{ext}"
    new_path = os.path.join(input_dir, new_name)

    print(f"处理：{filename} → {new_name}")

    # 打开图片并另存为新名称（不改内容，只改名）
    with Image.open(file_path) as img:
        img.save(new_path)

print("处理完成！")
