# import numpy as np
# import matplotlib.pyplot as plt

# # 假设我们有一个随机生成的5x5矩阵
# matrix = np.random.randn(5, 5) * 3  # 生成一些数据，使得有些数可能大于1

# # 创建图像
# plt.figure(figsize=(6, 6))

# # 绘制整个矩阵作为背景
# plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')

# # 在矩阵值大于1的位置绘制红点
# # 首先找到这些点的坐标
# points = np.where(matrix > 1)

# # 然后把这些点绘制上去
# plt.scatter(points[1], points[0], color='red')  # 注意scatter的坐标顺序是(y, x)，所以我们用了points[1], points[0]

# # 添加颜色条以供参考
# plt.colorbar()

# import os

# # 获取当前文件的路径
# current_file_path = os.path.abspath(__file__)

# print("当前文件的完整路径是:", current_file_path)

# # plt.savefig("try.png")

import matplotlib.pyplot as plt

# # 创建一个示例图形
plt.plot([1, 2, 3], [4, 5, 6])
# plt.title("示例图形")

# 获取当前脚本所在目录，并组合成目标子文件夹路径

# 组合图片保存路径

# 保存图形
# plt.savefig(output_path)

# print(f"图片已保存至: {output_path}")