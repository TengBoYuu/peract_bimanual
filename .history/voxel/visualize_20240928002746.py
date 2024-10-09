import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义体素网格的大小为 100x100x100
voxel_shape = (100, 100, 100)

# 生成随机的体素数据（True 表示该位置有体素）
voxels = np.random.rand(*voxel_shape) > 0.5  # 随机生成 0 或 1 的体素

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用 matplotlib 的 voxels 方法来渲染体素
ax.voxels(voxels, edgecolor='k')

# 添加标题
ax.set_title("100x100x100 Voxel Visualization")

# 保存为 EPS 矢量图
plt.savefig('voxel_plot_100x100x100.png', format='png')

# 显示绘图
plt.show()