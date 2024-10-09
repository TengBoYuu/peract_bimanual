import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据示例
right = np.array([0.01276664, 0.03088504, 0.0078185, 0.02223782, 0.02765255,
 0.01432539, 0.07736668, 0.07268056, 0.04530919, 0.06282461,
 0.06804, 0.08737376, 0.02679457, 0.06076052, 0.04170031,
 0.00083492, 0.10571442, 0.09227412])

left = np.array([0.06970922, 0.08886764, 0.02614825, 0.07639663, 0.030265144,
 0.0228037, 0.00183122, 0.00653663, 0.0445297, 0.010100639,
 0.04775079, 0.03478806, 0.0110981, 0.00356232, 0.02453193,
 0.00877989, 0.10778402, 0.00122408])

# 固定18种不同颜色的调色板
color_palette = plt.get_cmap('tab20').colors[:18]  # 提取固定的18种颜色

# X 轴值
x_vals = np.arange(len(right))

# 平滑曲线生成
def smooth_curve(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
    # 保证曲线不低于柱状图的最小值
    y_smooth = np.clip(y_smooth, a_min=y.min(), a_max=None)
    return x_smooth, y_smooth

x_smooth_right, y_smooth_right = smooth_curve(x_vals, right)
x_smooth_left, y_smooth_left = smooth_curve(x_vals, left)

# 创建图形
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 右边柱状图和平滑曲线
axs[0].bar(x_vals, right, alpha=0.6, color=[color_palette[i] for i in range(len(right))])
axs[0].plot(x_smooth_right, y_smooth_right, linestyle='--', color='darkgray')  # 使用深灰色虚线
axs[0].set_title('Right Data')
axs[0].set_xlabel('Skill')  # 设置横坐标标签为 'Skill'
axs[0].set_ylabel('Weight')  # 设置纵坐标标签为 'Weight'
axs[0].set_xticks([])  # 移除 X 轴刻度
axs[0].set_yticks([])  # 移除 Y 轴刻度

# 移除顶部和右侧的黑框，但保留左侧和底部的坐标轴
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# 左边柱状图和平滑曲线
axs[1].bar(x_vals, left, alpha=0.6, color=[color_palette[i] for i in range(len(left))])
axs[1].plot(x_smooth_left, y_smooth_left, linestyle='--', color='darkgray')  # 使用深灰色虚线
axs[1].set_title('Left Data')
axs[1].set_xlabel('Skill')  # 设置横坐标标签为 'Skill'
axs[1].set_ylabel('Weight')  # 设置纵坐标标签为 'Weight'
axs[1].set_xticks([])  # 移除 X 轴刻度
axs[1].set_yticks([])  # 移除 Y 轴刻度

# 移除顶部和右侧的黑框，但保留左侧和底部的坐标轴
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# 调整布局并显示图像
plt.tight_layout()
plt.show()
plt.savefig("da.png")