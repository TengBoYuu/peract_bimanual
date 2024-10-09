import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据示例
right = np.array([0.0043, 0.0049, 0.0487, 0.0161, 0.0039, 0.0256, 0.0131, 0.4327, 0.0977,
                  0.1135, 0.0578, 0.0101, 0.0152, 0.0583, 0.0115, 0.0291, 0.0087, 0.0488])

left = np.array([0.0266, 0.0510, 0.2489, 0.0389, 0.0051, 0.0623, 0.0514, 0.0409, 0.0498,
                 0.0613, 0.0241, 0.0741, 0.0124, 0.0238, 0.0434, 0.0588, 0.0623, 0.0649])

# 固定的颜色映射
right_colors = ['#1f77b4'] * len(right)  # 蓝色固定
left_colors = ['#ff7f0e'] * len(left)    # 橙色固定

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
axs[0].bar(x_vals, right, alpha=0.6, color=right_colors)
axs[0].plot(x_smooth_right, y_smooth_right, color='blue')
axs[0].set_title('Right Data')
axs[0].set_xticks([])  # 移除 X 轴刻度
axs[0].set_yticks([])  # 移除 Y 轴刻度

# 左边柱状图和平滑曲线
axs[1].bar(x_vals, left, alpha=0.6, color=left_colors)
axs[1].plot(x_smooth_left, y_smooth_left, color='orange')
axs[1].set_title('Left Data')
axs[1].set_xticks([])  # 移除 X 轴刻度
axs[1].set_yticks([])  # 移除 Y 轴刻度

# 调整布局并显示图像
plt.tight_layout()
plt.show()
plt.savefig("da.png")