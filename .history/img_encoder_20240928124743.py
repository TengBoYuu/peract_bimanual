import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据示例
right = np.array([0.0043, 0.0049, 0.0487, 0.0161, 0.0039, 0.0256, 0.0131, 0.4327, 0.0977,
                  0.1135, 0.0578, 0.0101, 0.0152, 0.0583, 0.0115, 0.0291, 0.0087, 0.0488])

left = np.array([0.0266, 0.0510, 0.2489, 0.0389, 0.0051, 0.0623, 0.0514, 0.0409, 0.0498,
                 0.0613, 0.0241, 0.0741, 0.0124, 0.0238, 0.0434, 0.0588, 0.0623, 0.0649])

# 生成平滑曲线的函数
def smooth_curve(x, y, num_points=300):
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)  # Cubic interpolation
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# 生成颜色列表（颜色循环）
def get_colors(num_bars):
    return plt.colormaps.get_cmap('tab20', num_bars)  # 使用新的方法

# 创建两个独立的图
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 生成 X 轴值
x_vals = np.arange(len(right))

# 为右侧柱状图生成颜色
right_colors = get_colors(len(right))

# 绘制右边的数据（柱状图 + 平滑曲线）
axs[0].bar(x_vals, right, alpha=0.6, color=right_colors(range(len(right))))
x_smooth, y_smooth = smooth_curve(x_vals, right)
axs[0].plot(x_smooth, y_smooth, color='blue', label='Smooth Curve')
axs[0].set_title('Right Data')
axs[0].set_xticks([])  # 移除 X 轴刻度
axs[0].set_yticks([])  # 移除 Y 轴刻度

# 为左侧柱状图生成颜色
left_colors = get_colors(len(left))

# 绘制左边的数据（柱状图 + 平滑曲线）
axs[1].bar(x_vals, left, alpha=0.6, color=left_colors(range(len(left))))
x_smooth, y_smooth = smooth_curve(x_vals, left)
axs[1].plot(x_smooth, y_smooth, color='red', label='Smooth Curve')
axs[1].set_title('Left Data')
axs[1].set_xticks([])  # 移除 X 轴刻度
axs[1].set_yticks([])  # 移除 Y 轴刻度

# 调整布局并显示图像
plt.tight_layout()
plt.show()
plt.savefig("skill.png")