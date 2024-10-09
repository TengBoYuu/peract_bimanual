import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据示例
right = np.array([0.08971294, 0.08947658, 0.03115267, 0.08312011, 0.01527105,
 0.07873582, 0.07802318, 0.0319122 , 0.0138005 , 0.07247028,
 0.0426523 , 0.07999989, 0.01702121, 0.05233093, 0.05445319,
 0.06243377, 0.05872655, 0.04870684])

left = np.array([0.0198, 0.1058, 0.2701, 0.0338, 0.0047, 0.0613, 0.0520, 0.0222, 0.0369,                    
         0.1116, 0.0322, 0.0532, 0.0088, 0.0149, 0.0336, 0.0430, 0.0500, 0.0461])

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