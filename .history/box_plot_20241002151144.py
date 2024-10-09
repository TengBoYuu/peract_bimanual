import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
# 定义数据
# data1 = [8.00, 16.00, 4.00, 40.00, 4.00, 0.00, 0.00, 12.00, 0.00, 0.00, 0.00, 24.00]
# data2 = [8.00, 32.00, 24.00, 20.00, 12.00, 16.00, 8.00, 84.00, 8.00, 4.00, 4.00, 28]
colors = ['#00B7EB', '#00FF7F']  # 浅蓝色到浅绿色
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
data1 = [0.00, 4.00, 0.00, 8.00, 0.00, 0.00, 0.00, 4.00, 8.00, 8.00, 0.00, 28.00]
data2 = [8.00, 12.00, 8.00, 4.00, 4.00, 28.00, 0.00, 48.00, 8.00, 4.00, 8.00, 24.00]

# 计算平均值和标准误差
means = [np.mean(data1), np.mean(data2)]
std_errors = [np.std(data1) / np.sqrt(len(data1)), np.std(data2) / np.sqrt(len(data2))]

# 确认平均值
print(f'平均值: Data1: {means[0]:.2f}, Data2: {means[1]:.2f}')

fig, ax = plt.subplots(figsize=(2, 4))  # 调整图像大小以符合你的格式

bars = ax.bar(['PerAct2', 'AnyBimanual'], means, yerr=std_errors, capsize=5, color=[cmap(0.2, alpha=0.7), cmap(0.8, alpha=0.7)])
ax.set_ylim(0, 20)

ax.tick_params(axis='both', which='major', labelsize=8)

# 自动调整布局，避免标签被裁剪
plt.tight_layout()

# 保存图像为PNG格式，调整分辨率
plt.savefig('rlbench_20.png', format='png', dpi=300)

# 显示图像
plt.show()