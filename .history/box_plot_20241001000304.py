import matplotlib.pyplot as plt
import numpy as np

# 定义数据
data1 = [8.00, 16.00, 4.00, 40.00, 4.00, 0.00, 0.00, 12.00, 0.00, 0.00, 0.00, 24.00]
data2 = [8.00, 32.00, 24.00, 20.00, 12.00, 16.00, 8.00, 84.00, 8.00, 4.00, 4.00, 28]

# 计算平均值和标准误差
means = [np.mean(data1), np.mean(data2)]
std_errors = [np.std(data1) / np.sqrt(len(data1)), np.std(data2) / np.sqrt(len(data2))]

# 确认平均值
print(f'平均值: Data1: {means[0]:.2f}, Data2: {means[1]:.2f}')

# 创建图像
fig, ax = plt.subplots(figsize=(2, 4))  # 调整图像大小以符合你的格式

# 绘制带有误差条的柱状图
bars = ax.bar(['PerAct2', 'AnyBimanual'], means, yerr=std_errors, capsize=5, color=['blue', 'orange'])
ax.set_ylim(0, 30)

# 设置y轴标签和标题
ax.set_ylabel('Success Rates', fontsize=10)
ax.set_title('RLBench2', fontsize=12)

# 在柱子上方标注平均值，移动到error bar之上
for bar, mean, err in zip(bars, means, std_errors):
    height = bar.get_height()
    ax.annotate(f'{mean:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height + err ),  # 位置在error bar上方
                xytext=(0, 3),  # 位置偏移
                textcoords="offset points", ha='center', va='bottom', fontsize=10)
# ax.set_xticklabels(['PerAct2', 'AnyBimanual'], fontsize=14)
plt.xticks([0, 1], ['PerAct2', 'AnyBimanual'], fontsize=1)
# 调整刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=9)

# 自动调整布局，避免标签被裁剪
plt.tight_layout()

# 保存图像为PNG格式
plt.savefig('rlbench2.png', format='png')

# 显示图像
plt.show()