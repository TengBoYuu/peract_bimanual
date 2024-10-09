import matplotlib.pyplot as plt
import numpy as np

# 定义数据
data1 = [8.00, 16.00, 4.00, 40.00, 4.00, 0.00, 0.00, 12.00, 0.00, 0.00, 0.00, 24.00]
data2 = [8.00, 32.00, 24.00, 20.00, 12.00, 16.00, 8.00, 84.00, 8.00, 4.00, 4.00]

# 创建图像和轴
fig, ax = plt.subplots()

# 绘制带有数据的box plot
boxplot = ax.boxplot([data1, data2], patch_artist=True, notch=False, vert=True)

# 计算平均值和标准误差
means = [np.mean(data1), np.mean(data2)]
std_errors = [np.std(data1)/np.sqrt(len(data1)), np.std(data2)/np.sqrt(len(data2))]

# 绘制误差线，表示平均值和标准误差
ax.errorbar([1, 2], means, yerr=std_errors, fmt='o', color='red', capsize=5)

# 设置标签
ax.set_xticks([1, 2])
ax.set_xticklabels(['Data1', 'Data2'])
ax.set_ylabel('Values')
ax.set_title('Boxplot with Error Bars')

# 保存为EPS格式
plt.savefig('boxplot_with_error_bars.eps', format='eps')

# 显示图像
plt.show()