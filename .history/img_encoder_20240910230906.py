import pickle
import matplotlib.pyplot as plt
import numpy as np

# 加载pickle文件
with open('/mnt/disk_1/tengbo/peract_bimanual/vocabulary_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# 假设嵌入有18个，每个大小为[77, 512]
embeddings_array = np.array(embeddings)

# 使用热图可视化每个嵌入
fig, axs = plt.subplots(3, 6, figsize=(18, 9))  # 创建3行6列的子图

for i in range(18):
    ax = axs[i // 6, i % 6]
    heatmap = ax.imshow(embeddings_array[i], cmap='viridis', aspect='auto')
    ax.set_title(f'Embedding {i+1}')
    plt.colorbar(heatmap, ax=ax)

plt.tight_layout()
plt.show()