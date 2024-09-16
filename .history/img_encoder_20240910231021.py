import pickle
import matplotlib.pyplot as plt
import numpy as np

# 加载pickle文件
with open('/mnt/disk_1/tengbo/peract_bimanual/vocabulary_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# 检查嵌入的结构
print(f"嵌入的类型: {type(embeddings)}")
print(f"嵌入的长度: {len(embeddings)}")

# 转换为NumPy数组
embeddings_array = np.array(embeddings)

# 打印每个嵌入的形状以检查其维度
for i, embedding in enumerate(embeddings):
    print(f"Embedding {i+1} shape: {np.array(embedding).shape}")

# 使用热图可视化每个嵌入
fig, axs = plt.subplots(3, 6, figsize=(18, 9))  # 创建3行6列的子图

for i in range(len(embeddings)):
    embedding = np.array(embeddings[i])
    if embedding.ndim == 2:  # 确保嵌入是二维的
        ax = axs[i // 6, i % 6]
        heatmap = ax.imshow(embedding, cmap='viridis', aspect='auto')
        ax.set_title(f'Embedding {i+1}')
        plt.colorbar(heatmap, ax=ax)
    else:
        print(f"Embedding {i+1} 不是二维数组，跳过可视化。")

plt.tight_layout()

# 保存为图片
plt.savefig('vocabulary_embeddings_visualization.png')
plt.show()