import numpy as np

# 生成两个形状为 [100, 100, 100, 10] 的随机变量，值在 [0, 1] 之间
voxel_left = np.random.rand(100, 100, 100, 10)
voxel_right = np.random.rand(100, 100, 100, 10)

# 重新定义KL散度计算函数，处理 0-1 之间的数据
def kl_divergence_multi_dim(p, q):
    """多维 KL 散度计算，p 和 q 是高维数据"""
    p = p / np.sum(p)  # 将 p 和 q 转为概率分布
    q = q / np.sum(q)
    
    # 避免除以 0 的问题，使用小值来平滑
    epsilon = 1e-10
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))

# 计算L_voxel的正则化损失
def voxel_loss(voxel_left, voxel_right):
    """计算体素的 JS 散度损失"""
    kl_left_right = kl_divergence_multi_dim(voxel_left, voxel_right)
    kl_right_left = kl_divergence_multi_dim(voxel_right, voxel_left)
    
    # Jensen-Shannon 散度公式
    js_loss = (kl_left_right + kl_right_left) / 2
    return js_loss

# 计算 L_voxel
L_voxel = voxel_loss(voxel_left, voxel_right)

# 输出 L_voxel 结果
print(f"L_voxel: {L_voxel}")