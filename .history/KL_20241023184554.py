import numpy as np

# 读取txt文件中的numpy数组
def read_voxel_data_from_txt(file_path):
    """从txt文件中读取NumPy数组"""
    # 假设文件中的数据以空格或换行符分隔，可以用 loadtxt 来读取
    return np.loadtxt(file_path)

# 重新定义KL散度计算函数，处理 0-1 之间的数据
def kl_divergence_multi_dim(p, q):
    """多维 KL 散度计算，p 和 q 是高维数据"""
    p = p / np.sum(p)
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

# 读取两个txt文件中的数据
voxel_left = read_voxel_data_from_txt("/mnt/disk_1/tengbo/peract_bimanual/L_voxel_result.txt")
voxel_right = read_voxel_data_from_txt("/mnt/disk_1/tengbo/peract_bimanual/L_voxel_result1.txt")

# 确保数据维度一致
assert voxel_left.shape == voxel_right.shape, "左右体素数据的维度必须一致"

# 计算 L_voxel
L_voxel = voxel_loss(voxel_left, voxel_right)

# 输出 L_voxel 结果
print(f"L_voxel: {L_voxel}")