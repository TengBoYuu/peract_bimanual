import numpy as np

# 读取txt文件中的多维NumPy数组
def read_voxel_data_from_txt(file_path):
    """从txt文件中读取NumPy数组，使用 eval() 将字符串转换为多维数组"""
    with open(file_path, 'r') as file:
        data_str = file.read()  # 读取整个文件内容
        data = eval(data_str)   # 使用 eval 将其转换为 Python 对象
        return np.array(data)   # 转换为 NumPy 数组

# 重新定义KL散度计算函数，处理 0-1 之间的数据
def kl_divergence_multi_dim(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    epsilon = 1e-10
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))

# 计算L_voxel的正则化损失
def voxel_loss(voxel_left, voxel_right):
    kl_left_right = kl_divergence_multi_dim(voxel_left, voxel_right)
    kl_right_left = kl_divergence_multi_dim(voxel_right, voxel_left)
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