import torch

# 加载 .pt 文件
weight = torch.load('/mnt/disk_1/tengbo/3d_diffuser_actor/weight_parameters.pt')

# 初始化一个变量用于存储总元素数量
total_elements = 0

# 遍历字典中的每个张量，并计算其元素数量
for key, tensor in weight.items():
    element_count = torch.numel(tensor)  # 计算当前张量的元素数量
    total_elements += element_count
    print(f"{key}: {element_count} elements")

# 输出所有参数的总元素数量
print(f"Total number of elements in 'weight': {total_elements}")