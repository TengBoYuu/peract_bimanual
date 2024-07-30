import torch

def compare_model_shapes(file1, file2):
    # 加载两个模型的权重文件
    weights1 = torch.load(file1)
    weights2 = torch.load(file2)
    
    # 获取两个模型权重的键
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    # 找出 file1 中存在而 file2 中不存在的参数
    unique_to_file1 = keys1 - keys2
    if unique_to_file1:
        print("Parameters that are in file1 but not in file2:")
        for key in unique_to_file1:
            print(f"  {key}")
    else:
        print("All parameters in file1 are also in file2.")
    
    unique_to_file2 = keys2 - keys1
    if unique_to_file2:
        print("Parameters that are in file2 but not in file1:")
        for key in unique_to_file2:
            print(f"  {key}")
    else:
        print("All parameters in file1 are also in file2.")

    # 比较共同键的值
    common_keys = keys1.intersection(keys2)
    for key in common_keys:
        if weights1[key].shape != weights2[key].shape:
            print(f"Shapes for parameter '{key}' are different: {weights1[key].shape} vs {weights2[key].shape}")
        else:
            print(f"Shapes for parameter '{key}' are identical: {weights1[key].shape}")

# 使用示例
file1 = '/mnt/disk_1/tengbo/peract_bimanual/ckpts/multi/PERACT_BC/seed0/weights/600000/QAttentionAgent_layer0.pt'
file2 = '/mnt/disk_1/tengbo/peract_bimanual/log/multi_noe_per2_bs2_0723/BIMANUAL_PERACT/seed0/weights/100000/QAttentionAgent_layer0.pt'
compare_model_shapes(file1, file2)