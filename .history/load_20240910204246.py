import torch

def compare_model_shapes(file1, file2):

    weights1 = torch.load(file1)
    weights2 = torch.load(file2)
    
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())


    unique_to_file1 = keys1 - keys2
    if unique_to_file1:
        print("Parameters that are in peract1 but not in peract2:")
        for key in unique_to_file1:
            print(f"  {key}")
    else:
        print("All parameters in peract1 are also in peract2.")
    
    unique_to_file2 = keys2 - keys1
    if unique_to_file2:
        print("Parameters that are in peract2 but not in peract1:")
        for key in unique_to_file2:
            print(f"  {key}")
    else:
        print("All parameters in file1 are also in file2.")

    common_keys = keys1.intersection(keys2)
    for key in common_keys:
        if weights1[key].shape != weights2[key].shape:
            print(f"Shapes for parameter '{key}' are different: {weights1[key].shape} vs {weights2[key].shape}")
        else:
            print(f"Shapes for parameter '{key}' are identical: {weights1[key].shape}")

file1 = '/mnt/disk_1/tengbo/peract_bimanual/ckpts/ANY_BIMANUAL/QAttentionAgent_layer0.pt'
file2 = '/mnt/disk_1/tengbo/peract_bimanual/log/buttons_0903_LF_skill/PERACT_BC/seed0/weights/70000/checkpoint_peract_bc_follower_layer_0.pt'
compare_model_shapes(file1, file2)


# import torch

# # 加载模型的 state_dict
# checkpoint_path = "/mnt/disk_1/tengbo/peract_bimanual/ckpts/PERACT_BC/checkpoint_peract_bc_follower_layer_0.pt"
# state_dict = torch.load(checkpoint_path)

# # 获取要修改的参数
# weight_key = "_qnet.module.proprio_preprocess.linear.weight"
# old_weight = state_dict[weight_key]

# # 检查原来参数的形状是否是 [64, 4]
# if old_weight.shape == torch.Size([64, 4]):
#     # 拼接三次，变成 [64, 12]
#     new_weight = torch.cat([old_weight, old_weight, old_weight], dim=1)
    
#     # 替换原来的权重
#     state_dict[weight_key] = new_weight
    
#     # 保存新的模型
#     new_checkpoint_path = "/mnt/disk_1/tengbo/peract_bimanual/ckpts/PERACT_BC/new_checkpoint_peract_bc_follower_layer_0.pt"
#     torch.save(state_dict, new_checkpoint_path)
    
#     print(f"新模型已保存到: {new_checkpoint_path}")
# else:
#     print(f"权重的形状不是预期的 [64, 4]，而是 {old_weight.shape}")