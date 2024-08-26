import torch

# 假设你保存的模型文件名为 'model.pth'
state_dict = torch.load('/mnt/disk_1/tengbo/3d_diffuser_actor/diffuser_actor_peract.pth')

# 检查是否有 'weight' 参数
if 'weight' in state_dict:
    weight = state_dict['weight']
    
    # 将 weight 参数单独保存到一个 .pt 文件
    torch.save(weight, '/mnt/disk_1/tengbo/3d_diffuser_actor/weight_parameters.pt')
    
    print("Weight parameters saved to 'weight_parameters.pt'.")
else:
    print("'weight' parameter not found in the state_dict.")