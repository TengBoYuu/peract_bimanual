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

file1 = '/mnt/disk_1/tengbo/peract_bimanual/log/buttons_0903_LF_skill/PERACT_BC/seed0/weights/40000/checkpoint_peract_bc_leader_layer_0.pt'
file2 = '/mnt/disk_1/tengbo/peract_bimanual/ckpts/PERACT_BC/checkpoint_peract_bc_leader_layer_0.pt'
compare_model_shapes(file1, file2)