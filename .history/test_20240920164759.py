import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 初始化进程组
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 销毁进程组
def cleanup():
    dist.destroy_process_group()

# 训练模型
def train(rank, world_size):
    setup(rank, world_size)

    # 将模型分布到多个 GPU 上
    model = SimpleModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 使用 DDP 的数据加载器
    batch_size = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    for epoch in range(5):
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(rank), batch_labels.to(rank)

            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {loss.item()}")

    cleanup()

# 主函数
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()