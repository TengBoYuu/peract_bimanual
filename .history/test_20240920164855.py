import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# 模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化进程组
def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 销毁进程组
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # 创建数据集
    input_data = torch.randn(10000, 1000)  # 10000条数据，每条1000维
    labels = torch.randint(0, 10, (10000,))  # 10类分类问题
    dataset = TensorDataset(input_data, labels)

    # 创建模型
    model = SimpleModel().to(rank)

    # 使用 DDP 包装模型
    model = DDP(model, device_ids=[rank])

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 创建分布式数据加载器
    batch_size = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # 训练模型
    for epoch in range(5):  # 假设我们训练 5 个 epoch
        train_sampler.set_epoch(epoch)  # DDP要求每个epoch重置sampler
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(rank), batch_labels.to(rank)

            # 前向传播
            outputs = model(batch_data)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {loss.item()}")

    cleanup()

# 主函数
def main():
    world_size = torch.cuda.device_count()  # 获取可用GPU的数量
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()