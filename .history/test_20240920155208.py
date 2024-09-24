import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你的模型是一个简单的多层感知机
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

# 创建数据集
input_data = torch.randn(10000, 1000)  # 10000条数据，每条1000维
labels = torch.randint(0, 10, (10000,))  # 10类分类问题
dataset = TensorDataset(input_data, labels)

# 设置 batch_size，使得每个 batch 的显存占用适合控制在 10GB 左右
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = SimpleModel()

# 使用 DataParallel 将模型分布到多个 GPU 上
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)

# 将模型移动到GPU
model = model.to(device)

# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 控制每张卡的显存使用
max_memory_usage_per_gpu = 10 * 1024**3  # 10GB
for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(max_memory_usage_per_gpu / torch.cuda.get_device_properties(i).total_memory, i)

# 训练模型
while True:  # 假设我们训练 5 个 epoch
    for batch_data, batch_labels in dataloader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        # 前向传播
        outputs = model(batch_data)

        # 确保标签在与输出相同的设备上
        batch_labels = batch_labels.to(outputs.device)

        # 计算损失
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 检查当前 GPU 的显存使用情况
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} Memory usage: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

print("Training finished.")