import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import clip
import pickle
from helpers.network_utils import (
    DenseBlock,
    SpatialSoftmax3D,
    Conv3DBlock,
    Conv3DUpsampleBlock,
)
from einops import rearrange
from einops import repeat


import torch
import torch.nn as nn
import torch.nn.functional as F

class Perception(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, mask_dim=128):
        super(Perception, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)

        self.fc_mask1 = nn.Linear(mask_dim, 8077) 
        self.fc_mask2 = nn.Linear(mask_dim, 8077) 

        # 定义一个简单的非线性激活函数
        self.activation = nn.ReLU()

    def forward(self, ins):
        # 输入 ins 的形状为 [B, 8077, 128]
        
        # 首先使用卷积提取特征，将其形状从 [B, 8077, 128] 变为 [B, 128, 8077]
        ins = ins.transpose(1, 2)  # 调整形状以符合 Conv1d 输入要求 [B, 128, 8077]
        features = self.activation(self.conv1(ins))  # [B, hidden_dim, 8077]
        features = self.activation(self.conv2(features))  # [B, mask_dim, 8077]

        # 生成两个掩码，注意掩码应该和输入形状匹配
        mask1 = torch.sigmoid(self.fc_mask1(features))  # [B, 8077, 8077]
        mask2 = torch.sigmoid(self.fc_mask2(features))  # [B, 8077, 8077]

        # 应用掩码到输入
        ins = ins.transpose(1, 2)  # 将形状恢复为 [B, 8077, 128]
        masked_ins1 = ins * mask1.unsqueeze(-1)  # [B, 8077, 128]，应用掩码1
        masked_ins2 = ins * mask2.unsqueeze(-1)  # [B, 8077, 128]，应用掩码2

        return masked_ins1, masked_ins2