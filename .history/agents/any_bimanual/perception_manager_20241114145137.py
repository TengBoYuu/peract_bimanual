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
        self.conv2_right = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)

        self.activation = nn.ReLU()

    def forward(self, ins):
        
        ins = ins.transpose(1, 2)  # [B, 128, 8077]
        features = self.activation(self.conv1(ins))  # [B, hidden_dim, 8077]
        mask_right = self.activation(self.conv2_right(features))  # [B, mask_dim, 8077]
        mask_left = self.activation(self.conv2_left(features))
        # print(mask_right.shape) # [b,8077,128]
        mask_right = mask_right.transpose(1,2)
        mask_left = mask_left.transpose(1,2)
        ins = ins.transpose(1, 2)  #[B, 8077, 128]
        # print(ins.shape)
        masked_ins1 = ins * mask_left  # [B, 8077, 128]
        masked_ins2 = ins * mask_right  # [B, 8077, 128]

        return masked_ins1, masked_ins2