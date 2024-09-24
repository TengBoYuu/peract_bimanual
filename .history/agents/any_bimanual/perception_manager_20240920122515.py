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

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier initialization
        if m.bias is not None:
            init.zeros_(m.bias)  # Set bias to zero

class Perception(nn.Module):
    def __init__(self):
        super(Perception, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=172, stride=31, padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(128*256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        # self.fc1 = nn.Linear(8077*128,num_classes)
        # self.fc1.apply(init_weights_xavier)
        # self.fc2 = nn.Linear(num_classes,77*512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        self.embeddings_matrix = self.embeddings_matrix.to(self.device)
        # print(embedding_matrix.shape)
        # self.init_fc2_weights()

    # def init_fc2_weights(self):
    #     self.fc2.weight = torch.nn.Parameter(self.embeddings_matrix.t().clone())
    #     self.fc2.bias = torch.nn.Parameter(torch.zeros_like(self.fc2.bias)) 
    # def init_fc2_weights(self):
    #     with torch.no_grad():
    #         self.fc2.weight.copy_(self.embeddings_matrix.t()).requires_grad_(False)
    #         self.fc2.bias.zero_().requires_grad_(False)
    
    def forward(self, voxel, lang, percep):



        return skill_right, skill_left