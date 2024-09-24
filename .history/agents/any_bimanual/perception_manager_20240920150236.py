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


class Perception(nn.Module):
    def __init__(self):
        super(Perception, self).__init__()


    def forward(self, voxel, lang, percep):

        

        return 