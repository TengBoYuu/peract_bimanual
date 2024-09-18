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

class OptionSelector(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(OptionSelector, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=172, stride=31, padding=0)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(128*256, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_classes)
        # )
        self.fc1 = nn.Linear(8077*128,num_classes)
        # self.fc1.apply(init_weights_xavier)
        self.fc2 = nn.Linear(num_classes,77*512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        self.init_fc2_weights()

    # def init_fc2_weights(self):
    #     self.fc2.weight = torch.nn.Parameter(self.embeddings_matrix.t().clone())
    #     self.fc2.bias = torch.nn.Parameter(torch.zeros_like(self.fc2.bias)) 
    def init_fc2_weights(self):
        self.fc2.weight.copy_(self.embeddings_matrix.t()).requires_grad_(False)
        self.fc2.bias.zero_().requires_grad_(False)
    
    def forward(self, ins, lang):

        skill_set = ['turn_tap','open_drawer','push_buttons','sweep_to_dustpan_of_size','slide_block_to_color_target','insert_onto_square_peg','meat_off_grill','place_shape_in_shape_sorter','place_wine_at_rack_location','put_groceries_in_cupboard','put_money_in_safe','close_jar','reach_and_drag','light_bulb_in','stack_cups','place_cups','put_item_in_drawer','stack_blocks']
        ins_ = torch.cat((lang, ins), dim=1) # [B,8077,128]
        # ins_ = ins_.permute(0, 2, 1)
        # ins_reduced = self.conv1(ins_) 
        # ins_flat = ins_reduced.view(ins_reduced.size(0), -1)
        ins_flat = ins_.view(ins_.size(0),-1)
        logits = self.fc1(ins_flat)
        probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        predicted_class = torch.argmax(probs, dim=1)
        print(predicted_class)
        # print(self.embeddings_matrix.shape) # [18,77*512]
        emb = self.embeddings_matrix[int(predicted_class)].to(self.device)
        emb_reshaped = emb.view(1, 77, 512)

        one_hot_predicted_class = torch.nn.functional.one_hot(predicted_class, num_classes=self.num_class).float()

        y = (one_hot_predicted_class - probs).detach() + probs 
        selected_embedding_flat = self.fc2(y)  # [1, 77*512]

        selected_embedding = selected_embedding_flat.view(-1, 77, 512)
        concatenated = torch.cat((selected_embedding, emb_reshaped), dim=1)
        # print(selected_embedding.shape) # [B,77,512]
        return selected_embedding