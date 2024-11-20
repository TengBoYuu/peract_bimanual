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
import numpy as np
# def init_weights_xavier(m):
#     if isinstance(m, nn.Linear):
#         init.xavier_uniform_(m.weight)  # Xavier initialization
#         if m.bias is not None:
#             init.zeros_(m.bias)  # Set bias to zero

class OptionSelector(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(OptionSelector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=172, stride=31, padding=0)
        self.fc1_right = nn.Sequential(
            nn.Linear(128*256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.fc1_left = nn.Sequential(
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
    
    def forward(self, ins_right, ins_left, lang):

        skill_set = ['turn_tap','open_drawer','push_buttons','sweep_to_dustpan_of_size','slide_block_to_color_target','insert_onto_square_peg','meat_off_grill','place_shape_in_shape_sorter','place_wine_at_rack_location','put_groceries_in_cupboard','put_money_in_safe','close_jar','reach_and_drag','light_bulb_in','stack_cups','place_cups','put_item_in_drawer','stack_blocks']
        ins_right = torch.cat((lang, ins_right), dim=1) # [B,8077,128]
        ins_left = torch.cat((lang, ins_left), dim=1) # [B,8077,128]
        ins_right = ins_right.transpose(1,2)
        ins_left = ins_left.transpose(1,2)
        ins_right = self.conv1(ins_right)
        ins_left = self.conv1(ins_left)
        ins_right = ins_right.view(ins_right.size(0),-1)
        ins_left = ins_left.view(ins_left.size(0),-1)
        # logits_right = self.fc1(ins_right)
        # logits_left = self.fc1(ins_left)
        logits_right = self.fc1_right(ins_right)
        logits_left = self.fc1_left(ins_left)
        probs_right = F.softmax(logits_right, dim=1)
        probs_left = F.softmax(logits_left, dim=1)

        # with open("/mnt/disk_1/tengbo/peract_bimanual/skill_embedding.csv", "a") as f:
        #     np.savetxt(f, probs_right.cpu().numpy(), delimiter=",")
        # print(probs_right)
        # print("right: ",torch.argmax(probs_right))
        # print("left: ",torch.argmax(probs_left))

        skill_right = torch.matmul(probs_right, self.embeddings_matrix.to(probs_right.device))
        skill_left = torch.matmul(probs_left, self.embeddings_matrix.to(probs_left.device))
        skill_right = skill_right.view(-1,77,512)
        skill_left  = skill_left.view(-1,77,512)

        # print("right: ",probs_right)
        # print("left: ", probs_left)
        # skill_right = skill_right + lang
        # skill_left = skill_left + lang
        # predicted_class = torch.argmax(probs, dim=1)
        # print(predicted_class)
        # emb = self.embeddings_matrix[int(predicted_class)].to(self.device)
        # emb_reshaped = emb.view(1, 77, 512)
        # one_hot_predicted_class = torch.nn.functional.one_hot(predicted_class, num_classes=self.num_class).float()
        # y = (one_hot_predicted_class - probs).detach() + probs 
        # selected_embedding_flat = self.fc2(y)  # [1, 77*512]
        # selected_embedding = selected_embedding_flat.view(-1, 77, 512)

        return skill_right, skill_left