# import torch
# import torch.nn as nn
# import transformers
# from trajectory_gpt2 import GPT2Model
# import torch.nn.functional as F
# class OptionSelector(nn.Module):
#     def __init__(
#             self,
#             num_classes,
#             embedding_matrix=None,
#             voxel_dim=128,
#             lang_dim=128,
#             hidden_size=256,
#             output_dim=18,
#             max_voxels=8000,
#             max_lang_tokens=77,
#             **kwargs):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.output_dim = output_dim

#         # GPT-2 configuration
#         config = transformers.GPT2Config(
#             vocab_size=1,  # not used
#             n_embd=hidden_size,
#             n_head=8, 
#             n_ctx=1077,
#         )

#         self.max_voxels = max_voxels
#         self.max_lang_tokens = max_lang_tokens

#         self.embed_voxel = nn.Linear(voxel_dim, hidden_size)
#         self.embed_lang = nn.Linear(lang_dim, hidden_size)

#         self.transformer = GPT2Model(config)

#         self.embed_ln = nn.LayerNorm(hidden_size)

#         self.predict_logits = nn.Linear(hidden_size, output_dim)

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.num_class = num_classes

#         if embedding_matrix is not None:
#             self.embeddings_matrix = embedding_matrix.to(self.device)

#     def forward(self, voxel_embedding, language_embedding):

#         batch_size = voxel_embedding.shape[0]

#         voxel_embeddings = self.embed_voxel(voxel_embedding)  # [b, 8000, hidden_size]
#         language_embeddings = self.embed_lang(language_embedding)  # [b, 77, hidden_size]

#         voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, hidden_size, 8000]

#         voxel_embeddings = F.avg_pool1d(voxel_embeddings, kernel_size=8, stride=8)  # [b, hidden_size, 1000]

#         voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, 1000, hidden_size]

#         inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)  # [b, 8077, hidden_size]

#         stacked_inputs = self.embed_ln(inputs)

#         attention_mask = torch.ones(
#             (batch_size, self.max_lang_tokens + self.max_voxels),
#             device=voxel_embedding.device,
#             dtype=torch.long  # Ensure correct dtype
#         )

#         assert torch.isfinite(attention_mask).all(), "attention_mask contains NaN or Inf"

#         assert torch.all((attention_mask == 1)), "attention_mask contains values not equal to 1"
#         transformer_outputs = self.transformer(
#             inputs_embeds=stacked_inputs,
#             attention_mask=None,
#         )

#         hidden_state = transformer_outputs.last_hidden_state  # [b, 8077, hidden_size]
#         aggregated_hidden = hidden_state.mean(dim=1)  # [b, hidden_size]
#         logits = self.predict_logits(aggregated_hidden)  # [b, output_dim]
#         probs = F.softmax(logits, dim=1)
#         skill = torch.matmul(probs, self.embeddings_matrix.to(probs.device))
#         skill = skill.view(-1,77,512)
#         return skill


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

class OptionSelector(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(OptionSelector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=172, stride=31, padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(128*256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        self.embeddings_matrix = self.embeddings_matrix.to(self.device)
    
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
        logits_right = self.fc1(ins_right)
        logits_left = self.fc1(ins_left)
        probs_right = F.softmax(logits_right, dim=1)
        probs_left = F.softmax(logits_left, dim=1)
        skill_right = torch.matmul(probs_right, self.embeddings_matrix.to(probs_right.device))
        skill_left = torch.matmul(probs_left, self.embeddings_matrix.to(probs_left.device))
        skill_right = skill_right.view(-1,77,512)
        skill_left  = skill_left.view(-1,77,512)

        return skill_right, skill_left