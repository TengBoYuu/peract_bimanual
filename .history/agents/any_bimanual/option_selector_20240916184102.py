import torch
import torch.nn as nn
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
        self.conv1 = nn.Conv1d(in_channels=8077, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Sequential(
            nn.Linear(128*256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.fc2 = nn.Linear(num_classes,77*512)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        self.init_fc2_weights()

    def init_fc2_weights(self):
        with torch.no_grad(): 
            self.fc2.weight.copy_(self.embeddings_matrix.t()).requires_grad_(False)
            self.fc2.bias.zero_().requires_grad_(False)

    def forward(self, ins, lang):

        skill_set = ['turn_tap','open_drawer','push_buttons','sweep_to_dustpan_of_size','slide_block_to_color_target','insert_onto_square_peg','meat_off_grill','place_shape_in_shape_sorter','place_wine_at_rack_location','put_groceries_in_cupboard','put_money_in_safe','close_jar','reach_and_drag','light_bulb_in','stack_cups','place_cups','put_item_in_drawer','stack_blocks']
        ins_ = torch.cat((lang, ins), dim=1) # [B, 8077, 128]
        ins_ = ins_.permute(0, 2, 1)          # Change shape to [B, 128, 8077] for Conv1d
        print(ins_.shape)
        ins_reduced = self.conv1(ins_)         # Apply Conv1d, output shape [B, 1024, 128]
        ins_flat = ins_reduced.view(ins_reduced.size(0), -1)  # Flatten the result [B, 1024*128]
        # print("Combined features shape:", combined_features.shape)
        logits = self.fc1(ins_flat)
        probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        predicted_class = torch.argmax(probs, dim=1)
        print(predicted_class)
        # print(probs)
        # print(predicted_class)
        one_hot_predicted_class = torch.nn.functional.one_hot(predicted_class, num_classes=self.num_class).float()
        # print(one_hot_predicted_class.shape)
        # print(predicted_class) 

        y = (one_hot_predicted_class - probs).detach() + probs 
        # print("y: ",y)
        selected_embedding_flat = self.fc2(y)  # [1, 77*512]

        selected_embedding = selected_embedding_flat.view(-1, 77, 512)
        # print(selected_embedding.shape)

        return selected_embedding

        # batch_vocabulary_sentences = []
        # for i in range(predicted_class.size(0)): 
        #     class_idx = predicted_class[i].item() 
        #     vocab_key = list(self.vocabulary.keys())[class_idx]
        #     mean_embedding = torch.tensor(self.embeddings_dict[vocab_key],requires_grad=True).to(self.device)
        #     # mean_embedding = mean_embedding.expand(1, -1)  # [1, 512]
        #     batch_vocabulary_sentences.append(mean_embedding)

        # final_embeddings = torch.stack(batch_vocabulary_sentences, dim=0)  # [batch_size, 1, 512]
        # return final_embeddings