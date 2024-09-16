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

class OptionSelector(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(OptionSelector, self).__init__()

        self.input_preprocess = Conv3DBlock(
            10,
            64,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation="lrelu",
        )
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels,
            64,
            5,
            5,
            norm=None,
            activation="lrelu",
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*2048+256+4, 512), 
            nn.ReLU(),
            nn.Linear(512, num_classes)  
        )
        self.fc2 = nn.Linear(num_classes,77*512)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        print(self.embeddings_matrix.shape)
        self.init_fc2_weights()

    def init_fc2_weights(self):
        with torch.no_grad(): 
            self.fc2.weight.copy_(self.embeddings_matrix.t()).requires_grad_(False)
            self.fc2.bias.zero_().requires_grad_(False)

    def create_embedding_matrix(self):
        flattened_embeddings = []
        for key in self.embeddings_dict.keys():
            embedding = torch.tensor(self.embeddings_dict[key]) 
            flattened_embedding = embedding.view(-1) 
            flattened_embeddings.append(flattened_embedding)
        embeddings_matrix = torch.stack(flattened_embeddings)  
        return embeddings_matrix

    def forward(self, voxel_grid, lang_input, proprio):



        lang_features = self.language_pool(lang_input.permute(0, 2, 1)).squeeze(-1)  # [b, 512]
        lang_features = F.relu(self.language_fc(lang_features)) 

        # print("RGB combined features shape:", rgb_combined_features.shape)
        # print("proprio combined features shape:", proprio.shape)
        combined_features = torch.cat((rgb_combined_features, lang_features, proprio), dim=1)


        
        # print("Combined features shape:", combined_features.shape)
        logits = self.fc1(combined_features)
        probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        predicted_class = torch.argmax(probs, dim=1)
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