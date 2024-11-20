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
import numpy as np
import torch
import torch.nn as nn
import transformers
from trajectory_gpt2 import GPT2Model


class OptionSelector(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_matrix,
            voxel_dim=128,
            lang_dim=128,
            hidden_size=256,
            output_dim=18,
            max_voxels=8000,
            max_lang_tokens=77,
            **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # GPT-2 configuration
        config = transformers.GPT2Config(
            vocab_size=1,  # not used
            n_embd=hidden_size,
            n_head=8, 
        )

        self.max_voxels = max_voxels
        self.max_lang_tokens = max_lang_tokens

        # Input embedding layers
        self.embed_voxel = nn.Linear(voxel_dim, hidden_size)
        self.embed_lang = nn.Linear(lang_dim, hidden_size)

        # Transformer
        self.transformer = transformers.GPT2Model(config)

        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Output layer
        self.predict_logits = nn.Linear(hidden_size, output_dim)
        # self.fc1 = nn.Linear(8077*128,num_classes)
        # self.fc1.apply(init_weights_xavier)
        # self.fc2 = nn.Linear(num_classes,77*512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        self.embeddings_matrix = embedding_matrix
        self.embeddings_matrix = self.embeddings_matrix.to(self.device)
        # print(embedding_matrix.shape)
        # self.init_fc2_weights()
    
    def forward(self, voxel_embedding, language_embedding):

        batch_size = voxel_embedding.shape[0]

        # Step 1: Embed voxel and language embeddings
        voxel_embeddings = self.embed_voxel(voxel_embedding)  # [b, 8000, hidden_size]
        language_embeddings = self.embed_lang(language_embedding)  # [b, 77, hidden_size]

        # Step 2: Concatenate language and voxel embeddings
        inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)  # [b, 8077, hidden_size]

        # Step 3: Apply layer normalization
        stacked_inputs = self.embed_ln(inputs)

        # Step 4: Generate attention mask (all ones)
        attention_mask = torch.ones((batch_size, self.max_lang_tokens + self.max_voxels), device=voxel_embedding.device)
assert attention_mask.shape[0] == inputs_embeds.shape[0], "Batch size mismatch"
assert attention_mask.shape[1] == inputs_embeds.shape[1], "Sequence length mismatch"

        # Step 5: Transformer forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )

        # Step 6: Get the last hidden state
        hidden_state = transformer_outputs['last_hidden_state']  # [b, 8077, hidden_size]

        # Step 7: Aggregate hidden state (e.g., by averaging or using the first token)
        aggregated_hidden = hidden_state.mean(dim=1)  # [b, hidden_size]

        # Step 8: Predict logits
        logits = self.predict_logits(aggregated_hidden)  # [b, output_dim]
        print(logits.shape)

        return logits