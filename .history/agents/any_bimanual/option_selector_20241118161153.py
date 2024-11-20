import torch
import torch.nn as nn
import transformers

class OptionSelector(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_matrix=None,
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes

        if embedding_matrix is not None:
            self.embeddings_matrix = embedding_matrix.to(self.device)
            # 如果需要使用 embeddings_matrix，请在 forward 方法中使用

    def forward(self, voxel_embedding, language_embedding):

        batch_size = voxel_embedding.shape[0]

        # Step 1: Embed voxel and language embeddings
        voxel_embeddings = self.embed_voxel(voxel_embedding)  # [b, 8000, hidden_size]
        language_embeddings = self.embed_lang(language_embedding)  # [b, 77, hidden_size]

        # Step 2: Concatenate language and voxel embeddings
        inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)  # [b, 8077, hidden_size]

        # Step 3: Apply layer normalization
        stacked_inputs = self.embed_ln(inputs)

        # Step 4: Generate attention mask (all ones) with correct dtype
        attention_mask = torch.ones(
            (batch_size, self.max_lang_tokens + self.max_voxels),
            device=voxel_embedding.device,
            dtype=torch.bool  # Ensure correct dtype
        )
        print("######################################")
        print(f"attention_mask device: {attention_mask.device}")
        print(f"inputs_embeds device: {stacked_inputs.device}")
        print(attention_mask.shape)
        print(f"voxel_embeddings shape: {voxel_embeddings.shape}")
        print(f"language_embeddings shape: {language_embeddings.shape}")
        print(f"stacked_inputs shape: {stacked_inputs.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"attention_mask dtype: {attention_mask.dtype}")
        assert attention_mask.dtype in [torch.float32, torch.float64, torch.int32, torch.int64], "attention_mask dtype must be float or int"

        # 检查是否包含 NaN 或 Inf
        assert torch.isfinite(attention_mask).all(), "attention_mask contains NaN or Inf"

        # 检查值是否全为 1
        assert torch.all((attention_mask == 1)), "attention_mask contains values not equal to 1"
        # Step 5: Transformer forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )

        # Step 6: Get the last hidden state
        hidden_state = transformer_outputs.last_hidden_state  # [b, 8077, hidden_size]

        # Step 7: Aggregate hidden state (e.g., by averaging or using the first token)
        aggregated_hidden = hidden_state.mean(dim=1)  # [b, hidden_size]

        # Step 8: Predict logits
        logits = self.predict_logits(aggregated_hidden)  # [b, output_dim]
        print(logits.shape)

        return logits