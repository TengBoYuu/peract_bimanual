import numpy as np
import torch
import torch.nn as nn
import transformers

from trajectory_gpt2 import GPT2Model
from img_encoder import Encoder


class OptionTransformer(nn.Module):

    """
    This model uses GPT-2 to select options for every horizon-th state
    """

    def __init__(
            self,
            state_dim,
            lang_dim,
            option_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs):
        super().__init__()

        self.option_dim = option_dim
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.state_dim = state_dim
        self.max_length = max_length
        self.output_attentions = kwargs["output_attentions"]
        self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)

        self.embed_lang = nn.Linear(lang_dim, hidden_size)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_options = torch.nn.Linear(hidden_size, self.option_dim)

    def forward(self, word_embeddings, states, attention_mask, **kwargs):
        # Ensure all tensors are on the same device as states
        device = word_embeddings.device
        word_embeddings = word_embeddings.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones((states.shape[0], states.shape[1]), dtype=torch.long, device=device)

        batch_size, seq_length = states.shape[0], states.shape[1]
        num_tokens = word_embeddings.shape[1]

        # Embed states and language
        state_emb = [self.embed_state(rgb) for rgb in states]
        print(len(state_emb))
        # print(state_emb[0].shape)
        state_embeddings = torch.mean(torch.stack)
        ret_state_embeddings = state_embeddings.clone().detach()
        lang_embeddings = self.embed_lang(word_embeddings).to(device)

        # Combine language and state embeddings
        lang_and_inputs = torch.cat([lang_embeddings, state_embeddings], dim=1).to(device)

        # Apply layer normalization
        stacked_inputs = self.embed_ln(lang_and_inputs).to(device)

        # Prepare the attention mask
        lang_attn_mask = torch.cat([torch.ones((batch_size, num_tokens), device=device), attention_mask], dim=1).to(device)

        # Pass through the transformer model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=lang_attn_mask,
        )

        # Split the output into language and trajectory outputs
        x = transformer_outputs['last_hidden_state']
        lang_out = x[:, :num_tokens, :].reshape(batch_size, num_tokens, self.hidden_size)
        traj_out = x[:, num_tokens:, :].reshape(batch_size, seq_length, self.hidden_size)

        # Get predictions
        option_preds = self.predict_options(traj_out)

        if self.output_attentions:
            attentions = transformer_outputs[-1]
            return option_preds, attentions, ret_state_embeddings

        return option_preds, None, ret_state_embeddings