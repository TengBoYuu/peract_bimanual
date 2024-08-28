import torch
import torch.nn as nn
from helpers.clip.core.clip import build_model, load_clip

class OptionTransformer(nn.Module):

    """
    This model uses CLIP to select options for every horizon-th state
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

        self.state_dim = state_dim
        self.max_length = max_length
        self.output_attentions = kwargs.get("output_attentions", False)

        # Load CLIP model and preprocessing
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        self.embed_state = nn.Linear(state_dim, hidden_size)

        # Use the text and vision encoders from CLIP
        self.lang_encoder = self.clip_model.encode_text
        self.image_encoder = self.clip_model.encode_image

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_options = torch.nn.Linear(hidden_size, self.option_dim)

    def forward(self, word_embeddings, states, attention_mask=None, **kwargs):
        device = states.device
        word_embeddings = word_embeddings.to(device)

        # Embed states using the CLIP image encoder
        state_embeddings = self.image_encoder(states).to(device)
        ret_state_embeddings = state_embeddings.clone().detach()

        # Embed language using the CLIP text encoder
        lang_embeddings = self.lang_encoder(word_embeddings).to(device)

        # Combine language and state embeddings
        lang_and_inputs = torch.cat([lang_embeddings, state_embeddings], dim=1).to(device)

        # Apply layer normalization
        stacked_inputs = self.embed_ln(lang_and_inputs).to(device)

        # Since we don't have a transformer model, you may use a different sequence model or simply a feedforward network here
        x = stacked_inputs  # replace this with your sequence model if needed

        # Get predictions
        option_preds = self.predict_options(x)

        if self.output_attentions:
            attentions = None  # CLIP doesn't provide attentions in the same way as transformers
            return option_preds, attentions, ret_state_embeddings

        return option_preds, None, ret_state_embeddings