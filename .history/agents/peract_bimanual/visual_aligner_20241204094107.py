import torch
import torch.nn as nn

class VisualAligner(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, mask_dim=128, num_layers=3, attention_heads=4, dropout=0.1):
        super(VisualAligner, self).__init__()
        
        self.initial_conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        
        # Stack of residual convolutional blocks
        self.residual_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Attention mechanism for mask generation
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        
        # Left and right mask generation
        self.conv2_right = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, ins):
        # Input shape: [b, seq_len, input_dim]
        ins = ins.transpose(1, 2)  # Convert to [b, input_dim, seq_len]
        
        # Initial convolution
        features = self.activation(self.initial_bn(self.initial_conv(ins)))
        
        # Residual convolutional blocks
        for block in self.residual_blocks:
            features = block(features)
        
        # Apply attention (convert to [b, seq_len, hidden_dim])
        features = features.transpose(1, 2)  # [b, seq_len, hidden_dim]
        attn_features, _ = self.multihead_attention(features, features, features)
        features = features + attn_features  # Residual connection
        
        # Convert back to [b, hidden_dim, seq_len]
        features = features.transpose(1, 2)
        
        # Generate masks
        mask_right = self.activation(self.conv2_right(features))
        mask_left = self.activation(self.conv2_left(features))
        
        # Mask application
        mask_right = mask_right.transpose(1, 2)  # [b, seq_len, mask_dim]
        mask_left = mask_left.transpose(1, 2)
        ins = ins.transpose(1, 2)  # Back to [b, seq_len, input_dim]
        masked_ins1 = ins * mask_left
        masked_ins2 = ins * mask_right
        
        return masked_ins1, masked_ins2

class ResidualConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x + residual  # Residual connection