import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualAligner(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, mask_dim=64, num_layers=2, attention_heads=2, dropout=0.1):
        super(VisualAligner, self).__init__()
        
        self.initial_conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
        
        self.conv2_right = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
        
        self.activation = nn.ReLU()

    def forward(self, ins):
        # 输入: [b, seq_len, input_dim]
        ins = ins.transpose(1, 2)  # 转换为 [b, input_dim, seq_len]

        features = self.activation(self.initial_bn(self.initial_conv(ins)))
        
        for block in self.residual_blocks:
            features = block(features)

        # 转换为 [b, seq_len, hidden_dim] 给多头注意力使用
        features = features.transpose(1, 2)
        attn_features, _ = self.multihead_attention(features, features, features)
        features = features + attn_features
        features = features.transpose(1, 2)  # 回到 [b, hidden_dim, seq_len]

        mask_right = self.activation(self.conv2_right(features))
        mask_left = self.activation(self.conv2_left(features))
        mask_right = mask_right.transpose(1, 2)  # [b, seq_len, mask_dim]
        mask_left = mask_left.transpose(1, 2)    # [b, seq_len, mask_dim]

        ins = ins.transpose(1, 2)  # 回到 [b, seq_len, input_dim]
        masked_ins1 = ins * mask_right
        masked_ins2 = ins * mask_left

        return masked_ins1, masked_ins2

class ResidualConvBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualConvBlock, self).__init__()
        
        # 深度可分离卷积：分为 depthwise 和 pointwise 两步
        self.depthwise_conv1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.pointwise_conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.depthwise_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.pointwise_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        # 第一次卷积
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.activation(self.bn1(x))
        
        # 第二次卷积
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.activation(self.bn2(x))
        
        x = self.dropout(x)
        return x + residual

# class VisualAligner(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=256, mask_dim=128):
#         super(VisualAligner, self).__init__()

#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
#         self.conv2_right = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
#         self.conv2_left = nn.Conv1d(in_channels=hidden_dim, out_channels=mask_dim, kernel_size=3, padding=1)
#         self.activation = nn.ReLU()

#     def forward(self, ins):
#         ins = ins.transpose(1, 2)
#         features = self.activation(self.conv1(ins)) 
#         mask_right = self.activation(self.conv2_right(features)) 
#         mask_left = self.activation(self.conv2_left(features))
#         mask_right = mask_right.transpose(1,2)
#         mask_left = mask_left.transpose(1,2)
#         ins = ins.transpose(1, 2) 
#         masked_ins1 = ins * mask_left
#         masked_ins2 = ins * mask_right
#         return masked_ins1, masked_ins2