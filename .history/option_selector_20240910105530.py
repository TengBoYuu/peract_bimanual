import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class OptionSelector(nn.Module):  # 继承自 torch.nn.Module
    def __init__(self, num_classes=18):
        super(OptionSelector, self).__init__()

        # 使用共享的ResNet特征提取器处理每个RGB图像
        resnet = models.resnet50(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后两层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 将输出压缩为固定尺寸
        
        # 语言特征处理部分
        self.language_fc = nn.Linear(512, 256)  # 将输入维度压缩为256
        self.language_pool = nn.AdaptiveAvgPool1d(1)  # 对语言维度进行全局池化

        # 融合后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(10496, 512),  # 5个RGB图像的特征 + 语言特征
            nn.ReLU(),
            nn.Linear(512, num_classes)  # 最终输出18个类别的logits
        )

    def forward(self, rgb_list, lang_input):
        # 处理RGB图像列表中的每一个元素
        rgb_features_list = []
        for rgb_image in rgb_list:
            # 提取每个图像的特征 [b, 3, 256, 256] -> [b, 2048, 8, 8]
            features = self.rgb_extractor(rgb_image)
            # 全局池化 [b, 2048, 8, 8] -> [b, 2048, 1, 1]
            features = self.avgpool(features).view(rgb_image.size(0), -1)  # 展平为 [b, 2048]
            rgb_features_list.append(features)

        # 将5个图像的特征拼接在一起 [b, 2048] * 5 -> [b, 2048 * 5]
        rgb_combined_features = torch.cat(rgb_features_list, dim=1)

        # 处理语言输入 [b, 77, 512] -> [b, 512] (通过池化提取语言全局特征)
        lang_features = self.language_pool(lang_input.permute(0, 2, 1)).squeeze(-1)  # [b, 512]
        lang_features = F.relu(self.language_fc(lang_features))  # 压缩到256维度

        # 融合RGB特征和语言特征 [b, 2048 * 5 + 256]
        combined_features = torch.cat((rgb_combined_features, lang_features), dim=1)

        # 输出类别预测的 logits
        logits = self.fc(combined_features)
        return logits