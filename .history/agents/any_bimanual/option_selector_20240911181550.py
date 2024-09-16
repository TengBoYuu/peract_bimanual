import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import clip
import pickle
vocabulary = {
    'turn_tap': [
        'turn tap' ,
        'rotate the tap',
        'grasp the tap and turn it'
    ],
    'open_drawer': [
        'open drawer',
        'grip the handle and pull the drawer open',
        'slide the drawer open'
    ],
    'push_buttons': [
        'push the button',
        'press the  button',
        'push down the button with the base'
    ],
    'sweep_to_dustpan_of_size': [
        'sweep dirt to the  dustpan',
        'sweep the dirt up into the  dustpan', 
        'use the broom to brush the dirt into the dustpan', 
        'clean up the dirt with the  pan'
    ],
    'slide_block_to_color_target': [
        'slide the block to target',
        'slide the block onto the square', 
        'push the block until it is sitting on top of the  target', 
        'slide the block towards the  plane', 
        'cover the target with the block by pushing the block in its direction'
    ],
    'insert_onto_square_peg': [
        'put the ring on the spoke', 
        'slide the ring onto the colored spoke', 
        'place the ring onto the spoke'
    ],
    'meat_off_grill': [
        'take the off the grill',
        'pick up the and place it next to the grill', 
        'remove the  from the grill and set it down to the side'
    ],
    'place_shape_in_shape_sorter': [
        'put the  in the shape sorter', 
        'pick up the and put it in the sorter', 
        'place the into its slot in the shape sorter', 
        'slot the into the shape sorter'
    ],
    'place_wine_at_rack_location': [
        'stack the wine bottle to the of the rack', 
        'slide the bottle onto the part of the rack', 
        'put the wine on the ', 
        'leave the wine on the section of the shelf', 
        'grasp the bottle and put it away on the'
    ],
    'put_groceries_in_cupboard': [
        'put the in the cupboard', 
        'pick up the and place it in the cupboard', 
        'move the to the shelf', 
        'put away the  in the cupboard'
    ],
    'put_money_in_safe': [
        'put the money away in the safe on the shelf', 
        'leave the money on the shelf on the safe', 
        'place the stack of bank notes on the shelf of the safe'
    ],
    'close_jar': [
        'close the  jar', 
        'screw on the jar lid', 
        'grasping the lid, lift it from the table and use it to seal the jar', 
        'pick up the lid from the table and put it on the jar'
    ],
    'reach_and_drag': [
        'use the stick to drag the cube onto the target', 
        'pick up the stick and use it to push or pull the cube onto the  target', 
        'drag the block towards the square on the table top', 
        'grasping the stick by one end, pick it up and use the its other end to move the block onto the target'
    ],
    'light_bulb_in': [
        'screw in the light bulb', 
        'screw the light bulb from the holder into the lamp', 
        'pick up the light bulb from the stand, lift it up to just above the lamp, then screw it down into the lamp in a clockwise fashion', 
        'put the light bulb from the casing into the lamp'
    ],
    'stack_cups': [
        'stack the other cups on top of the cup', 
        'place two of the cups onto the odd cup out', 
        'put the remaining two cups on top of the cup', 
        'pick up and set the cups down into the  cup', 
        'create a stack of cups with the cup as its base', 
        'keeping the cup on the table, stack the other two onto it'
    ],
    'place_cups': [
        'place cup on the cup holder', 
        'pick up  cup and put it on the mug tree', 
        'move  mug from the table to the cup holder', 
        'pick up  cup and slide its handle onto a spoke on the mug holder'
    ],
    'put_item_in_drawer': [
        'put the item in the drawer', 
        'put the block away in the drawer', 
        'open the drawer and place the block inside of it', 
        'leave the block in the drawer'
    ],
    'stack_blocks': [
        'stack blocks', 
        'place of thecubes on top of each other', 
        'pick up and set down blocks on top of each other', 
        'build a tall tower out of cubes', 
        'arrange blocks in a vertical stack on the table top', 
        'set cubes on top of each other'
    ]
}
class OptionSelector(nn.Module):
    def __init__(self, num_classes=18, vocab=vocabulary):
        super(OptionSelector, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(resnet.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.language_fc = nn.Linear(512, 256)  
        self.language_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc1 = nn.Sequential(
            nn.Linear(4*2048+256+4, 512), 
            nn.ReLU(),
            nn.Linear(512, num_classes)  
        )
        self.fc2 = nn.Linear(num_classes,77*512)

        self.vocabulary = vocab

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes
        with open("/mnt/disk_1/tengbo/peract_bimanual/lang_token.pkl", "rb") as f:
            self.embeddings_dict = pickle.load(f)

        self.embeddings_matrix = self.create_embedding_matrix()

    def create_embedding_matrix(self):
        # 提取所有的嵌入，展平并创建一个 [18, 77*512] 的矩阵
        flattened_embeddings = []
        for key in self.embeddings_dict.keys():
            embedding = torch.tensor(self.embeddings_dict[key])  # 形状为 [77, 512]
            flattened_embedding = embedding.view(-1)  # 将其展平为 [77*512]
            flattened_embeddings.append(flattened_embedding)
        
        # 将列表中的展平嵌入拼接为 [18, 77*512] 的矩阵
        embeddings_matrix = torch.stack(flattened_embeddings)  # 形状为 [18, 77*512]
        return embeddings_matrix
    def forward(self, rgb_list, lang_input, proprio):
    # 确保 lang_input 可以计算梯度
        # rgb_list = [rgb_image.clone().detach().requires_grad_(True) if not rgb_image.requires_grad else rgb_image for rgb_image in rgb_list]
        # if not lang_input.requires_grad:
        #     lang_input = lang_input.clone().detach()  # 克隆一份，并确保与原来的计算图脱离
        #     lang_input.requires_grad_(True)  # 设置为可微分
        # if not proprio.requires_grad:
        #     proprio = proprio.clone().detach()  # 克隆一份，并确保与原来的计算图脱离
        #     proprio.requires_grad_(True)  # 设置为可微分
        # print("lang_input" ,lang_input.requires_grad)
        # print("proprio ",proprio.requires_grad)

        rgb_features_list = []
        for rgb_image in rgb_list:
            features = self.rgb_extractor(rgb_image)
            features = self.avgpool(features).view(rgb_image.size(0), -1)  
            rgb_features_list.append(features)

        rgb_combined_features = torch.cat(rgb_features_list, dim=1)

        lang_features = self.language_pool(lang_input.permute(0, 2, 1)).squeeze(-1)  # [b, 512]
        lang_features = F.relu(self.language_fc(lang_features)) 

        # print("RGB combined features shape:", rgb_combined_features.shape)
        # print("proprio combined features shape:", proprio.shape)
        combined_features = torch.cat((rgb_combined_features, lang_features, proprio), dim=1)


        
        # print("Combined features shape:", combined_features.shape)
        logits = self.fc1(combined_features)
        probs = F.softmax(logits, dim=1)
        print(probs.shape)
        print(probs)
        predicted_class = torch.argmax(probs, dim=1)
        print(predicted_class)
        one_hot_predicted_class = torch.nn.functional.one_hot(predicted_class, num_classes=self.num_class)
        print(one_hot_predicted_class)
        print(self.embeddings_matrix.shape)
        # print(predicted_class) 
        # 将 one-hot 编码和 embeddings 矩阵相乘，得到 [1, 77*512] 的向量
        selected_embedding_flat = torch.matmul(one_hot_predicted_class, self.embeddings_matrix)  # [1, 77*512]

        # 将该向量还原为 [77, 512] 的形状
        selected_embedding = selected_embedding_flat.view(77, 512)
        print(selected_embedding.shape)
        
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