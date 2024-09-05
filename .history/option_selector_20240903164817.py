import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel
import pickle
import os

class OptionSelector:
    def __init__(self, vocabulary, embedding_file="template_embeddings.pkl", model_name="openai/clip-vit-base-patch32", recalculate=False):
        """
        初始化 OptionSelector 类。

        参数:
        - vocabulary: 语言模板字典
        - embedding_file: 存储模板嵌入的文件路径
        - model_name: 使用的CLIP模型名称
        - recalculate: 是否重新计算模板嵌入
        """
        self.vocabulary = vocabulary
        self.embedding_file = embedding_file
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        if not recalculate and os.path.exists(embedding_file):
            # 如果文件存在并且不要求重新计算，则从文件中加载嵌入
            with open(embedding_file, 'rb') as f:
                self.lang_template_embs, self.template_names = pickle.load(f)
        else:
            # 否则计算嵌入并保存到文件
            self.lang_template_embs, self.template_names = self._encode_templates()
            with open(embedding_file, 'wb') as f:
                pickle.dump((self.lang_template_embs, self.template_names), f)

    def _encode_templates(self):
        """
        编码语言模板。

        返回:
        - lang_template_embs: 编码后的模板嵌入
        - template_names: 模板名称列表
        """
        lang_template_embs = []
        template_names = []

        for task_name, templates in self.vocabulary.items():
            for template in templates:
                inputs = self.tokenizer(template, return_tensors="pt")
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs).squeeze(0)
                lang_template_embs.append(emb)
                template_names.append(task_name)
        
        return lang_template_embs, template_names

    def compute_similarity(self, emb1, emb2):
        """
        计算两个嵌入向量之间的余弦相似度。

        参数:
        - emb1: 第一个嵌入向量
        - emb2: 第二个嵌入向量

        返回:
        - 余弦相似度值
        """
        return F.cosine_similarity(emb1, emb2, dim=-1)  # 在最后一维上计算余弦相似度

    def find_most_similar(self, input_emb):
        """
        计算输入嵌入与已知语言模板嵌入之间的相似度，并返回形状相同的最相似的模板嵌入。

        参数:
        - input_emb: 输入的语言嵌入，形状为 [b, 77, 512]

        返回:
        - 返回形状为 [b, 77, 512] 的最相似的模板嵌入
        """
        b, seq_len, emb_dim = input_emb.shape
        output_emb = torch.zeros_like(input_emb)

        for i in range(b):
            for j in range(seq_len):
                similarities = []
                for template_emb in self.lang_template_embs:
                    similarity = self.compute_similarity(input_emb[i, j], template_emb)
                    similarities.append(similarity.item())

                # 找到最大相似度的索引
                max_index = similarities.index(max(similarities))
                
                # 将最相似的模板嵌入赋值给输出嵌入
                output_emb[i, j] = self.lang_template_embs[max_index]
        
        return output_emb

# lang template
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
