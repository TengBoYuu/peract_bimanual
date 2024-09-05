import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel
import pickle
import os

class OptionSelector:
    def __init__(self, vocabulary=None, embedding_file="template_embeddings.pkl", model_name="openai/clip-vit-base-patch32", recalculate=False):
        """
        初始化 OptionSelector 类。

        参数:
        - vocabulary: 语言模板字典
        - embedding_file: 存储模板嵌入的文件路径
        - model_name: 使用的CLIP模型名称
        - recalculate: 是否重新计算模板嵌入
        """
        self.embedding_file = embedding_file
        self.lang_template_embs = None
        self.template_names = None
        
        if not recalculate and os.path.exists(embedding_file):
            # 如果文件存在并且不要求重新计算，则从文件中加载嵌入
            with open(embedding_file, 'rb') as f:
                self.lang_template_embs, self.template_names = pickle.load(f)
        elif vocabulary is not None:
            # 如果需要重新计算嵌入或提供了新的词汇表，则初始化模型并计算嵌入
            self.vocabulary = vocabulary
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            self.lang_template_embs, self.template_names = self._encode_templates()
            with open(embedding_file, 'wb') as f:
                pickle.dump((self.lang_template_embs, self.template_names), f)
        else:
            raise ValueError("Vocabulary must be provided if recalculation is needed.")

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
    
    def __call__(self, input_emb):
        """
        使得类实例可以像函数一样被调用。

        参数:
        - input_emb: 输入的语言嵌入，形状为 [b, 77, 512]

        返回:
        - 返回形状为 [b, 77, 512] 的最相似的模板嵌入
        """
        return self.find_most_similar(input_emb)