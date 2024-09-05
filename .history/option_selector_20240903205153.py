import torch
import torch.nn.functional as F
import pickle

class OptionSelector:
    def __init__(self, embedding_file="template_embeddings.pkl"):
        """
        初始化 OptionSelector 类。

        参数:
        - embedding_file: 存储模板嵌入的文件路径
        """
        self.embedding_file = embedding_file
        self.lang_template_embs = None
        with open(embedding_file, 'rb') as f:
            self.lang_template_embs = pickle.load(f)

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
            best_similarity = float('-inf')
            best_emb = None
            for task_name, template_embs in self.lang_template_embs.items():
                for template_emb in template_embs:
                    print(template_emb.shape) # [77, 512]
                    template_emb = template_emb.to(input_emb.device)
                    similarity = self.compute_similarity(input_emb, template_emb)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_emb = template_emb

                # 将最相似的模板嵌入赋值给输出嵌入
            output_emb[i] = best_emb
        
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