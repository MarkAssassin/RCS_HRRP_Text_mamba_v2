import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# def encode_text(text):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased').to('cuda' if torch.cuda.is_available() else 'cpu')
#     # MLP
#     # mlp = nn.Sequential(
#     #     nn.Linear(768, 512),
#     #     nn.ReLU(),
#     #     nn.Linear(512, 256)
#     # ).to(model.device)
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         cls_feature = outputs.pooler_output  # 取[CLS] token的向量作为文本特征
#     # return mlp(cls_feature)
#     return cls_feature

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
from safetensors.torch import save_file


class TextEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # model_name = "prajjwal1/bert-tiny"
        load_model_dir = './local_bert_tiny'

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.bert = AutoModel.from_pretrained(model_name, weight_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(load_model_dir)
        self.bert = AutoModel.from_pretrained(load_model_dir)

        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[RCS]", "[HRRP]"]})
        # self.bert.resize_token_embeddings(len(self.tokenizer))

        # 冻结 BERT（对雷达任务几乎一定要冻结）
        for p in self.bert.parameters():
            p.requires_grad = False
        

        self.bert.to(self.device)
        self.bert.eval()

        # 可选：维度压缩，减少后续显存 & 计算
        # self.mlp = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.ReLU()
        # ).to(self.device)

    @torch.no_grad()
    def forward(self, texts):
        """
        texts: List[str]
        """
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',  # 按最长文本进行padding'',
            truncation=True,
            max_length=128   # ⚠ 很重要，限制长度
        ).to(self.device)

        outputs = self.bert(**inputs)
        cls = outputs.pooler_output # (样本数, 128)
        return cls
