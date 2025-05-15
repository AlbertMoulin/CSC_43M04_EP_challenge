import torch
import torch.nn as nn
from transformers import BertModel


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.norm.normalized_shape[0], 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x["image"])
        x = self.regression_head(x)
        return x

class DinoV2BertMultimodalRegressor(nn.Module):
    def __init__(self, frozen_image=False, frozen_text=True, bert_model_name="bert-base-uncased"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.img_dim = self.backbone.norm.normalized_shape[0]
        if frozen_image:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_dim = self.bert.config.hidden_size
        if frozen_text:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.regression_head = nn.Sequential(
            nn.Linear(self.img_dim + self.text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x["image"]: Tensor [B, 3, H, W]
        # x["input_ids"], x["attention_mask"]: pour BERT
        img_feat = self.backbone(x["image"])  # [B, img_dim]

        # BERT encodeur (on prend le [CLS] token)
        text_out = self.bert(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"]
        )
        text_feat = text_out.last_hidden_state[:, 0, :]  # [B, text_dim]

        # Fusion
        features = torch.cat([img_feat, text_feat], dim=1)
        out = self.regression_head(features)
        return out.squeeze(-1)