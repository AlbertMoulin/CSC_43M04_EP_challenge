import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, BertModel, DistilBertModel


class CLIPWrapper(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch32", frozen=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(name)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.frozen = frozen
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, texts):
        """
        Args:
            images: list of PIL.Image or batched tensors (B, C, H, W)
            texts: list of strings or batch of tokenized strings

        Returns:
            image_embeds: Tensor (B, D)
            text_embeds: Tensor (B, D)
        """
        device = next(self.model.parameters()).device
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.image_embeds, outputs.text_embeds
    
class DinoV2BertMultimodalEncoder(nn.Module):

    def __init__(self, output_dim=None, frozen_image=False, frozen_text=True, bert_model_name="distilbert-base-uncased",patchs=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.backbone.head = nn.Identity()
        self.img_dim = self.backbone.norm.normalized_shape[0]
        if frozen_image:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.text_dim = self.bert.config.hidden_size
        if frozen_text:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.patchs = patchs
        self.output_dim = output_dim
        if output_dim is not None:
            self.img_proj = nn.Linear(self.img_dim, output_dim)
            self.text_proj = nn.Linear(self.text_dim, output_dim)
        
    
    def forward(self, x):

        if self.patchs:
            # IMAGE : [B, N_img, D]
            features = self.backbone.forward_features(x["image"])
            img_feat = features["x_norm_patchtokens"]  # remove [CLS] if needed
            if self.output_dim is not None: img_feat = self.img_proj(img_feat) 

            # TEXTE : [B, N_txt, D]
            text_out = self.bert(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"]
            )
            text_feat = text_out.last_hidden_state  # keep all tokens
            if self.output_dim is not None: text_feat = self.text_proj(text_feat)
            
            return img_feat, text_feat
        else:
            # x["image"]: Tensor [B, 3, H, W]
            # x["input_ids"], x["attention_mask"]: pour BERT
            img_feat = self.backbone(x["image"])  # [B, img_dim]
            if self.output_dim is not None: img_feat = self.img_proj(img_feat)

            # BERT encodeur (on prend le [CLS] token)
            text_out = self.bert(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"]
            )
            text_feat = text_out.last_hidden_state[:, 0, :]  # [B, text_dim]
            if self.output_dim is not None: text_feat = self.text_proj(text_feat)

            return img_feat, text_feat
    

