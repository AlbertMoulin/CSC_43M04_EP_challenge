import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


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
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.image_embeds, outputs.text_embeds