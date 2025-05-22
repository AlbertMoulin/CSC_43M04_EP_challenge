import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SimplifiedMultimodalModel(nn.Module):
    """
    Simplified multimodal model based on the working approach.
    Focuses on image + text with simpler architecture.
    """
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 image_mlp_layers=[1024, 1024,1024, 1],
                 text_mlp_layers=[1024,1024, 1024, 1],
                 max_token_length=128,
                 text_model_name='bert-base-uncased'):
        super().__init__()
        
        # Image branch - DinoV2
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        self.image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if dinov2_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Build image MLP
        image_layers = []
        input_dim = self.image_dim
        for output_dim in image_mlp_layers[:-1]:
            image_layers.append(nn.Linear(input_dim, output_dim))
            image_layers.append(nn.ReLU())
            input_dim = output_dim
        image_layers.append(nn.Linear(input_dim, image_mlp_layers[-1]))
        self.image_mlp = nn.Sequential(*image_layers)
        
        # Text branch - BERT
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_backbone.config.hidden_size  # 768
        self.max_token_length = max_token_length
        
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Build text MLP
        text_layers = []
        input_dim = self.text_dim
        for output_dim in text_mlp_layers[:-1]:
            text_layers.append(nn.Linear(input_dim, output_dim))
            text_layers.append(nn.ReLU())
            input_dim = output_dim
        text_layers.append(nn.Linear(input_dim, text_mlp_layers[-1]))
        self.text_mlp = nn.Sequential(*text_layers)
        
        # Simple learnable weights (not softmax normalized)
        self.weight_image = nn.Parameter(torch.tensor(0.5))
        self.weight_text = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, batch):
        # Image processing
        image_features = self.image_backbone(batch["image"])
        image_pred = self.image_mlp(image_features)
        
        # Text processing with proper tokenization
        raw_text = batch["text"]
        
        # Convert to list if needed and tokenize consistently
        if isinstance(raw_text, (tuple, list)):
            text_list = list(raw_text)
        else:
            text_list = [raw_text] if isinstance(raw_text, str) else list(raw_text)
        
        # Tokenize with consistent padding
        encoded_text = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        )
        
        # Move to same device as image
        device = batch["image"].device
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Get text features (use [CLS] token)
        with torch.no_grad() if not self.text_backbone.training else torch.enable_grad():
            text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        text_pred = self.text_mlp(text_features)
        
        # Simple weighted combination (no softmax)
        combined_pred = self.weight_image * image_pred + self.weight_text * text_pred
        
        return combined_pred


class UltraSimpleModel(nn.Module):
    """
    Ultra-simplified version that exactly mirrors the working approach.
    """
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 max_token_length=128):
        super().__init__()
        
        # Image branch
        dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dinov2_backbone.head = nn.Identity()
        if dinov2_frozen:
            for param in dinov2_backbone.parameters():
                param.requires_grad = False
        
        self.image_backbone = dinov2_backbone
        image_dim = dinov2_backbone.norm.normalized_shape[0]
        
        # Simple image MLP: 768 -> 256 -> 1
        self.image_mlp = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Text branch
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_backbone = AutoModel.from_pretrained('bert-base-uncased')
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        text_dim = self.text_backbone.config.hidden_size
        self.max_token_length = max_token_length
        
        # Simple text MLP: 768 -> 256 -> 1
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Learnable weights
        self.weight_image = nn.Parameter(torch.tensor(0.5))
        self.weight_text = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, batch):
        # Image processing
        image_features = self.image_backbone(batch["image"])
        image_pred = self.image_mlp(image_features)
        
        # Text processing
        raw_text = list(batch["text"])
        
        encoded_text = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        )
        
        device = batch["image"].device
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Text features
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_pred = self.text_mlp(text_features)
        
        # Combine
        combined_pred = self.weight_image * image_pred + self.weight_text * text_pred
        
        return combined_pred