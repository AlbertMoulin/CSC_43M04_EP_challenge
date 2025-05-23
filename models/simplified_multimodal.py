import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class EnhancedPhase1LargeMLP(nn.Module):
    """
    Phase 1 enhanced text processing + [1024]*3 MLPs that you found work well.
    """
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 max_token_length=256,  # Enhanced from Phase 1
                 dropout_rate=0.1,      # Add some regularization
                 text_model_name='bert-base-uncased'):
        super().__init__()
        
        # Image branch with [1024]*3 MLP
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if dinov2_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Large image MLP: 768 -> 1024 -> 1024 -> 1024 -> 1
        self.image_mlp = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1)
        )
        
        # Enhanced text branch from Phase 1
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_backbone.config.hidden_size  # 768
        self.max_token_length = max_token_length
        
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Large text MLP: 768 -> 1024 -> 1024 -> 1024 -> 1
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1)
        )
        
        # Learnable weights (start with slight image bias since images often matter more for views)
        self.weight_image = nn.Parameter(torch.tensor(0.6))
        self.weight_text = nn.Parameter(torch.tensor(0.4))
    
    def _preprocess_text(self, raw_text):
        """Enhanced text preprocessing from Phase 1."""
        processed_text = []
        for text in raw_text:
            if isinstance(text, str):
                # Clean text
                text = text.strip()
                
                # Handle edge cases
                if len(text) < 3:
                    text = f"Short video: {text}" if text else "[EMPTY]"
                elif len(text) > 500:  # Very long titles
                    text = text[:500] + "..."
                
                processed_text.append(text)
            else:
                processed_text.append("[EMPTY]")
        return processed_text
    
    def forward(self, batch):
        # Image processing
        image_features = self.image_backbone(batch["image"])
        image_pred = self.image_mlp(image_features)
        
        # Enhanced text processing from Phase 1
        raw_text = list(batch["text"])
        processed_text = self._preprocess_text(raw_text)
        
        # Better tokenization
        encoded_text = self.tokenizer(
            processed_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt',
            add_special_tokens=True  # Ensure [CLS] and [SEP] tokens
        )
        
        device = batch["image"].device
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Text features
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_pred = self.text_mlp(text_features)
        
        # Combine predictions
        combined_pred = self.weight_image * image_pred + self.weight_text * text_pred
        
        return combined_pred


