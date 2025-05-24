import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class MLP(nn.Module):
    """
    Simple MLP
    """
    def __init__(self, input_dim, output_dim, hidden_dim : list = [1024, 1024,1024],dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
        self.hidden_layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Initialize the MLP with the specified hidden layers
        self.mlp = nn.Sequential(*self.hidden_layers)
    
    def forward(self, x):
        # Forward pass through the MLP
        x = self.mlp(x)
        return x
        

class EnhancedPhase1LargeMLP(nn.Module):
    """
    Phase 1 enhanced text processing + [1024]*3 MLPs that you found work well.
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=[1024, 1024, 1024],
                 max_token_length=256,  # Enhanced from Phase 1
                 dropout_rate=0.1,      # Add some regularization
                 text_model_name="google/gemma-3-1b-it",
                 proportion_date=0.1,
                 proportion_channel=0.1):
        super().__init__()
        
        # Image branch with [1024]*3 MLP
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if image_model_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Enhanced text branch from Phase 1
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModelForCausalLM.from_pretrained(text_model_name,torch_dtype=torch.float16)
        text_dim = self.text_backbone.config.hidden_size # 1152
        self.max_token_length = max_token_length
        
        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Proportion of date in the final MLP
        self.proportion_date = proportion_date
        self.ind_date_dim = int(self.proportion_date * (image_dim + text_dim))
        # Proportion of channel in the final MLP
        self.proportion_channel = proportion_channel
        self.ind_channel_dim = int(self.proportion_channel * (image_dim + text_dim))

        # final MLP
        self.mlp = MLP(input_dim=image_dim + text_dim + self.ind_date_dim + self.ind_channel_dim , output_dim=1, hidden_dim=final_mlp_layers, dropout_rate=dropout_rate)
        
    
    def forward(self, batch):
        # Image processing
        image_features = self.image_backbone(batch["image"])
        
        # Enhanced text processing from Phase 1
        raw_text = list(batch["text"])
        
        # Better tokenization
        encoded_text = self.tokenizer(
            raw_text,
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
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = text_outputs.hidden_states[-1]  # [CLS] token
        last_token_hidden = last_hidden_state[:, -1, :]

        # Embedding the year date
        if "date" in batch:
            date = batch["date"].unsqueeze(1).to(device)
            # Ensure date_embedding is float32 and on the correct device
            date_embedding = date.repeat(1, self.ind_date_dim).to(dtype=image_features.dtype)

        if "channel" in batch:
            channel = batch["channel"].unsqueeze(1).to(device)
            # Ensure channel_embedding is float32 and on the correct device
            channel_embedding = channel.repeat(1, self.ind_channel_dim).to(dtype=image_features.dtype)

        # Concatenate image and text and date features
        combined_features = torch.cat((image_features, last_token_hidden, date_embedding, channel_embedding), dim=1)
        # MLP processing
        mlp_output = self.mlp(combined_features)
        return mlp_output
