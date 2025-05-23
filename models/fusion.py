import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeds):
        return torch.cat(embeds, dim=1)  # Concaténer les embeddings le long de la dimension des caractéristiques
        
    
class Zorrofusion(nn.Module):
    def __init__(self, dim=512, n_fusion_tokens=1, n_heads=4, n_layers=1):
        super().__init__()
        self.n_fusion_tokens = n_fusion_tokens
        self.dim = dim
        self.fusion_tokens = nn.Parameter(torch.randn(1, n_fusion_tokens, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def build_zorro_mask(self, n_img, n_txt, n_fusion, device):
        N = n_img + n_txt + n_fusion
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        # Image tokens: voient eux-mêmes + fusion
        mask[:n_img, :n_img] = False
        mask[:n_img, n_img:n_img+n_txt] = True
        mask[:n_img, n_img+n_txt:] = False
        # Texte tokens: voient eux-mêmes + fusion
        mask[n_img:n_img+n_txt, :n_img] = True
        mask[n_img:n_img+n_txt, n_img:n_img+n_txt] = False
        mask[n_img:n_img+n_txt, n_img+n_txt:] = False
        # Fusion tokens: voient tout
        mask[n_img+n_txt:, :] = False
        return mask  # (N, N)

    def forward(self, embed):
        image_embed, text_embed = embed
        if image_embed.dim() == 2:
            image_embed = image_embed.unsqueeze(1)  # (B, 1, D)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)    # (B, 1, D)
        B = image_embed.size(0)
        
        n_img = image_embed.size(1)
        n_txt = text_embed.size(1)
        n_fusion = self.n_fusion_tokens

        fusion_tokens = self.fusion_tokens.expand(B, -1, -1)  # (B, n_fusion, D)
        tokens = torch.cat([image_embed, text_embed, fusion_tokens], dim=1)  # (B, N, D)
        attn_mask = self.build_zorro_mask(n_img, n_txt, n_fusion, tokens.device)  # (N, N)

        out = self.transformer(tokens, mask=attn_mask)
        image_out = out[:, :n_img, :].mean(dim=1)
        text_out = out[:, n_img:n_img+n_txt, :].mean(dim=1)
        fusion_out = out[:, n_img+n_txt:, :].mean(dim=1)
        return torch.cat([image_out, text_out, fusion_out],dim=1)