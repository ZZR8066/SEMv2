import math
import torch 
import torch.nn as nn
from mmcv.cnn import ConvModule


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim
        )
        self.max_positions = int(1e5)

    def get_embedding(num_embeddings: int, embedding_dim: int):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(
        self,
        positions
    ):
        self.weights = self.weights.to(positions.device)
        return (
            self.weights[positions.reshape(-1)]
            .view(positions.size() + (-1,))
            .detach()
        )


class SinusoidalPositionalConv(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(in_channels//2)
        self.pos_conv = ConvModule(in_channels, in_channels, kernel_size=1, conv_cfg=None, act_cfg=None)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, feats):
        device = feats.device
        batch_size, channels, y_dim, x_dim = feats.size()

        xx_pos = torch.arange(x_dim).repeat(batch_size, y_dim, 1).to(device=device) # (B, H, W)
        yy_pos = torch.arange(y_dim).repeat(batch_size, x_dim, 1).transpose(1, 2).to(device=device) # (B, H, W)

        xx_pos_embeddings = self.pos_embed(xx_pos).permute(0, 3, 1, 2).contiguous() # (B, C/2, H, W)
        yy_pos_embeddings = self.pos_embed(yy_pos).permute(0, 3, 1, 2).contiguous() # (B, C/2, H, W)
        pos_embeddings = torch.cat([xx_pos_embeddings, yy_pos_embeddings], dim=1) # (B, C, H, W)
        pos_embeddings = self.pos_conv(pos_embeddings)

        feats = feats.permute(0,2,3,1).reshape(-1, channels).contiguous() # (N, C)
        pos_embeddings = pos_embeddings.permute(0,2,3,1).reshape(-1, channels).contiguous() # (N, C)
        feats = self.norm(feats + pos_embeddings) # (N, C)
        feats = feats.view(batch_size, y_dim, x_dim, channels).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        return feats
    

def build_posemb_head(cfg):
    posemb_head = SinusoidalPositionalConv(cfg['in_channels'])
    return posemb_head