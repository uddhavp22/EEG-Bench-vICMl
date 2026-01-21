from typing import Dict
from x_transformers import AttentionPool, Encoder
import torch
import torch.nn as nn

class AttentivePooler(nn.Module):
    """Attentive Pooler using x-transformers AttentionPool"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,  # depth>1: use transformer blocks after pooling
        init_std=0.02,
        attn_flash=False,
        **kwargs
    ):
        super().__init__()
        # x-transformers AttentionPool pools context into num_queries tokens
        self.pooler = AttentionPool(
            dim=embed_dim,
            num_pooled_tokens=num_queries,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            depth=1,  # single cross-attention (can be >1 if you want stacking)
            attn_kwargs={"flash": attn_flash},
        )
        # Optionally add transformer blocks after pooling
        self.blocks = None
        if depth > 1:
            self.blocks = Encoder(
                dim=embed_dim,
                depth=depth - 1,
                heads=num_heads,
                ff_glu=True,  # x-transformers default for MLP/FFN
                ff_mult=mlp_ratio,
            )

    def forward(self, x, mask=None):
        # x: [batch, seq, dim]
        pooled = self.pooler(x, mask=mask)  # [batch, num_queries, dim]
        if self.blocks is not None:
            pooled = self.blocks(pooled)
        return pooled



class AttentivePoolerAdapter(nn.Module):
    """Wrapper around AttentivePooler to return logits format"""

    def __init__(self, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.pooler = AttentivePooler(**kwargs)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask=None) -> Dict[str, torch.Tensor]:
        x = self.dropout(x)
        pooled = self.pooler(x, mask=mask)  # (B, num_queries, embed_dim)
        return {"logits": pooled}