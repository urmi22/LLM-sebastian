# The transformer block component of GPT

import pdb
import torch
import torch.nn as nn

from ch03.multihead_attention_1 import MultiHeadAttention
from GPT_CONFIG_124M import GPT_CONFIG_124M as cfg
from layer_norm import LayerNorm
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
            )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)

        # Add the original input back
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)

        #Adds the original input back
        x = x + shortcut

        return x
    



if __name__ == "__main__":
    torch.manual_seed(123)
    # Creates sample input of shape [batch_size, num_tokens, emb_dim]
    x = torch.rand(2, 4, 768)
    transformer_block = TransformerBlock(cfg)
    output = transformer_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

