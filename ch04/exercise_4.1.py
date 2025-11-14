'''
Calculate and compare the number of parameters that are contained in the feed forward module 
and those that are contained in the multi-head attention module.

'''

import pdb
import torch
import torch.nn as nn

from GPT_model import GPTModel
from feed_forward import FeedForward
from ch03.multihead_attention_1 import MultiHeadAttention
from transformer_block_GPT import TransformerBlock
from GPT_CONFIG_124M import GPT_CONFIG_124M as cfg


ff = FeedForward(cfg)
total_params = sum(p.numel() for p in ff.parameters())
print(f"Number of parameters in Feed Forward module: {total_params}")


# or we can calculate from transformer block
tf_block = TransformerBlock(cfg)
total_params_ff = sum(p.numel() for p in tf_block.ff.parameters())
total_params_mha = sum(p.numel() for p in tf_block.attention.parameters())
print(f"Number of parameters in Feed Forward module: {total_params_ff}")
print(f"Number of parameters in Multi-head attention module: {total_params_mha}")

'''
The results above are for a single transformer block
Optionally multiply by 12 to capture all transformer blocks in the 124M GPT model

'''

