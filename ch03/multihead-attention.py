# A wrapper class to implement multi-head attention

'''
if we use this MultiHeadAttentionWrapper class with two attention heads (via num_heads=2) 
and CausalAttention output dimension d_out=2, we get a fourdimensional context vector (d_out*num_heads=4)
'''

import pdb
import torch
import torch.nn as nn

from ca_1 import CausalAttention

torch.manual_seed(123)

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], #your  --> x1
        [0.55, 0.87, 0.66], #journey  --> x2
        [0.57, 0.85, 0.64], #starts  --> x3
        [0.22, 0.58, 0.33], #with  --> x4
        [0.77, 0.25, 0.10], #one  --> x5
        [0.05, 0.80, 0.55] #step  --> x6
    ]
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
d_in = batch.shape[-1]
d_out = 2
context_length = batch.shape[1]

class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

mha = MultiheadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch)
print(f"Context_vectors using Multihead Attention:")
print(context_vectors)
print(f"Context vectors shape:")
print(context_vectors.shape)