import torch
import torch.nn as nn

from multihead_attention_1 import MultiHeadAttention

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
context_length = 1024
d_in, d_out = 768, 768
num_heads = 12

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)

