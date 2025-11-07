# sa --> self-attention
# This code will generate context vector for all the inputs using self attention

import pdb
import torch
import torch.nn as nn





class SelfAttention_V1(nn.Module):
    def __init__(self, d_in,  d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vectors = attention_weights @ values
        
        return context_vectors
    


if __name__ == "__main__":

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
    d_in = inputs.shape[1]
    d_out = 2
    sa_v1 = SelfAttention_V1(d_in, d_out)
    print(sa_v1(inputs))

