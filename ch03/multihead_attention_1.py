'''

An efficient multi-head attention class.
In the MultiHeadAttentionWrapper class with two attention heads, we initialized two weight matrices, Wq1 and Wq2, and computed two query matrices, Q1 and Q2 (top). 
In the MultiheadAttention class, we initialize one larger weight matrix Wq, 
only perform one matrix multiplication with the inputs to obtain a query matrix Q, and then split the query matrix into Q1 and Q2 (bottom). 
We do the same for the keys and values.

'''


import pdb
import torch
import torch.nn as nn




class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # d_out must be divisible by num_heads
        assert(d_out % num_heads == 0)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_query(x)
        values = self.W_value(x)

        '''
        We implicitly split the matrix by adding a num_heads dimension. 
        Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).

        '''
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Computes dot product for each head
        attention_scores = queries @ keys.transpose(2, 3)

        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Uses the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vectors = (attention_weights @ values).transpose(1, 2)

        # Combines heads, where self.d_out  = self.num_heads * self.head_dim
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)

        #Adds an optional  linear projection
        context_vectors = self.out_proj(context_vectors)
        
        return context_vectors
    

def main():

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
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vectors = mha(batch)
    print(f"Compact Multihead Attention:")
    print(context_vectors)
    print(f"Shape: {context_vectors.shape}")



if __name__ == "__main__":
    main()



