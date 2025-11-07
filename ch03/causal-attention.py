import pdb
import torch
import torch.nn as nn



class SelfAttention_V2(nn.Module):
    def __init__(self, d_in,  d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        print(f"Attention_weights before masking:")
        print(attention_weights)

        #masking above the diagonal so that each token only have the access to its previous and current tokens, does not have the access to its future tokens.
        context_length = attention_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        # print(mask_simple)
        masked_simple = attention_weights * mask_simple
        # print(masked_simple)
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums
        print(f"masked with zeros:")
        print(masked_simple_norm)
        context_vectors = masked_simple_norm @ values
        print(f"Context vectors with zero masking:")
        print(context_vectors)
        
        '''
        another masking technique using -inf and softmax
        when we use this trick we do not need to compute the attention weights before,
        normalization will also be done in one time

        '''
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
        # print(masked)
        attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
        print(f"Masked with inf:")
        print(attention_weights)

        # dropout layer masking to prevent overfitting in the training
        dropout = torch.nn.Dropout(0.5)
        example = torch.ones(6,6)
        # print(dropout(example))
        print(f"Attention weights after dropout:")
        print(dropout(attention_weights))
        context_vectors = attention_weights @ values
        print(f"Context vectors:")
        print(context_vectors)
        


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
    sa_v2 = SelfAttention_V2(d_in, d_out)
    sa_v2(inputs)