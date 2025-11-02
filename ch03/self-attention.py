# self-attention is also called scaled dot-product attention
# computing context vector, z2, for input x2

import pdb
import torch

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

x2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x2 @ W_query

# key_2 = x2 @ W_key
# value = x2 @ W_value

# we still require the key and value vectors for all input elements as they are involved in computing the attention weights with respect to the query_2
# we can get key_2 from keys
keys = inputs @ W_key
values = inputs@ W_value
print(f"Keys shape: {keys.shape}")
print(f"values shape: {values.shape}")


# keys_2 = keys[1]
# torch.dot() only works for 1D tensors. If you pass 2D or higher tensors, you’ll get an error
# attention_score_22 = torch.dot(query_2, keys_2)

# for that reason we use the following line of code
# attention_score_22 = query_2.dot(keys_2)
# print(attention_score_22)

attention_scores_2 = query_2 @ keys.T
print(f"Attention_scores: {attention_scores_2}")

#scaled dot-product ----> we scale the attention scores by dividing them by the square root of the embedding dimension of the keys.
d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k ** 0.5, dim=-1)
print(f"Attention weights: {attention_weights_2}")

# The last step is multiplying each value vector with its respective attention weight and then summing them to obtain the context vector
context_vector_2 = attention_weights_2 @ values
print(f"Context vector: {context_vector_2}")

