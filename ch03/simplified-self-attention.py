# generate context vector for x2

import pdb
import torch


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

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

query = inputs[1]

#attention score of x2
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)
print(attention_scores_2)

# divided by sum normalization
attention_weights_2_tmp = attention_scores_2 / attention_scores_2.sum()
print(f"Attention weights: {attention_weights_2_tmp}")
print(f"Sum: {attention_weights_2_tmp.sum()}")

#use softmax function for normalization --> 1) better at managing extreme values, 2) offers more favorable gradients  propoerties during training
#                                           3) ensures attention weights betweeb 0 to 1 (always positive)

attention_weights_2_naive = softmax_naive(attention_scores_2)
print(f"Attention weights: {attention_weights_2_naive}")
print(f"Sum: {attention_weights_2_naive.sum()}")

# softmax_naive may produce numerical instability for large or smaill imput values, so its better to use PyTorch's in built softmax function
attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
print(f"Attention weights: {attention_weights_2}")
print(f"Sum: {attention_weights_2.sum()}")

context_vector_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vector_2 += attention_weights_2[i] * x_i
print(context_vector_2)

pdb.set_trace()
