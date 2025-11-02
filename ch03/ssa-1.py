#ssa---> simplified  self-attention
# this code will generate context vector for all the inputs

import pdb
import torch

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

attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])
# print(attention_scores)

# using for loops
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i, j] = torch.dot(x_i, x_j)

# print(attention_scores)

# for loops are generally slow, and we can achieve the same results using matrix multiplication
attention_scores = inputs @ inputs.T
print(f"Attention_scores: {attention_scores}\n")

attention_weights = torch.softmax(attention_scores, dim=-1)
print(f"Attention_weights: {attention_weights}")

# We can verify that the rows indeed all sum to 1 
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attention_weights.sum(dim=-1))

all_context_vectors = attention_weights @ inputs
print(f"All context vectors: {all_context_vectors}")
