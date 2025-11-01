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

query = inputs[1]

#attention score of x2
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)
print(attention_scores_2)

# divided by sum normalization
attention_weights_2_tmp = attention_scores_2 / attention_scores_2.sum()
print(f"Attention weights: {attention_weights_2_tmp}")

#use softmax function for normalization --> 1) better at managing extreme values, 2) offers more favorable gradients  propoerties during training


pdb.set_trace()
