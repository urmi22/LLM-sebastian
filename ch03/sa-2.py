# same as sa-1 except that, instead of nn.Parameter we use nn.Linear layer
'''
because, nn.Linear effectively perform matrix multiplication when the bias units are disabled. 
Additionally, a significant advantage of using nn.Linear is instead of manually implementing nn.Parameter(torch.rand(...))
nn.Linear  has an optimized weight initialization scheme, contributing to more stable and effective model training.

'''

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
        '''
        pass x to an Linear layer of dim (3,2), so x-->(6,3) and W of Linear layer is (3,2) so after self.W_key(x), we will get keys-->(6,2)
        same for queries and values

        '''
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vectors = attention_weights @ values
        
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

    d_in = inputs.shape[1]
    d_out = 2
    sa_v2 = SelfAttention_V2(d_in, d_out)
    print(sa_v2(inputs))



if __name__ == "__main__":
    main()