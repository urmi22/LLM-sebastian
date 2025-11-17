import torch

from multihead_attention import MultiheadAttentionWrapper



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
    d_in = batch.shape[-1]
    d_out = 1
    context_length = batch.shape[1]
    mha = MultiheadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    print(f"Exercise-2 (change the output dimension to 2")
    print(mha(batch))
    print(f"Answer:")
    print(mha(batch).shape)



if __name__=="__main__":
    main()
