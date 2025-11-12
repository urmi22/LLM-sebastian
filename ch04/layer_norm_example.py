# Execution of Layer Normalization by an example

import pdb
import torch
import torch.nn as nn
import tiktoken

from dummy_GPT_model import DummyGPTModel
from GPT_CONFIG_124M import GPT_CONFIG_124M as cfg
from layer_norm import LayerNorm









if __name__ == "__main__":

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim = 0)
    # print(batch)

    model = DummyGPTModel(cfg)
    logits = model(batch)
    print(f"Output shape:\n {logits.shape}")
    print(f"Output:\n {logits}")

    
    # Example of layer normalization creating a batch with random numbers
    batch_example = torch.randn(2, 5)
    print(f"batch_example:\n {batch_example}")


    # use layer normalization without LayerNorm class
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)

    # Before we apply layer normalization to these outputs, let’s examine the mean and variance
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print(f"Before Layer Normalization")
    print(f"-----------------------------")
    print(f"out: {out}")
    print(f"Mean: {mean},\n Variance: {var}\n")

    # Now we going to apply the layer normalization --> ((x - miu) / sigma)), sigma--> sqrt(var)
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print(f"After Layer Normalization")
    print(f"-----------------------------")
    print(f"Normalized layer outputs:\n {out_norm}")

    # # for better readability
    torch.set_printoptions(sci_mode=False)
    print(f"Mean: {mean},\n Variance: {var}\n")

    # try the LayerNorm module in practice and apply it to the batch input
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    print(f"Using LayerNorm class")
    print(f"-----------------------")
    print(f"out_ln: {out_ln}\n")
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print(f"Mean: {mean},\n Variance: {var}")

