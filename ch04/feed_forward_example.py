import pdb
import torch
import torch.nn as nn

from feed_forward import FeedForward
from GPT_CONFIG_124M import GPT_2_small as cfg


def main():

    ffn = FeedForward(cfg)

    # Creates sample input with batch dimension 2
    x = torch.randn(2, 3, 768)
    out = ffn(x)
    print(f"Output dimension: {out.shape}")



if __name__ == "__main__":
    main()
