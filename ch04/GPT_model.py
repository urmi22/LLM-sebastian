# The GPT model architecture implementation


import pdb
import torch
import tiktoken
import torch.nn as nn

from transformer_block_GPT import TransformerBlock
from layer_norm import LayerNorm
from GPT_CONFIG_124M import GPT_2_small as cfg



class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_block = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_idx):
        batch_size, seq_len = input_idx.shape
        tok_embeds = self.tok_emb(input_idx)

        # The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits



def main():

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim = 0)
    model = GPTModel(cfg)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    # weight tying
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = (total_params - sum(p.numel() for p in model.out_head.parameters()))
    print(f"Number of trainable parameters " f"considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    




if __name__ == "__main__":
    main()