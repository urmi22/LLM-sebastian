import pdb
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


with open("the-verdict.txt", 'r', encoding="utf-8") as f1:
    raw_text = f1.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:       {y}")


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"{context} ------> {desired}")
    print(f"{tokenizer.decode(context)} ------> {tokenizer.decode([desired])}")


