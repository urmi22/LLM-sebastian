import pdb
import torch
import dataloader

from dataloader import GPTDatasetV1, create_dataloader_v1


# input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 50257
output_dim = 256

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight.shape)
# print(embedding_layer(torch.tensor([3])).shape)
# print(embedding_layer(input_ids).shape)

with open("the-verdict.txt", 'r', encoding="utf-8") as f1:
    raw_text = f1.read()

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, target = next(data_iter)
print(f"Input_tokens: {inputs}")
print(f"Target_tokens: {target}")

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)



