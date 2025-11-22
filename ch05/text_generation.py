'''
Here, we have used two utility functions
1) text_to_token_ids, 2) token_ids_to_text
We have used chapter 04 generate_text_simple function along with the above two to generate the text

This model is generated gibberish because we haven't trained it yet.

'''


import pdb
import torch
import tiktoken
import torch.nn as nn


from ch04.GPT_model import GPTModel
from ch05.GPT_CONFIG_124M import GPT_2_small as cfg
from ch04.generate_text import generate_text_simple



def text_to_token_ids(text, tokenizer):

    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # .unsqueeze(0) adds the batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):

    # Removes batch dimension
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def main():
    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.eval()
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=cfg["context_length"])
    output_text = token_ids_to_text(token_ids, tokenizer)

    print(f"Output text: {output_text}")
    








if __name__=="__main__":
    main()