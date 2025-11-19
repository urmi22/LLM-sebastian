'''
An utility function for the GPT model to generate text.

This model is generated gibberish because we haven't trained it yet.
So far, we have only implemented the GPT architecture and initialized a GPT model instance with initial random weights.

'''


import pdb
import torch
import torch.nn as nn
import tiktoken

from ch04.GPT_model import GPTModel
from GPT_CONFIG_124M import GPT_2_small as cfg



def generate_text_simple(model, idx, max_new_tokens, context_size):

    # idx is a (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):

        '''
        Crops current context if it exceeds the supported context size, e.g., 
        if LLM supports only 5 tokens, and the context size is 10, 
        then only the last 5 tokens are used as context
        which is done in idx_cond

        '''
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # probas has shape (batch, vocab_size).
        probas = torch.softmax(logits, dim=-1)

        # idx_next has shape (batch, 1).
        idx_next = torch.argmax(probas, keepdim=True)

        # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():

    torch.manual_seed(123)
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print(f"encoded: {encoded}")

    # adding batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"encoded tensor shape: {encoded_tensor.shape}")

    model = GPTModel(cfg)
    # Disables dropout since we are not training the model
    model.eval()
    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=cfg["context_length"])
    print(f"output: {out}")
    print(f"length of output: {len(out[0])}")
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"Decoded text: {decoded_text}")
    


if __name__ == "__main__":
    main()
