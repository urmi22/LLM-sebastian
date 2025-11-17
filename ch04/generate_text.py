# A function for the GPT model to generate text


import pdb
import torch
import torch.nn as nn

from GPT_model import GPTModel



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
        idx_next = torch.argmax(probas)

        # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx