# Consider these two input examples, which have already been mapped to token IDs

import pdb
import torch
import tiktoken
import torch.nn as nn

from ch04.GPT_model import GPTModel
from GPT_CONFIG_124M import GPT_2_small as cfg
from ch05.generate_text import token_ids_to_text




def main():
    torch.manual_seed(123)
    inputs = torch.tensor([[16833, 3626, 6100],    # ["every effort moves", 
                           [40, 1107, 588]]        # "I really like"]
                         )

    targets = torch.tensor([[3626, 6100, 345 ],     # [" effort moves you", 
                            [1107, 588, 11311]]     # " really like chocolate"]
                           )
    
    tokenizer = tiktoken.get_encoding("gpt2")
    '''
    Now we feed the inputs into the model to calculate logits vectors for the two input examples, 
    each comprising three tokens. 
    Then we apply the softmax function to transform these logits into probability scores

    '''
    model = GPTModel(cfg)
    with torch.no_grad():
        logits = model(inputs)

    print(logits.shape)
    probas = torch.softmax(logits, dim=-1)
    print(f"Probabilities of Inputs: {probas}")
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print(f"Output Token IDs: {token_ids}")

    target_text = token_ids_to_text(targets[0], tokenizer)
    output_text = token_ids_to_text(token_ids[0].flatten(), tokenizer)
    print(f"Targets batch 0: {target_text}")
    print(f"Output batch 0: {output_text}")

    # we can print the initial softmax probability scores corresponding to the target tokens using the following code
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 1: {target_probas_1}")
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 2: {target_probas_2}")

    # calculate negative average log probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(f"log probas: {log_probas}")
    avg_log_probas = torch.mean(log_probas) 
    print(f"average log prob: {avg_log_probas}")
    neg_avg_log_probas = avg_log_probas * -1
    print(f"Negative average log prob: {neg_avg_log_probas}")

    # calculate cross-entropy loss
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    '''
    For the cross_entropy loss function in PyTorch, we want to flatten these tensors by combining them over the batch dimension
    PyTorch’s cross_entropy function will take care of all these steps for us.

    '''
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"Cross entropy loss: {loss}")
    perplexity = torch.exp(loss)
    print(f"Perplexity: {perplexity}")
    









if __name__=="__main__":
    main()