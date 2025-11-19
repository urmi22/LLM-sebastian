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
    print(f"Token IDs: {token_ids}")

    target_text = token_ids_to_text(targets[0], tokenizer)
    output_text = token_ids_to_text(token_ids[0].flatten(), tokenizer)
    print(f"Targets batch 0: {target_text}")
    print(f"Output batch 0: {output_text}")
    pdb.set_trace()









if __name__=="__main__":
    main()