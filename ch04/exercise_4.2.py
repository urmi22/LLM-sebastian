'''
We initialized a 124-million-parameter GPT model, which is known as “GPT-2 small.” 
Without making any code modifications besides updating the configuration file, use the GPTModel class to implement GPT-2 medium (using 1,024-dimensional embeddings, 
24 transformer blocks, 16 multi-head attention heads), GPT-2 large (1,280dimensional embeddings, 36 transformer blocks, 20 multi-head attention heads), and 

GPT-2 XL (1,600-dimensional embeddings, 48 transformer blocks, 25 multi-head attention heads). 

As a bonus, calculate the total number of parameters in each GPT model.
'''

from GPT_model import GPTModel
from feed_forward import FeedForward
from ch03.multihead_attention_1 import MultiHeadAttention
from transformer_block_GPT import TransformerBlock
from GPT_CONFIG_124M import GPT_2_small as cfg0
from GPT_CONFIG_124M import GPT_2_medium as cfg1
from GPT_CONFIG_124M import GPT_2_large as cfg2
from GPT_CONFIG_124M import GPT_2_xl as cfg3



def calculate_size(model):

    if model == 'small':

        model = GPTModel(cfg0)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        out_head_params = sum(p.numel() for p in model.out_head.parameters())
        num_trainable_params = total_params - out_head_params
        print(f"Total number of tarinable parameters: {num_trainable_params}")

        tf_block = TransformerBlock(cfg0)
        total_params_ff = sum(p.numel() for p in tf_block.ff.parameters())
        total_params_mha = sum(p.numel() for p in tf_block.attention.parameters())
        print(f"Number of parameters in Feed Forward module: {total_params_ff}")
        print(f"Number of parameters in Multi-head attention module: {total_params_mha}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB\n")


    elif model == 'medium':

        model = GPTModel(cfg1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        out_head_params = sum(p.numel() for p in model.out_head.parameters())
        num_trainable_params = total_params - out_head_params
        print(f"Total number of tarinable parameters: {num_trainable_params}")

        tf_block = TransformerBlock(cfg1)
        total_params_ff = sum(p.numel() for p in tf_block.ff.parameters())
        total_params_mha = sum(p.numel() for p in tf_block.attention.parameters())
        print(f"Number of parameters in Feed Forward module: {total_params_ff}")
        print(f"Number of parameters in Multi-head attention module: {total_params_mha}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB\n")


    elif model == 'large':

        model = GPTModel(cfg2)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        out_head_params = sum(p.numel() for p in model.out_head.parameters())
        num_trainable_params = total_params - out_head_params
        print(f"Total number of tarinable parameters: {num_trainable_params}")

        tf_block = TransformerBlock(cfg2)
        total_params_ff = sum(p.numel() for p in tf_block.ff.parameters())
        total_params_mha = sum(p.numel() for p in tf_block.attention.parameters())
        print(f"Number of parameters in Feed Forward module: {total_params_ff}")
        print(f"Number of parameters in Multi-head attention module: {total_params_mha}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB\n")


    elif model == 'xl':

        model = GPTModel(cfg3)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        out_head_params = sum(p.numel() for p in model.out_head.parameters())
        num_trainable_params = total_params - out_head_params
        print(f"Total number of tarinable parameters: {num_trainable_params}")

        tf_block = TransformerBlock(cfg3)
        total_params_ff = sum(p.numel() for p in tf_block.ff.parameters())
        total_params_mha = sum(p.numel() for p in tf_block.attention.parameters())
        print(f"Number of parameters in Feed Forward module: {total_params_ff}")
        print(f"Number of parameters in Multi-head attention module: {total_params_mha}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB\n")

    '''
    The results above are for a single transformer block
    Optionally multiply by 12 to capture all transformer blocks in the 124M GPT model

    '''


if __name__ == "__main__":
    models = ['small', 'medium', 'large', 'xl']
    for model in models:
        model_name = model
        print(f"Model name: {model_name}\n--------------")
        calculate_size(model)