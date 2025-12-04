import tiktoken

from ch05.GPT_weight.gpt_download import download_and_load_gpt2
from ch04.GPT_model import GPTModel
from ch05.GPT_weight.load_GPT_weight import load_weights_into_gpt
from ch04.generate_text import generate_text_simple
from ch05.text_generation import text_to_token_ids, token_ids_to_text

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = { "vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True }
model_configs = { "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
                 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
                 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
                 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
                 }


BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2( model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

'''
After loading the model weights into the GPTModel, 
we reuse the text generation utility function from chapters 4 and 5 to ensure that the model generates coherent text

'''

tokenizer = tiktoken.get_encoding("gpt2")

text_1 = "Every effort moves you"
text_2 = ("Is the following text 'spam'? Answer with 'yes' or 'no':" " 'You are a winner you have been specially" 
          " selected to receive $1000 cash or a $2000 award.'" 
          )
token_ids = generate_text_simple(model=model,
                                 idx=text_to_token_ids(text_2, tokenizer),
                                 max_new_tokens=15,
                                 context_size=BASE_CONFIG["context_length"]
                                 )
print(token_ids_to_text(token_ids, tokenizer))