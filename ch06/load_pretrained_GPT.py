import tiktoken
import torch

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

print(model)

# To get the model ready for classification fine-tuning, we first freeze the model, meaning that we make all layers nontrainable:
for param in model.parameters():
    param.requires_grad = False

# Adding a classification layer, change the originial out_head
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],
                                 out_features=num_classes
                                 )

# To make the final LayerNorm and last transformer block trainable, we set their  respective requires_grad to True
for param in model.transformer_block[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# For instance, we can feed it an example text identical to our previously used example text:
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

# Then, we can pass the encoded token IDs to the model as usual:
with torch.no_grad():
    outputs = model(inputs)

print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

'''
see page no 188-190, to know why we need only last output token
To extract the last output token from the output tensor, we use the following code

'''
print("Last output token:", outputs[:, -1, :])