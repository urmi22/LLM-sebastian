import tiktoken
import torch
import pdb

from ch05.GPT_weight.gpt_download import download_and_load_gpt2
from ch04.GPT_model import GPTModel
from ch05.GPT_weight.load_GPT_weight import load_weights_into_gpt
from ch04.generate_text import generate_text_simple
from ch05.text_generation import text_to_token_ids, token_ids_to_text
from ch06.dataloader import main
from ch06.utils import calc_accuracy_loader, calc_loss_loader


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
                                 idx=text_to_token_ids(text_1, tokenizer),
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
see page no 188-190, to know why we need only last output token.
The last token is the only token with an attention score to all other tokens.
To extract the last output token from the output tensor, we use the following code

'''
print("Last output token:", outputs[:, -1, :])
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print(f"Class label: {label.item()}")

'''
Using the softmax function here is optional because the largest outputs directly correspond to the highest probability scores. 
Hence, we can simplify the code without using softmax

'''
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print(f"Class label: {label.item()}")

# Let’s use the function to determine the classification accuracies across various datasets estimated from 10 batches for efficiency:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_loader, val_loader, test_loader = main()
train_accuracy = calc_accuracy_loader(train_loader,
                  model,
                  device,
                  num_batches=10
                  )
val_accuracy = calc_accuracy_loader(val_loader,
                  model,
                  device,
                  num_batches=10
                  )
test_accuracy = calc_accuracy_loader(test_loader,
                  model,
                  device,
                  num_batches=10
                  )
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Similar to calculating the training accuracy, we now compute the initial loss for each data set
# Disables gradient tracking "with torch.no_grad()" for efficiency because we are not training yet
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")
