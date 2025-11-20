

import pdb
import tiktoken
import torch
import torch.nn as nn
import ch05.training_validation_loss as tvl

from ch05.GPT_CONFIG_124M import GPT_2_small as cfg
from ch04.GPT_model import GPTModel
from ch05.training_validation_loss import calc_loss_batch, calc_loss_loader
from ch05.text_generation import text_to_token_ids, token_ids_to_text
from ch04.generate_text import generate_text_simple
from ch05.training_validation_loss import train_val_loader




def generate_and_print_sample(start_context, tokenizer, model, device):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()



def evaluate_model(train_loader, val_loader, model, device, eval_iter):
    
    # Dropout is disabled during evaluation for stable, reproducible results.
    model.eval()

    # Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss



def train_model_simple(train_loader, val_loader, model, device, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    # Initializes lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Starts the main training loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:

            #Resets loss gradients from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Calculates loss gradients
            loss.backward()

            # Updates model weights using loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(train_loader, val_loader, model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (step {global_step:06d}): " f"Train loss: {train_loss:.3f}\t" f"Validation loss: {val_loss:.3f}")

        print("\n")
        # Prints a sample text after each epoch
        generate_and_print_sample(start_context, tokenizer, model, device)
        print("\n")

    return train_losses, val_losses, track_tokens_seen




def main():

    torch.manual_seed(123)

    file_path = "./ch05/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f0:
        text_data = f0.read()

    train_ratio = 0.9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader, val_loader = train_val_loader(train_data, val_data)

    model.to(device)

    train_losses, val_losses, track_tokens_seen = train_model_simple(train_loader, 
                                                                     val_loader, 
                                                                     model, 
                                                                     device, 
                                                                     optimizer, 
                                                                     num_epochs, 
                                                                     eval_freq=5, 
                                                                     eval_iter=5, 
                                                                     start_context="Every effort moves you",
                                                                     tokenizer=tokenizer
                                                                     )






if __name__=="__main__":
    main()