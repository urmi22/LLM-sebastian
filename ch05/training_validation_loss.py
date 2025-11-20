


import pdb
import torch
import torch.nn as nn
import tiktoken

from ch02.dataloader import create_dataloader_v1
from ch05.GPT_CONFIG_124M import GPT_2_small as cfg
from ch04.GPT_model import GPTModel




# cross entropy loss of a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# computes the loss over all the batches sampled by a given data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iteratives over all batches if no fixed num_batches is specified
        num_batches = len(data_loader)
    else:
        # Reduces the number of batches to match the total number of batches in the data loader if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    
    for i , (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Sums loss for each batch
            total_loss += loss.item()
        else:
            break
        
    # Averages the loss over all batches
    avg_loss = total_loss / num_batches
    return avg_loss




def train_val_loader(train_data, val_data):

    '''
    Using the train_data and val_data subsets, 
    we can now create the respective data loader reusing the create_dataloader_v1 code from chapter 2

    '''

    train_loader = create_dataloader_v1(train_data, 
                                        batch_size=2, 
                                        max_length=cfg["context_length"], 
                                        stride=cfg["context_length"],
                                        drop_last=True, 
                                        shuffle=True,
                                        num_workers=0
                                        )
    
    val_loader = create_dataloader_v1(val_data,
                                      batch_size=2,
                                      max_length=cfg["context_length"],
                                      stride=cfg["context_length"],
                                      drop_last=False,
                                      shuffle=False,
                                      num_workers=0
                                      )
    
    return train_loader, val_loader


def main():

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_ratio = 0.9

    # loads the “The Verdict” short story:
    file_path = "./ch05/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f0:
        text_data = f0.read()

    # check the number of characters and tokens in the dataset
    total_characters = len(text_data)
    print(f"Total characters: {total_characters}")
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total tokens: {total_tokens}")

    '''
    To implement the data splitting and loading, 
    we first define a train_ratio to use 90% of the data for training and the remaining 10% as validation data for model evaluation during training

    '''

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader, val_loader = train_val_loader(train_data, val_data)

    
    
    print(f"Train Loader\n-----------")
    train_tokens = 0
    for x, y in train_loader:
        print(x.shape, y.shape)
        train_tokens += x.numel()
    print(f"Training tokens: {train_tokens}")

    val_tokens = 0
    print(f"validation Loader\n-------------")
    for x, y in val_loader:
        print(x.shape, y.shape)
        val_tokens += x.numel()
    print(f"Validation tokens: {val_tokens}")
    print(f"All tokens: {train_tokens + val_tokens}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"Training loss: {train_loss}")
    print(f"validation loss : {val_loss}")

    return train_loader, val_loader









if __name__=="__main__":
    main()