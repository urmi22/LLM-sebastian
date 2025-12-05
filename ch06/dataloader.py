import pdb
import tiktoken
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader


'''
The SpamDataset class loads data from the CSV files we created earlier, 
tokenizes the text using the GPT-2 tokenizer from tiktoken, 
and allows us to pad or truncate the sequences to a uniform length determined by either the longest sequence or a predefined maximum length. 
This ensures each input tensor is of the same size, which is necessary to create the batches in the training data loader we implement next.

'''

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pretokenized texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncates sequences if they are longer than max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        # Pads sequences to the longest sequence
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length

        return max_length



    



def main():

    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_dataset = SpamDataset(csv_file="./ch06/input_dataset/train.csv", 
                                max_length=None, 
                                tokenizer=tokenizer
                                )
    validation_dataset = SpamDataset(csv_file="./ch06/input_dataset/validation.csv", 
                                     max_length=train_dataset.max_length,
                                     tokenizer=tokenizer
                                     )
    test_dataset = SpamDataset(csv_file="./ch06/input_dataset/test.csv",
                               max_length=train_dataset.max_length,
                               tokenizer=tokenizer
                               )
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True
                              )
    val_loader = DataLoader(dataset=validation_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False
                            )
    
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False
                            )
    
    # to ensure dataloader is working properly
    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    # to get an idea of the dataset size, let’s print the total number of batches in each dataset
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
    
    
    return train_loader, val_loader, test_loader

    
    





if __name__=="__main__":
    train_loader, val_loader, test_loader = main()