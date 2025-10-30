import pdb
import re


with open("the-verdict.txt", 'r', encoding='utf-8') as f1:
    raw_text = f1.read()
print(f"total number of characters: {len(raw_text)}")

# tokenized
preprocessed = re.split(r'(--|[,.:;?_!"()\]\s])', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

#convert token into token ID
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 5:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encoder(self, text):
        preprocessed = re.split(r'(--|[,.:;?_!"()\]\s])', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decoder(self, ids):
        text = " ".join(self.int_to_str[id] for id in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"
        It's the last he painted, you know," Mrs. 
        Gisburn said with pardonable pride.
        """
# text = "Hello, do you like tea?"
token_ids = tokenizer.encoder(text)
print(token_ids)
decoded_text = tokenizer.decoder(token_ids)
print(decoded_text)


