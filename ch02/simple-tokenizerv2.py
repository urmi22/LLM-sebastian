# handle unknown words (words are not present in the vocab) and end-of-text (concatenate multiple documents)

import pdb
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
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_tokens)}
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encoder(self, text):
        preprocessed = re.split(r'(--|[,.:;?_!"()\]\s])', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decoder(self, ids):
        text = " ".join(self.int_to_str[id] for id in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
# pdb.set_trace()
token_ids = tokenizer.encoder(text)
print(token_ids)
decoded_text = tokenizer.decoder(token_ids)
print(decoded_text)


