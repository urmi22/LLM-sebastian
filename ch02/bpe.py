import pdb

from importlib.metadata import version
import tiktoken

print("tiktoken version", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
# text = ( "Hello, do you like tea? <|endoftext|> In the sunlit terraces" "of someunknownPlace." )
text = "AKwirw ier"
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(token_ids)
original_text = tokenizer.decode(token_ids)
print(original_text)