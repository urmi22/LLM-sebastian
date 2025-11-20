


import pdb
import tiktoken










def main():

    tokenizer = tiktoken.get_encoding("gpt2")
    # loads the “The Verdict” short story:
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f0:
        text_data = f0.read()

    # check the number of characters and tokens in the dataset
    total_characters = len(text_data)
    print(f"Total characters: {total_characters}")
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total tokens: {total_tokens}")








if __name__=="__main__":
    main()