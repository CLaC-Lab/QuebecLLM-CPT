import tiktoken

def count_tokens_in_file(file_path, encoding_name="cl100k_base"):
    """Counts the number of tokens in a text file using tiktoken."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
file_name = "/home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/FreCDO_train.txt" 
token_count = count_tokens_in_file(file_name)
print(f"Number of tokens in '{file_name}': {token_count}")