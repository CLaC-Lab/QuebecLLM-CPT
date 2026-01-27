from transformers import AutoTokenizer
import json

# Use the exact checkpoint you train/infer with:
# "croissantllm/CroissantLLMBase" or "croissantllm/CroissantLLMChat-v0.1"
tok = AutoTokenizer.from_pretrained("croissantllm/CroissantLLMChat-v0.1", use_fast=True)

def croissant_token_count(path: str) -> int:
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Count raw tokens; don't add BOS/EOS/etc.
            total += len(tok.encode(line, add_special_tokens=False))
    return total

def count_tokens(path):
    with open(path, "r") as f_handle:
        dataset = [json.loads(line) for line in f_handle]
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
    total_tokens = 0
    num_tokens = 0
    for item in dataset:
        system = item.find(lambda x: x['role'] == "system")
        system_tokens = len(tok.encode(system['content'], add_special_tokens=False))
        user = item.find(lambda x: x['role'] == "user")
        user_tokens = len(tok.encode(user['content'], add_special_tokens=False))
        assistant = item.find(lambda x: x['role'] == "assistant")
        assistant_tokens = len(tok.encode(len(tok.encode(assistant['content'], add_special_tokens=False)), add_special_tokens=False))
        #
        total_tokens += system_tokens + user_tokens + assistant_tokens
        num_tokens += user_tokens + assistant_tokens

    print(f"{path} has {total_tokens} tokens, of which {num_tokens} are unique")



print(croissant_token_count("/home/k_ammade/CPT_scratch/data/ALL_DATA_FRECDO/train.txt"))
