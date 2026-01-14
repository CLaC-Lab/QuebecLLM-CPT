from transformers import AutoTokenizer

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

print(croissant_token_count("/home/k_ammade/CPT_scratch/data/ALL_DATA_FRECDO/train.txt"))
