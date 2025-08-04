#!/usr/bin/env python'
import os
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_abs_path(model_name):
    gh_repo = "QuebecLLM-CPT"
    cwd = os.getcwd()
    if cwd.endswith(gh_repo):
        return f"{cwd}{model_name}"
    elif gh_repo in cwd:
        base_dir = f"{cwd.split(gh_repo)[0]}/{gh_repo}{model_name}"
    else:
        return model_name

def generate(model_args):
    model_dir = get_abs_path(model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, #model_args.model_name_or_path,
        #attn_implementation="flash_attention_2" if model_args.flash_attention else None,
        #cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, #model_args.model_name_or_path,
        #cache_dir=training_args.cache_dir,
        model_max_length=2048,#training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )

    model = model.to("cuda")

    input = "Répondez à la question suivante: La racine carrée de 4 est"
    inputs = tokenizer.encode(
        input,
        return_tensors = "pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(input_ids = inputs, pad_token_id=tokenizer.eos_token_id, streamer = text_streamer, max_new_tokens = 128,
                    early_stopping=True, repetition_penalty=2.0,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


def main():
    parser = argparse.ArgumentParser(description="Tokenize text data or decode tokenized data")

    # Tokenize mode arguments
    parser.add_argument("--model_name", type=str,
                        help="Name of the model (required for tokenize mode)")
    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()
