#!/usr/bin/env python

import random
import json
import argparse
from transformers import AutoTokenizer

def load_files(files):
    data_obj = []
    for file_name in files:
        with open(file_name, "r") as f_handle:
            data = [json.loads(line) for line in f_handle]
            data_obj.append(data)
    return data_obj

def count_tokens(datasets):
    
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
    for i, ds in enumerate(datasets):
        ds_name = files[i]
        total_tokens = 0
        num_tokens = 0
        for text in ds:
            line = text['text']
            all_tokens = len(tok.encode(line, add_special_tokens=False))
            line = line.replace("<|begin_of_text|>","")
            turns = line.split("<|eot_id|>")
            #system = turns[0].split('<|end_header_id|>')[1]
            user = turns[1].split('<|end_header_id|>')[1] 
            user_tokens = len(tok.encode(user, add_special_tokens=False))
            response = turns[2].split('<|end_header_id|>')[1] if len(turns[2]) >= 0 else ""
            response_tokens = len(tok.encode(response, add_special_tokens=False))
            #
            total_tokens += all_tokens
            num_tokens += user_tokens + response_tokens#len(user_tokens) + len(response_tokens)
            a = 1 

        print(f"{ds_name} has {total_tokens} tokens, of which {num_tokens} are unique")

def export(datasets):
    all_lines = []
    for i, ds in enumerate(datasets):
        lines = [text['text'] for text in ds]
        print(f"len {len(lines)}")
        all_lines.extend(lines)
    random.shuffle(all_lines)
    with open("train.txt", "w") as f_handle:
        for line in all_lines:
            line = line.replace("\n", "\\n")
            f_handle.write(line + "\n")



            

def main():
    parser = argparse.ArgumentParser(description="Continual Pretraining for LLaMA")
    parser.add_argument("--count_tokens", type=bool, default=True, required=True)
    parser.add_argument("--train_files", type=str, nargs='+' required=True, help="Path to training corpus")

    args = parser.parse_args()

    files = [args.train_files] if type(args.train_files) is str else args.train_files
    data_obj = load_files()
    #count_tokens(data_obj)
    export(data_obj)
    a = 1
    

if __name__ == "__main__":
    main()