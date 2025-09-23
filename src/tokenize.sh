#!/bin/bash

#MODEL_NAME="meta-llama/Llama-3.2-3B"
MODEL_NAME="croissantllm/CroissantLLMBase"
model=$(cut -d'/' -f2 <<< "$MODEL_NAME")
MODEL_DIR="../data/tokenized"
JSON_DATA_PATH="../models/$model/"
MAX_LENGTH=128  

mkdir $JSON_DATA_PATH
cp -r "../data/firas_json" "$JSON_DATA_PATH/json"

python tokenize_text.py \
    --model_name $MODEL_DIR \
    --tokenizer_path $MODEL_NAME \
    --data_path $JSON_DATA_PATH \
    --num_files 30 \
    --text_key text