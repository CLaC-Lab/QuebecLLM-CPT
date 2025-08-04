#!/bin/bash

MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR="data/tokenized"
RAW_DATA_PATH="../data/firas"
JSON_DATA_PATH="../data/firas_json"
MAX_LENGTH=128  

echo $JSON_DATA_PATH

if [ ! -d "$JSON_DATA_PATH" ]; then
    python convert_to_json.py \
        --input_dir $RAW_DATA_PATH \
        --output_dir $JSON_DATA_PATH \
        --chunk_size 2000
fi
    

python tokenize_text.py \
    --model_name $MODEL_DIR \
    --tokenizer_path $MODEL_NAME \
    --data_path $JSON_DATA_PATH \
    --num_files 30 \
    --text_key text