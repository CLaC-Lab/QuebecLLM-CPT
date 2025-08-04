#!/bin/bash

# Training with TinyLlama - compatible with your existing tokenized data!
MODEL_NAME="meta-llama/Llama-3.2-3B"  # 1.1B params, uses Llama tokenizer
# Alternative: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" (base model)

DATA_PATH="../data/tokenized_data_ids/firas_json"
OUTPUT_DIR="../models/Llama-3.2-3B"
MAX_LENGTH=128  

# Can use larger batch size with smaller model
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-5
NUM_EPOCHS=0.01
WARMUP_STEPS=100

echo "Starting training with Llama..."
echo "Model: $MODEL_NAME"
echo "This model uses the same tokenizer as Llama, so your data is compatible!"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_strategy "steps" \
    --logging_dir "./logs" \
    --report_to "none" \
    --gradient_checkpointing \
    --fp16 \
    --dataloader_num_workers 2

# Optional flags:
# --bf16  # If your GPU supports bfloat16 (instead of fp16)
# --push_to_hub  # If you want to upload to HuggingFace
# --run_name "tinyllama_test"  # For experiment tracking