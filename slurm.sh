#!/usr/bin/env bash
#SBATCH -J quebec-cpt-croissant_ALL_DATA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 90:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- Paths & params ---
WORKDIR="/home/k_ammade"
MODEL="croissantllm/CroissantLLMChat-v0.1"
TRAIN="$WORKDIR/CPT_scratch/data/ALL_DATA/train.txt"
OUT="$WORKDIR/CPT_scratch/quebec_croissant_chat_ALL_DATA_6EPOCHS"

# Optional: keep HF caches out of $HOME
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$WORKDIR/logs" "$OUT"

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "Start time: $(date)"
nvidia-smi || true
python3 -V || true

# Activate venv
source /home/k_ammade/CPT_scratch/venv/bin/activate

# CUDA setup
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run training with improved parameters
python "$WORKDIR/CPT_scratch/train.py" \
  --model_name "$MODEL" \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --train_file "$TRAIN" \
  --output_dir "$OUT" \
  --num_epochs 6 \
  --learning_rate 2e-6 \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --inspect_data \
  --inspect_samples 5 \
  2>&1 | tee -a "$WORKDIR/logs/train_${SLURM_JOB_ID}.log"

echo "End time: $(date)"
echo "Job completed with exit code: $?"