#!/usr/bin/env bash
#SBATCH -J replay-qcpt-83M
# #SBATCH -p phys
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 30:00:00
#SBATCH -o %x-%j.out  

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- Paths & params (edit if needed) ---
WORKDIR="$HOME/CPT_scratch"

MODEL="croissantllm/CroissantLLMChat-v0.1"
TRAIN="/home/k_ammade/CPT_scratch/data/83M_data/train.txt"
VAL="/home/k_ammade/CPT_scratch/data/83M_data/val.txt"
REP="/home/k_ammade/CPT_scratch/data/croissant.jsonl"
OUT="/home/k_ammade/CPT_scratch/quebec_croissant_chat_83M_replay"

# Optional: keep HF caches out of $HOME if desired
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$WORKDIR/logs" "$OUT"

echo "== Job $SLURM_JOB_ID on $(hostname) =="
nvidia-smi || true
python3 -V || true

# Activate venv
source /home/k_ammade/CPT_scratch/venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH  
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python "$WORKDIR/train_replay.py" \
  --model_name "$MODEL" \
  --use_lora \
  --train_file "$TRAIN" \
  --val_file "$VAL" \
  --replay_file "$REP" \
  --output_dir "$OUT" \
2>&1 | tee -a "$WORKDIR/logs/train_${SLURM_JOB_ID}.log"
