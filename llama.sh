#!/usr/bin/env bash
#SBATCH -J quebec-cpt-llama3-3b-fsdp-4mig
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2    
#SBATCH -t 120:00:00
#SBATCH -o %x-%j.out
## If needed, pin a known A100 node:
## #SBATCH --nodelist=virya5

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- CUDA hygiene ---
module purge || true
unset CUDA_HOME CUDA_PATH TRANSFORMERS_CACHE
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vi '/usr/local/cuda' | paste -sd: -)"
fi
if command -v nvidia-modprobe >/dev/null 2>&1; then nvidia-modprobe -u -c=0 || true; fi

# --- Paths & params ---
WORKDIR="/home/k_ammade"
MODEL="meta-llama/Llama-3.2-3B"
TRAIN="$WORKDIR/CPT_scratch/data/ALL_DATA/train.txt"
OUT="$WORKDIR/CPT_scratch/quebec_llama3_3b_ALL_DATA"

# HF caches
export HF_HOME="$WORKDIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$WORKDIR/logs" "$OUT"

echo "== Job $SLURM_JOB_ID on $(hostname) =="
echo "Start time: $(date)"
nvidia-smi || true
python3 -V || true

# Activate venv
source "$WORKDIR/CPT_scratch/venv/bin/activate"

# --- Hugging Face auth (trim + non-interactive) ---
TOK_FILE="$WORKDIR/.hf_token"
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "$TOK_FILE" ]]; then
    export HF_TOKEN="$(tr -d ' \t\r\n' < "$TOK_FILE")"
  else
    echo "ERROR: No HF token. Set \$HF_TOKEN or create $TOK_FILE"; exit 1
  fi
fi
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

python - <<'PY'
import os, sys
from huggingface_hub import login, whoami, HfApi
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    print("HF OK:", whoami())
    info = HfApi(token=os.environ["HF_TOKEN"]).model_info("meta-llama/Meta-Llama-3-8B-Instruct")
    print("Model access OK. Private:", info.private, "| Gated:", info.gated)
except Exception as e:
    print("HF auth/model check failed:", e, file=sys.stderr); sys.exit(1)
PY

# --- Ensure TWO GPUs visible (before torchrun) ---
python - <<'PY'
import torch, sys
print("torch:", torch.__version__, "| torch.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available(), "| count:", torch.cuda.device_count())
if torch.cuda.device_count() < 2:
    print("ERROR: Need 2 GPUs (MIG) but see", torch.cuda.device_count(), file=sys.stderr)
    sys.exit(2)
for i in range(torch.cuda.device_count()):
    print(f"GPU[{i}] ->", torch.cuda.get_device_name(i))
PY

# --- NCCL/MIG safety (MIG has no NVLink P2P) ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Explicitly set your 4 MIG devices
export OMP_NUM_THREADS=4  # Limit CPU threads to avoid system OOM

# --- Common training args for FSDP run ---
ARGS=(
  --model_name "$MODEL"
  --train_file "$TRAIN"
  --output_dir "$OUT"
  --num_epochs 3
  --learning_rate 2e-6
  --batch_size 16
  --gradient_accumulation_steps 16
  --max_length 1024
  --use_lora
  --fsdp_enable
  --fsdp_sharding full_shard
  --fsdp_min_num_params 1e8
  --fsdp_wrap_cls LlamaDecoderLayer
)

torchrun --nproc_per_node=2 \
    --master_port 29500 \
    train_llama.py \
    --train_file /home/k_ammade/CPT_scratch/data/ALL_DATA/train.txt \
    --model_name meta-llama/Llama-3.2-3B \
    --use_lora \
    --lora_r 4 \
    --lora_alpha 8 \
    --batch_size 8 \
    --max_length 256 \
    --gradient_accumulation_steps 32 \
    --output_dir ./quebec_llama3_3b_ALL_DATA

echo "End time: $(date)"
echo "Job completed with exit code: $?"
