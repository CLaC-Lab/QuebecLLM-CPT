#!/bin/bash
#SBATCH -J qcpt-llama1b-6E-fsdp
#SBATCH -p phys
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o %x.%j.out
# #SBATCH -w virya2  # optional pin to 8-GPU host (but we only use 2 here)

set -euo pipefail

echo "=========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST = ${SLURM_NODELIST}"
echo "=========================================="

curr_dir=$(pwd)
echo "${curr_dir}"
regex="^\/home\/(\w*)\/"
user="k_ammade"
if [[ $curr_dir =~ $regex ]]; then
    user="${BASH_REMATCH[1]}"
    echo $user  
fi

########## ENV / PYTHON ##########
source "/home/${user}/miniconda3/etc/profile.d/conda.sh"
conda activate cpt-env

# UTF-8 logs (helps with accented chars)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8

which python
python -V

########## HF LOGIN ##########
HF_TOKEN="${HF_TOKEN:-}"; if [ -z "${HF_TOKEN}" ] && [ -f "${HOME}/.hf_token" ]; then HF_TOKEN="$(cat "${HOME}/.hf_token")"; fi
if [ -z "${HF_TOKEN}" ]; then echo "[ERROR] No HF token found"; exit 2; fi
set +x
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
set -x

########## PERF / NCCL / CACHES ##########
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
unset HF_HUB_ENABLE_HF_TRANSFER

# NCCL: single-node safer defaults
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# DataLoader stability & Trainer hints
export DATALOADER_NUM_WORKERS=0
export TORCH_DISTRIBUTED_DEBUG=OFF

########## SCRATCH / PATHS ##########
export SLURM_TMPDIR="home/${USER}/slurm_tmpdir/${SLURM_JOB_ID}"
mkdir -p "${SLURM_TMPDIR}"
WORKDIR="${SLURM_TMPDIR}/qcpt_run_slurm"
HF_CACHE="${SLURM_TMPDIR}/hf_cache"
mkdir -p "${WORKDIR}" "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"

echo "[INFO] Node: $(hostname)"
echo "[INFO] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv

CORPUS_SRC="${curr_dir}/data/ALL_DATA/train.txt"
CODEDIR="${curr_dir}"
OUTDIR="${curr_dir}/quebec_french_llama3.2_1b_6E_2gpu_fsdp"
mkdir -p "${OUTDIR}"
cp -v "${CORPUS_SRC}" "${WORKDIR}/train.txt"

# Preflight HF access
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download("meta-llama/Llama-3.2-1B", "config.json", token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
print("[Preflight] OK:", p)
PY

########## RUN (FSDP across 2 GPUs) ##########
cd "${CODEDIR}"
set +e

# Start conservative per-GPU batch; increase after a clean first epoch
PER_DEVICE_BS=16
GAS=16   # effective global batch ~ PER_DEVICE_BS * n_gpus * GAS

srun -u torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_PER_NODE:-2} \
  train.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --use_lora \
  --train_file "${WORKDIR}/train.txt" \
  --max_length 1024 \
  --batch_size "${PER_DEVICE_BS}" \
  --output_dir "${curr_dir}/models/quebec_french_llama3.2_1b_6E_fsdp" \
  --num_epochs 6 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps "${GAS}" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" 
STATUS=$?
set -e

echo "[INFO] Training exit status: ${STATUS}"

if [ -d "${WORKDIR}/quebec_french_llama3.2_1b_6E_fsdp" ]; then
  rsync -avh "${WORKDIR}/quebec_french_llama3.2_1b_6E_fsdp/" "${OUTDIR}/"
else
  echo "[WARN] Output dir not found; nothing to copy."
fi

exit "${STATUS}"
