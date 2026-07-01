#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --job-name=ash_vllm_server
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a.magadum/ondemand/work/job_logs/ash_vllm_server_%j.out
#SBATCH --error=/home/a.magadum/ondemand/work/job_logs/ash_vllm_server_%j.err
#SBATCH --mail-type=ALL

#modules
module load anaconda3/2024.06
module load cuda/12.8.0

echo "modules loaded"

# Comment out when downloading model for the first time
# export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="/scratch/$USER/vllm_cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="/scratch/$USER/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source /home/$USER/ondemand/work/.venv/bin/activate
#python main.py

echo "================== running vllm server =================="

# Model (special args list below)
MODEL="google/gemma-4-26B-A4B-it"

# env vars passed to container, max tokens shhould be 16000 for others
export MAX_TOKENS=32000
# only for Qwen3.6-27B
# export EXTRA_BODY='{"chat_template_kwargs": {"thinking_budget": 8192}}'

case "$MODEL" in
  "Qwen/Qwen3.6-27B")
    # Thinking mode enabled by default, reasoning-parser handles <think> blocks
    EXTRA_ARGS=(--reasoning-parser deepseek_r1 --max-model-len 65536)
    ;;
  "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    # MoE 30B total / 3B active, no thinking mode on this model
    EXTRA_ARGS=(--max-model-len 65536)
    ;;
  "google/gemma-4-26B-A4B-it")
    # MoE 26B total / 4B active, needs trust-remote-code
    EXTRA_ARGS=(--max-model-len 65536 --trust-remote-code)
    ;;
  "Qwen/Qwen3.5-9B")
    EXTRA_ARGS=(--reasoning-parser qwen3 --max-model-len 65536)
    ;;
esac

echo "================== running vllm server: $MODEL =================="

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8089 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  "${EXTRA_ARGS[@]}"

echo "completed"

# capture exit code
EXIT_CODE=$?

# stop apptainer instances -a for all instances of $USER (this doesn't work on HPC)
# apptainer instance stop -a


echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE