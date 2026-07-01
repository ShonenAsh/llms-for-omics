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

# env vars passed to container, max tokens should be 16000 for others
export MAX_TOKENS=32000

# Model-Specific config. 
# Developer note: Please use model author's recommended settings
# extra_args  : serve-time engine flags
# gen_cfg     : sampling params  -> --override-generation-config
# chat_kwargs : chat-template kwargs (e.g. thinking_budget) -> --default-chat-template-kwargs

chat_kwargs=''   # default: none
case "$MODEL" in
  "Qwen/Qwen3.6-27B")
    # thinking on; coding profile; cap reasoning at 8192 tokens
    extra_args=(--reasoning-parser deepseek_r1 --max-model-len 65536)
    gen_cfg='{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.0}'
    chat_kwargs='{"thinking_budget": 8192}'
    ;;
  "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    extra_args=(--max-model-len 65536)
    gen_cfg='{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}'
    ;;
  "google/gemma-4-26B-A4B-it")
    extra_args=(--max-model-len 65536 --trust-remote-code)
    gen_cfg='{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}'
    ;;
  "Qwen/Qwen3.5-9B")
    # coding profile; presence_penalty bumped to 1.0 since 9B loops
    extra_args=(--reasoning-parser qwen3 --max-model-len 65536)
    gen_cfg='{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":1.0}'
    ;;
esac

echo "================== running vllm server: $MODEL =================="

# assemble serve args; only add chat-template kwargs when set
serve_args=(
  --host 0.0.0.0
  --port 8089
  --dtype bfloat16
  --gpu-memory-utilization 0.90
  --generation-config auto
  --override-generation-config "$gen_cfg"
)
if [[ -n "$chat_kwargs" ]]; then
  serve_args+=(--default-chat-template-kwargs "$chat_kwargs")
fi
serve_args+=("${extra_args[@]}")

vllm serve "$MODEL" "${serve_args[@]}"


echo "completed"

# capture exit code
EXIT_CODE=$?

# stop apptainer instances -a for all instances of $USER (this doesn't work on HPC)
# apptainer instance stop -a


echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE