#!/usr/bin/env bash
set -euo pipefail

TASKS_DIR=/workspace/tasks
DOCS_DIR=/workspace/docs
PROMPT=/workspace/Prompt.md
SUBMISSIONS_DIR=/workspace/submissions

exp_tag="${EXPERIMENT}${LEVEL:+_${LEVEL}}"

echo "==> model: ${MODEL}  experiment: ${EXPERIMENT:-?}  level: ${LEVEL:-?}  runs: ${RUNS}"

API_BASE_FLAG=""
if [ -n "${API_BASE:-}" ]; then
    API_BASE_FLAG="--api-base $API_BASE"
fi

for i in $(seq -w 0 $(( RUNS - 1 ))); do
    (
        run_dir="$SUBMISSIONS_DIR/${exp_tag}/run_${i}"
        mkdir -p "$run_dir"
        echo "==== ${exp_tag} run ${i} starting ===="

        for task_file in "$TASKS_DIR"/task_*.py; do
            task_name=$(basename "$task_file")
            echo "    [${exp_tag}/run_${i}] generating $task_name"
            python generate.py \
                --task        "$task_file" \
                --docs        "$DOCS_DIR" \
                --prompt      "$PROMPT" \
                --output      "$run_dir/$task_name" \
                --model       "$MODEL" \
                --context-dir "$run_dir" \
                $API_BASE_FLAG
        done

        echo "    [${exp_tag}/run_${i}] running tests"
        PYTHONPATH="$run_dir:/workspace/tasks" pytest tests/ -v > "$run_dir/results.md" 2>&1 || true
        echo "==== ${exp_tag} run ${i} done ===="
    ) &
done

wait
echo "==> All runs complete"