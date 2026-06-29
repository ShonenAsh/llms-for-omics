#!/usr/bin/env bash
set -euo pipefail

TASKS_DIR=/workspace/tasks
DOCS_DIR=/workspace/docs
PROMPT=/workspace/Prompt.md
SUBMISSIONS_DIR=/workspace/submissions
DATA_DIR=/workspace/data

exp_tag="${EXPERIMENT}${LEVEL:+_${LEVEL}}"

echo "==> model: ${MODEL}  experiment: ${EXPERIMENT:-?}  level: ${LEVEL:-?}  runs: ${RUNS}"

# Optional args array
GENERATE_ARGS=()
[ -n "${API_BASE:-}"   ] && GENERATE_ARGS+=(--api-base   "$API_BASE")
[ -n "${MAX_TOKENS:-}" ] && GENERATE_ARGS+=(--max-tokens "$MAX_TOKENS")
[ -n "${EXTRA_BODY:-}" ] && GENERATE_ARGS+=(--extra-body "$EXTRA_BODY")

for i in $(seq -w 0 $(( RUNS - 1 ))); do
    (
        run_dir="$SUBMISSIONS_DIR/${exp_tag}/run_${i}"
        mkdir -p "$run_dir"
        echo "==== ${exp_tag} run ${i} starting ===="

        for task_file in "$TASKS_DIR"/task_*.R; do
            task_name=$(basename "$task_file")
            echo "    [${exp_tag}/run_${i}] generating $task_name"
            python generate.py \
                --task        "$task_file" \
                --docs        "$DOCS_DIR" \
                --prompt      "$PROMPT" \
                --output      "$run_dir/$task_name" \
                --model       "$MODEL" \
                --context-dir "$run_dir" \
                "${GENERATE_ARGS[@]}"
        done

        echo "    [${exp_tag}/run_${i}] running tests"
        Rscript benchmark.R \
            --submission-dir "$run_dir" \
            --test-dir       /workspace/tests \
            --results        "$run_dir/results.md" \
            --data-dir       "$DATA_DIR" || true
        echo "==== ${exp_tag} run ${i} done ===="
    ) &
done

wait
echo "==> All runs complete"
