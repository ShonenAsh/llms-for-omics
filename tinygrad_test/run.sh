#!/usr/bin/env bash
set -euo pipefail

TASKS_DIR=/workspace/tasks
DOCS_DIR=/workspace/docs
PROMPT=/workspace/Prompt.md
SUBMISSIONS_DIR=/workspace/submissions

mkdir -p "$SUBMISSIONS_DIR"

echo "==> Generating solutions (model: ${MODEL}, experiment: ${EXPERIMENT:-?}, level: ${LEVEL:-?})"

API_BASE_FLAG=""
if [ -n "${API_BASE:-}" ]; then
    API_BASE_FLAG="--api-base $API_BASE"
fi

for task_file in "$TASKS_DIR"/task_*.py; do
    task_name=$(basename "$task_file")
    output="$SUBMISSIONS_DIR/$task_name"
    echo "    $task_name"
    python generate.py \
        --task        "$task_file" \
        --docs        "$DOCS_DIR" \
        --prompt      "$PROMPT" \
        --output      "$output" \
        --model       "$MODEL" \
        --context-dir "$SUBMISSIONS_DIR" \
        $API_BASE_FLAG
done

echo "==> Running tests"
RESULTS="$SUBMISSIONS_DIR/results.md"
pytest tests/ -v | tee "$RESULTS"
echo "==> Results saved to $RESULTS"
