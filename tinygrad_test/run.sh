#!/usr/bin/env bash
set -euo pipefail

TASKS_DIR=/workspace/tasks
DOCS_DIR=/workspace/docs
PROMPT=/workspace/Prompt.md
SUBMISSIONS_DIR=/workspace/submissions

mkdir -p "$SUBMISSIONS_DIR"

echo "==> Generating solutions (model: ${MODEL}, experiment: ${EXPERIMENT:-?}, level: ${LEVEL:-?})"

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
        --context-dir "$SUBMISSIONS_DIR"
done

echo "==> Running tests"
pytest tests/ -v
