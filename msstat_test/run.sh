#!/usr/bin/env bash
set -euo pipefail

# Runs one doc condition's (task x k-run) generation + scoring locally against
# a running vLLM (or other OpenAI-compatible) endpoint. Adapted from
# r_polars_test/run.sh: that version picks TASKS_DIR at container-build time
# (one image per condition); since this study isn't containerized yet, we
# parameterize on CONDITION at run time instead and point TASKS_DIR straight
# at the matching docs_conditions/<CONDITION>/ directory built by
# extract_docs.R.

CONDITION="${CONDITION:?set CONDITION to one of: one_example, pitfall_note}"

TASKS_DIR="docs_conditions/${CONDITION}"
DOCS_DIR="docs"                 # unused placeholder -- docs are baked into task files
PROMPT="exp_2a/Prompt.md"
# Override per model when running multiple models (e.g. SUBMISSIONS_DIR=submissions/qwen
# ./run.sh) -- otherwise a second model run overwrites the first model's run_00/01/.../
# output, since neither this path nor the per-run task filenames include the model name.
SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-submissions}"
TEST_DIR="tests"

[ -d "$TASKS_DIR" ] || { echo "No such condition dir: $TASKS_DIR (run: Rscript extract_docs.R)"; exit 1; }

RUNS="${RUNS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-$RUNS}"
export TEST_TIMEOUT="${TEST_TIMEOUT:-120}"

echo "==> model: ${MODEL:-?}  condition: ${CONDITION}  runs: ${RUNS}  max_parallel: ${MAX_PARALLEL}"

# Optional args array
GENERATE_ARGS=()
[ -n "${API_BASE:-}"   ] && GENERATE_ARGS+=(--api-base   "$API_BASE")
[ -n "${MAX_TOKENS:-}" ] && GENERATE_ARGS+=(--max-tokens "$MAX_TOKENS")
[ -n "${EXTRA_BODY:-}" ] && GENERATE_ARGS+=(--extra-body "$EXTRA_BODY")

for i in $(seq -w 0 $(( RUNS - 1 ))); do
    (
        run_dir="$SUBMISSIONS_DIR/${CONDITION}/run_${i}"
        mkdir -p "$run_dir"
        echo "==== ${CONDITION} run ${i} starting ===="

        for task_file in "$TASKS_DIR"/task_*.R; do
            task_name=$(basename "$task_file")
            echo "    [${CONDITION}/run_${i}] generating $task_name"
            python generate.py \
                --task        "$task_file" \
                --docs        "$DOCS_DIR" \
                --prompt      "$PROMPT" \
                --output      "$run_dir/$task_name" \
                --model       "$MODEL" \
                --context-dir "$run_dir" \
                "${GENERATE_ARGS[@]}"
        done

        echo "    [${CONDITION}/run_${i}] running tests"
        Rscript benchmark.R \
            --submission-dir "$run_dir" \
            --test-dir       "$TEST_DIR" \
            --results        "$run_dir/results.md" || true
        echo "==== ${CONDITION} run ${i} done ===="
    ) &

    # Throttle: keep at most MAX_PARALLEL runs in flight.
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || wait
    done
done

wait
echo "==> All runs complete"
