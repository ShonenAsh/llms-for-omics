#!/usr/bin/env bash
set -euo pipefail

# Build base
echo "==> Building base image..."
apptainer build tinygrad-bench_base.sif base.def

# Build exp_1
echo "==> Building tinygrad-bench_exp_1..."
apptainer build \
    --build-arg EXPERIMENT=exp_1 \
    --build-arg LEVEL=docs \
    tinygrad-bench_exp_1.sif \
    experiment.def

# Build experiment variants
declare -A EXPERIMENTS=(
    [exp_2a]="level_1 level_2 level_3 level_4 level_5"
    [exp_2b]="ablation_1a_no_types ablation_1b_no_return_shapes ablation_2_no_examples ablation_3_no_special ablation_4_no_migration"
)

for EXPERIMENT in "${!EXPERIMENTS[@]}"; do
    for LEVEL in ${EXPERIMENTS[$EXPERIMENT]}; do
        TAG="tinygrad-bench_${EXPERIMENT}_${LEVEL}"
        echo "==> Building ${TAG}..."
        apptainer build \
            --build-arg EXPERIMENT="${EXPERIMENT}" \
            --build-arg LEVEL="${LEVEL}" \
            "${TAG}.sif" \
            experiment.def
    done
done

echo "==> Done. Images built:"
ls -lh tinygrad-bench_*.sif