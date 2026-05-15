#!/usr/bin/env bash
set -euo pipefail

# Build the base image first
echo "==> Building base image..."
docker build -f Dockerfile.base -t tinygrad-bench:base .

# Build exp_1 (no docs baseline, no levels)
echo "==> Building tinygrad-bench:exp_1..."
docker build -f Dockerfile.experiment \
    --build-arg EXPERIMENT=exp_1 --build-arg LEVEL=docs \
    -t tinygrad-bench:exp_1 .

# Map experiment variants to their levels
declare -A EXPERIMENTS=(
    [exp_2a]="level_1 level_2 level_3 level_4 level_5"
    [exp_2b]="ablation_1a_no_types ablation_1b_no_return_shapes ablation_2_no_examples ablation_3_no_special ablation_4_no_migration"
)

for EXPERIMENT in "${!EXPERIMENTS[@]}"; do
    for LEVEL in ${EXPERIMENTS[$EXPERIMENT]}; do
        TAG="tinygrad-bench:${EXPERIMENT}-${LEVEL}"
        echo "==> Building ${TAG}..."
        docker build \
            -f Dockerfile.experiment \
            --build-arg EXPERIMENT="${EXPERIMENT}" \
            --build-arg LEVEL="${LEVEL}" \
            -t "${TAG}" \
            .
    done
done

echo "==> Done. Images built:"
docker images tinygrad-bench --format "  {{.Repository}}:{{.Tag}}"
