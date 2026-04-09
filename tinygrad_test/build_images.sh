#!/usr/bin/env bash
set -euo pipefail

# Build the base image first
echo "==> Building base image..."
docker build -f Dockerfile.base -t tinygrad-bench:base .

# Map each experiment to its levels
declare -A EXPERIMENTS=(
    [exp_2a]="level_1 level_2 level_3 level_4 level_5"
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
