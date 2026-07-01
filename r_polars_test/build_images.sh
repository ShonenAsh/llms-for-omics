#!/usr/bin/env bash
set -euo pipefail

# Parse --skip-base flag
SKIP_BASE=false
for arg in "$@"; do
    if [ "$arg" = "--skip-base" ]; then
        SKIP_BASE=true
        break
    fi
done

# Build the base image first (skip with --skip-base)
if ! $SKIP_BASE; then
    echo "==> Building base image..."
    docker build -f Dockerfile.base -t r-polars-bench:base .
fi

# Build exp_1 (no docs baseline)
echo "==> Building r-polars-bench:exp_1..."
docker build -f Dockerfile.experiment \
    --build-arg EXPERIMENT=exp_1 --build-arg LEVEL=none \
    -t r-polars-bench:exp_1 .

# Map experiment variants to their levels (condition names)
declare -A EXPERIMENTS=(
    [exp_2a]="2a_1_signatures 2a_2_sig_description 2a_3_one_example 2a_4_examples_io 2a_5_full"
    [exp_2b]="2b_1_no_dtypes 2b_2_no_examples"
)

for EXPERIMENT in "${!EXPERIMENTS[@]}"; do
    for LEVEL in ${EXPERIMENTS[$EXPERIMENT]}; do
        TAG="r-polars-bench:${EXPERIMENT}-${LEVEL}"
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
docker images r-polars-bench --format "  {{.Repository}}:{{.Tag}}"
