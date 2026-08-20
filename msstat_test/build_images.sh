#!/usr/bin/env bash
set -euo pipefail

# Build Docker images: one base image + one experiment image per doc
# condition. Mirrors r_polars_test/build_images.sh.
#
# Usage:
#   ./build_images.sh              # build base image, then all 4 conditions
#   ./build_images.sh --skip-base  # reuse an existing msstats-bench:base

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
    docker build -f Dockerfile.base -t msstats-bench:base .
fi

# msstat_test has one experiment (exp_2a) with 2 doc conditions ("docs +
# examples" and "docs + examples + callouts"), unlike r_polars_test's
# multiple exp_1/exp_2a/exp_2b/exp_2c experiments.
EXPERIMENT="exp_2a"
LEVELS="one_example pitfall_note"

for LEVEL in $LEVELS; do
    TAG="msstats-bench:${EXPERIMENT}-${LEVEL}"
    echo "==> Building ${TAG}..."
    docker build \
        -f Dockerfile.experiment \
        --build-arg EXPERIMENT="${EXPERIMENT}" \
        --build-arg LEVEL="${LEVEL}" \
        -t "${TAG}" \
        .
done

echo "==> Done. Images built:"
docker images msstats-bench --format "  {{.Repository}}:{{.Tag}}"
