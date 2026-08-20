#!/usr/bin/env bash
set -euo pipefail

# Build Apptainer/Singularity images: one base image + one experiment image
# per doc condition. Mirrors r_polars_test/build_sif.sh.
#
# Usage:
#   ./build_sif.sh              # build base image, then all 4 conditions
#   ./build_sif.sh --skip-base  # reuse an existing msstats-bench-base.sif

TEMPLATE="experiment.def"

# Parse --skip-base flag
SKIP_BASE=false
for arg in "$@"; do
    if [ "$arg" = "--skip-base" ]; then
        SKIP_BASE=true
        break
    fi
done

# Build base image (skip with --skip-base)
if ! $SKIP_BASE; then
    echo "==> Building base SIF..."
    apptainer build msstats-bench-base.sif base.def
fi

# msstat_test has one experiment (exp_2a) with 2 doc conditions ("docs +
# examples" and "docs + examples + callouts"), unlike r_polars_test's
# multiple exp_1/exp_2a/exp_2b/exp_2c experiments.
EXPERIMENT="exp_2a"
LEVELS="one_example pitfall_note"

for LEVEL in $LEVELS; do
    TAG="msstats-bench-${EXPERIMENT}-${LEVEL}"
    echo "==> Building ${TAG}.sif..."
    sed -e "s|%%EXPERIMENT%%|${EXPERIMENT}|g" \
        -e "s|%%LEVEL%%|${LEVEL}|g" \
        "$TEMPLATE" > .def_tmp && \
    apptainer build "${TAG}.sif" .def_tmp && \
    rm -f .def_tmp
done

echo "==> Done. SIF files:"
ls -lh msstats-bench-*.sif
