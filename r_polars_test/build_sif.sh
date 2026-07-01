#!/usr/bin/env bash
set -euo pipefail

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
    apptainer build r-polars-bench-base.sif base.def
fi

# Build exp_1 (no docs baseline)
echo "==> Building r-polars-bench-exp_1.sif..."
sed -e "s|%%EXPERIMENT%%|exp_1|g" \
    -e "s|%%EXPERIMENT_DIR%%|exp_1|g" \
    -e "s|%%LEVEL%%|none|g" \
    "$TEMPLATE" > .def_tmp && \
apptainer build r-polars-bench-exp_1.sif .def_tmp && \
rm -f .def_tmp

# Map experiment variants to their levels
declare -A EXPERIMENTS=(
    [exp_2a]="2a_1_signatures 2a_2_sig_description 2a_3_one_example 2a_4_examples_io 2a_5_full"
    [exp_2b]="2b_1_no_dtypes 2b_2_no_examples"
)

for EXPERIMENT in "${!EXPERIMENTS[@]}"; do
    for LEVEL in ${EXPERIMENTS[$EXPERIMENT]}; do
        TAG="r-polars-bench-${EXPERIMENT}-${LEVEL}"
        echo "==> Building ${TAG}.sif..."
        sed -e "s|%%EXPERIMENT%%|${EXPERIMENT}|g" \
            -e "s|%%EXPERIMENT_DIR%%|${EXPERIMENT}|g" \
            -e "s|%%LEVEL%%|${LEVEL}|g" \
            "$TEMPLATE" > .def_tmp && \
        apptainer build "${TAG}.sif" .def_tmp && \
        rm -f .def_tmp
    done
done

echo "==> Done. SIF files:"
ls -lh r-polars-bench-*.sif
