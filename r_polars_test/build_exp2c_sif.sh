#!/usr/bin/env bash
set -euo pipefail

# Build only the exp_2c "examples-only" image.
#
# exp_2c isolates the impact of examples: the doc block contains just the
# method-name header (# --- <name> ---) plus example code, with no signatures,
# descriptions, arguments, return values, or captured I/O output. See the
# "2c_only_examples" condition in extract_docs.R.
#
# Uses the shared base.def (via r-polars-bench-base.sif) and experiment.def
# template, mirroring build_sif.sh. Produces: exp_2c_only_examples.sif
#
# Usage:
#   ./build_exp2c_sif.sh              # build base image, then exp_2c
#   ./build_exp2c_sif.sh --skip-base  # reuse existing r-polars-bench-base.sif

TEMPLATE="experiment.def"
BASE_SIF="r-polars-bench-base.sif"
EXPERIMENT="exp_2c"
LEVEL="2c_only_examples"
OUT_SIF="exp_2c_only_examples.sif"

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
    echo "==> Building base SIF ($BASE_SIF)..."
    apptainer build "$BASE_SIF" base.def
elif [ ! -f "$BASE_SIF" ]; then
    echo "ERROR: --skip-base set but $BASE_SIF not found. Run without --skip-base first." >&2
    exit 1
fi

# Build exp_2c only
echo "==> Building ${OUT_SIF}..."
sed -e "s|%%EXPERIMENT%%|${EXPERIMENT}|g" \
    -e "s|%%EXPERIMENT_DIR%%|${EXPERIMENT}|g" \
    -e "s|%%LEVEL%%|${LEVEL}|g" \
    "$TEMPLATE" > .def_tmp && \
apptainer build "$OUT_SIF" .def_tmp && \
rm -f .def_tmp

echo "==> Done:"
ls -lh "$OUT_SIF"
