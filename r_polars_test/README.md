# R-Polars Tasks for Evaluating Documentation

A benchmark that measures how the **amount and kind of API documentation** given to an
LLM affects its ability to write correct [`polars`](https://pola-rs.github.io/r-polars/)
code in R.

Each benchmark task is a stub file of `polars` function signatures with the bodies left
empty. A build-time pipeline extracts documentation from the installed `polars` package
and injects a configurable slice of it (signatures only => full docs with examples) into
each task as R comments. An LLM then fills in the stubs, and a `testthat` suite scores
the result. Comparing scores across documentation "conditions" isolates the effect of
documentation on model performance.

## Tasks

The `tasks/` directory holds five task files, each a set of related `polars` functions
to implement. Every file declares its API surface in a single `@requires` line (the
source of truth for doc extraction) and marks where docs are injected with `<<DOCS>>`.
The 31 functions span three capability tiers:  basic ops, function chains, and
integrated multi-source workflows.

- **`task_01_basic.R`** — Basic DataFrame operations: create from R vectors, `select`,
  `filter`, `rename`, `cast`, `drop`, plus inner and left `join`. Single-operation
  fundamentals on in-memory data.
- **`task_02_chains.R`** — Chained operations: drop high-null columns by threshold,
  join taxi trips to their pickup borough, conditional string relabeling
  (`when`/`lit`), wide→long `unpivot`, and windowed `rank` with `over`.
- **`task_03_groupby.R`** — GroupBy and aggregation: trips per borough, average fare by
  passenger count, four stats in one `agg`, top-k zones by revenue, fare quantile per
  borough, and payment-type share of total.
- **`task_04_strings_dates.R`** — String and datetime operations: extract pickup hour,
  trip duration in minutes, weekend-only filter, hourly trip counts, airport-zone flag
  via substring match, split zone hierarchy on `/`, and ISO date formatting.
- **`task_05_lazy.R`** — LazyFrame pipelines: filter-and-`sink_parquet` without
  materializing, lazy join→group→sum revenue per borough, lazy pickups-per-hour,
  schema inspection via `collect_schema()`, and top-k fares with column pruning.

Tasks 2-5 operate on the NYC Yellow Taxi dataset in `data/` (see below); task 1 uses
small in-memory vectors.

## Documentation conditions

`extract_docs.R` composes documentation at a configurable **detail level** and writes
one variant of every task file into `docs_conditions/<condition>/`. Conditions form two
experiments plus controls, all driven by one spec table:

**2a: compounding ladder** (each level adds one section):
| Condition | Contents |
|---|---|
| `2a_1_signatures` | function signatures only |
| `2a_2_sig_description` | + description |
| `2a_3_one_example` | + one example |
| `2a_4_examples_io` | + all examples with captured output |
| `2a_5_full` | all five Rd sections (usage, description, arguments, value, examples) |

**2b: full-minus-one ablations** (start from full, remove one thing):
| Condition | Contents |
|---|---|
| `2b_1_no_dtypes` | full docs with dtype/`shape:` lines stripped |
| `2b_2_no_examples` | full docs without examples or output |

**Controls:**
| Condition | Contents |
|---|---|
| `none` | no docs — `<<DOCS>>` removed entirely (baseline) |
| `2c_only_examples` | method-name header + example code only (no signatures/descriptions/args/values) |

## Repository layout

### Task & test sources
- `tasks/task_*.R` — task stubs with `@requires` and `<<DOCS>>` markers (inputs to the pipeline).
- `tests/test_*.R` — `testthat` suites, one per task; source implementations from the `SUBMISSION_DIR` env var.
- `data/` — NYC Yellow Taxi fixtures: `yellow_tripdata_2024-01.parquet` (~50 MB trips) and `taxi_zone_lookup.csv` (LocationID→Borough/Zone).

### Pipeline
- `extract_docs.R` — main pipeline: reads the installed `polars` Rd database, resolves each `@requires` token to its `dataframe__group_by`-style alias, composes the doc block for a condition, and injects it at `<<DOCS>>`. Run `Rscript extract_docs.R` for all conditions or `Rscript extract_docs.R <condition>` for one.
- `docs_conditions/<condition>/` — generated task files, one directory per condition (build output).
- `generate.py` — LLM generation harness (litellm/instructor): sends a prompt + doc-augmented task to the model and writes the completed file to a submission directory.
- `benchmark.R` — `testthat` runner. Executes each test file in its own `callr` subprocess with a per-file wall-clock timeout (`--timeout`/`TEST_TIMEOUT`, default 120s) and a capped polars thread pool (`--polars-threads`/`POLARS_MAX_THREADS`) so hung or looping submissions fail cleanly and parallel runs don't oversubscribe the node. Emits `results.md`.

### Experiments
- `exp_1/`, `exp_2a/`, `exp_2b/`, `exp_2c/` — one directory per experiment, each with a `Prompt.md` (system prompt for the LLM). `exp_1` is the no-docs baseline; `exp_2a`–`exp_2c` map to the conditions above.
- `run.sh` — container entrypoint: loops `RUNS` times, generates a solution per task with `generate.py`, then scores it with `benchmark.R`. Reads `MODEL`, `RUNS`, `MAX_PARALLEL`, and optional `API_BASE`/`MAX_TOKENS`/`EXTRA_BODY`.

### Container images
- `base.def` / `Dockerfile.base` — base image: Python 3.13 + R + `polars` (from GitHub, Rust pinned to 1.96.0) + litellm. `polars` is not on CRAN, so it is built from source.
- `experiment.def` / `Dockerfile.experiment` — experiment overlay: copies tasks/tests/data/scripts, runs `extract_docs.R <LEVEL>`, and installs the chosen `Prompt.md`. Templated by `%%EXPERIMENT%%` / `%%LEVEL%%`.
- `build_sif.sh` — build all Apptainer/Singularity images (base + every experiment×level); `--skip-base` reuses an existing base.
- `build_exp2c_sif.sh` — build only the `2c_only_examples` image.
- `build_images.sh` — Docker equivalent of `build_sif.sh`.

### Environment
- `renv.lock`, `renv/`, `.Rprofile` — `renv` project library pinning R package versions for local runs.
- `AGENTS.md` — detailed engineering notes on the pipeline internals and known gotchas.

## Building & running

```bash
# Run in containers (recommended)

# Build container images (Apptainer) 
./build_sif.sh                    # base + all experiments
./build_sif.sh --skip-base        # reuse existing base image

# Run one experiment (generate + score)
working_dir="<WORKING_DIR>"
cd $working_dir
unset http_proxy # For HPC users
apptainer run \
    --cleanenv \
    --pwd /workspace \
    --env MODEL=huggingface/zai-org/GLM-4.7-Flash \
    --env RUNS=25 \
    --env API_BASE=http://d4053:8089/v1 \
	--bind $working_dir/GLM-4_7-Flash:/workspace/submissions \
    $working_dir/images/tinygrad-bench_exp_1.sif
```

### Parse benchmark results, generate pass@k csv file

```python
python parse_results.py <SUBMISSION_DIR>
```

Results are written to `results.md` in the submission directory; each test file
contributes its pass/fail counts to the aggregate.
