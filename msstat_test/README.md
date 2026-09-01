# MSstats Documentation → LLM Hallucination Study

A benchmark that measures whether **the shape of MSstats' documentation** changes how
often an LLM hallucinates when writing R code against it. It has two tiers:

1. **Structural** (§1-6 below): re-tests a finding from this workspace's prior tinygrad
   and r-polars studies — that adding **one worked example** to otherwise-minimal docs
   measurably reduces hallucination — against MSstats' API surface (argument names,
   column names).
2. **Value-based** (§7): does documentation change a real *judgment call* — specifically,
   whether the model chooses `dataProcess()`'s default normalization when it's
   statistically inappropriate — scored against known spike-in ground truth, not a
   structural pass/fail.

This README covers what the experiment actually does, the MSstats domain it's built on,
and how a result here would feed back into rewriting MSstats' real documentation.

---

## 1. What MSstats does (the workflows under test)

MSstats is a Bioconductor package for **differential protein abundance analysis** in
mass-spectrometry proteomics: given intensity measurements across experimental
conditions, it outputs a table of proteins with log2 fold-changes and adjusted p-values.
It's a stateful statistical pipeline, not a Q&A surface — so the hallucination risk is
concentrated in a few specific places: argument names, input-schema column names, and
output-schema column names.

**Acquisition-type workflows.** MSstats serves three experiment types, each fed by
different upstream instrument software, which is exactly where cross-tool/cross-package
hallucination risk lives:

| Workflow | Typical upstream tools | Converter function |
|---|---|---|
| DDA (label-free) | MaxQuant, Proteome Discoverer, Skyline, FragPipe, Spectronaut | `MaxQtoMSstatsFormat`, `PDtoMSstatsFormat`, `SkylinetoMSstatsFormat`, `FragPipetoMSstatsFormat`, `SpectronauttoMSstatsFormat` |
| DIA | Spectronaut, DIA-NN, OpenSWATH, DIA-Umpire | `SpectronauttoMSstatsFormat`, `DIANNtoMSstatsFormat`, `OpenSWATHtoMSstatsFormat`, `DIAUmpiretoMSstatsFormat` |
| SRM/PRM (targeted) | Skyline | `SkylinetoMSstatsFormat` |

(TMT/isobaric labeling and PTM analysis live in sibling packages, `MSstatsTMT` /
`MSstatsPTM` — out of scope here, but a real cross-package hallucination trap: an LLM that
confuses `dataProcess`/`dataSummarizationPTM` naming is hallucinating across packages, not
just within one.)

**The canonical pipeline** (every workflow converges here after conversion):

```
raw tool output → *toMSstatsFormat converter → dataProcess() → groupComparison() → filter by adj.pvalue
   (tool-specific)      (standardize columns)     (normalize,        (fit models,        (FDR-based
                                                    summarize to        return log2FC +     significance
                                                    protein level)       adjusted p-value)   call)
```

Every stage is a distinct, well-documented function with its own argument surface and its
own return schema — which is why the task design here makes each pipeline stage its own
isolated stub: it lets us attribute a hallucination to a specific function's docs, not to
"the pipeline" vaguely.

---

## 2. The method: fixed-signature stubs scored by a real oracle

```
fixed-signature stub + docstring task + doc variant
  → LLM completes the function body
  → execute in real R
  → testthat structural assertions (pinned to reference/probe_output.txt)
  → pass / fail
```

The **stub signature and docstring are held constant**; only the documentation injected
above the stub varies across conditions. Because each stub is one pipeline stage with a
typed handoff (e.g. `task_04` takes exactly the object `dataProcess()` really returns),
the model can only hallucinate the *API surface* inside the function body — not the
harness plumbing or how stages wire together. A pass means the model read the injected
docs (or already knew the API) and used it correctly; a fail is a caught hallucination,
mechanically — no LLM-as-judge anywhere in the scoring path.

### Tasks built so far

- **`task_03_data_process`** — `raw` MSstats-format data.frame → `dataProcess()`'s
  processed object. Targets **argument names/defaults** (`normalization`,
  `summaryMethod`, `MBimpute`, ...).
- **`task_04_group_comparison`** — processed object + contrast matrix →
  `groupComparison()`'s results table. Targets the **real output column names**
  (`log2FC`, `adj.pvalue`, not `foldChange`/`padj`/`FDR`) — the single highest-value
  hallucination site, because this is the table a real analysis pipeline downstream would
  key off of by column name.

Both tests build their own fixtures independently (from the in-package `DIARawData`
dataset) rather than chaining off each other's submissions, so a bad `task_03` completion
can't cause a false failure in `task_04`'s grading.

---

## 3. The doc conditions

Only 2 conditions are built, both including the worked example — the narrower question
of "does one worked example help at all" is taken as settled (the prior tinygrad/r-polars
studies established it, and it's the premise this whole project starts from, not what's
being actively tested here). What's actually varying now is the incremental value of one
specific callout on top of an example that's already present:

| Condition | Contents | Role |
|---|---|---|
| `one_example` | function signature (`\usage`) + `\description` + the function's one real `\examples` script | "docs + examples" — the baseline |
| `pitfall_note` | `one_example` **+ a short callout paragraph on when `equalizeMedians`' core assumption fails** | "docs + examples + callouts" — the manipulation |

An earlier build of this project also had `none` (nothing injected) and `sig_description`
(signature + description, no example) as separate conditions, set up to directly
reproduce the `2a_2_sig_description` → `2a_3_one_example` rung from the prior studies
against MSstats. No LLM was ever actually run against them (the harness was only
hand-verified with gold solutions), so that comparison was never made — the conditions
were dropped to narrow scope to the callout question instead, not because the earlier
question was answered. It's a real gap in this project's evidence, not a settled result;
see §4.

One MSstats-specific wrinkle worth noting: unlike r-polars (many short, disconnected
`\dontrun`-style snippets per function, needing first-chunk extraction), MSstats' Rd
`\examples` sections are each a single continuous narrative script — `dataProcess`'s is 6
lines, `groupComparison`'s is 10, ending in exactly the line that prints the real
`ComparisonResult` schema (`testResultOneComparison$ComparisonResult`). So "one example"
here is unambiguous: it's the whole block, and it's also the most schema-revealing single
artifact MSstats' own docs contain for that function.

Docs are extracted straight from the **installed package's Rd database**
(`tools::Rd_db("MSstats")`) by `extract_docs.R`, not hand-copied — so the injected text is
guaranteed to match the real, currently-installed API, and updating MSstats later just
means re-running the script.

---

## 4. Why this matters for improving MSstats' actual documentation

The point of this study isn't the benchmark score — it's using the score to decide **where
to spend documentation-writing effort**.

**Open question, not yet run**: whether the prior tinygrad/r-polars finding ("one worked
example beats no example") transfers to MSstats specifically. The harness for that
comparison (`sig_description` vs `one_example`) was built and hand-verified this session,
but no LLM was ever actually queried against it before the conditions were narrowed to 2
(§3) — so this is a real gap, not a settled result. If it were run: a wide gap would mean
the highest-leverage documentation investment for MSstats is making sure *every* exported
function's man page has one concrete, runnable, schema-revealing example (Bioconductor
already mandates `\examples`, so this is a cheap, mechanical fix). A small or absent gap
would be a useful negative result pointing at a different mechanism (e.g. MSstats'
`\value` sections are unusually prose-rich and may already be doing an example's job —
see the excerpts in `reference/probe_output.txt`), motivating an ablation tier instead
(full docs minus schema, minus example output, minus type info) to isolate what's
actually load-bearing.

**What's actually being tested now** (§7 has the full detail): given examples are already
present in both conditions, does adding the specific `pitfall_note` callout change
`task_05`/`task_06`'s outcome in the right direction — and does it do so *without*
over-correcting on the task where the default is already right?

Because assertions are pinned to `reference/probe_output.txt` / `probe_normalization*.txt`
(real installed-package output, not memory), any result from either question is a direct,
reproducible measurement, not a proxy, and the same harness re-runs cleanly against a
future MSstats release to check whether a documentation fix actually moved the needle.
Extending either question to the (currently deferred) input-schema and
converter-selection tasks would test how much of the effect is specific to richly-schema'd
docs like `groupComparison`'s, versus general across the whole API surface.

---

## 5. Repository layout

- `reference/probe.R`, `probe_output.txt` — the one-time environment probe; every test
  assertion is pinned to this file, never to memory.
- `tasks/`, `tests/` — the fixed-signature stubs and their `testthat` oracles.
- `extract_docs.R` — builds `docs_conditions/{one_example,pitfall_note}/` from the
  installed MSstats Rd database.
- `generate.py`, `exp_2a/Prompt.md`, `benchmark.R`, `run.sh` — generation (litellm +
  instructor, against a local vLLM endpoint) and scoring harness.
- `reference/prepare_normalization_fixture.R`, `normalization_fixture.rds`,
  `probe_normalization.R`, `probe_normalization_output.txt` — one-time data prep and
  ground-truth probe for `task_06` (spike-in, value-based tier, §7).
- `reference/prepare_talus_fixture.R`, `talus_fixture.rds`, `probe_talus_normalization.R`,
  `probe_talus_normalization_output.txt` — same, for `task_05` (Talus, §7).
- `analyze_normalization_results.py` / `analyze_talus_results.py` — tabulate `task_06`'s
  FDP/bias evidence and `task_05`'s SE evidence, respectively, across doc
  conditions/runs.
- `base.def`/`Dockerfile.base`, `experiment.def`/`Dockerfile.experiment`,
  `build_sif.sh`/`build_images.sh` — Apptainer and Docker build files for HPC runs,
  mirroring `r_polars_test`'s container setup (see §6). Not built or tested locally.

## 6. Running it

```bash
Rscript extract_docs.R          # rebuild docs_conditions/ (idempotent)

# per condition, against a running vLLM (or other OpenAI-compatible) endpoint:
CONDITION=one_example  MODEL=huggingface/<model> API_BASE=http://<host>:port/v1 RUNS=5 ./run.sh
CONDITION=pitfall_note MODEL=huggingface/<model> API_BASE=http://<host>:port/v1 RUNS=5 ./run.sh
```

Compare `submissions/<condition>/run_*/results.md` pass rates across conditions; the
`one_example` vs `pitfall_note` delta is the headline number.

For the value-based tier (§7), first build both fixtures once (`Rscript
reference/prepare_normalization_fixture.R` and `Rscript reference/prepare_talus_fixture.R`,
both require `msstat-data/` locally), then run `run.sh` the same way but with
`CONDITION=pitfall_note` included. `task_06`'s fixture is large enough that a single
`dataProcess()` call takes ~75s — set `TEST_TIMEOUT=300` when running it; `task_05`'s
fixture is small and fast at the default timeout. Then:
`python analyze_normalization_results.py submissions/` (task_06) and
`python analyze_talus_results.py submissions/` (task_05).

### Running in a container (HPC)

`build_sif.sh` (Apptainer) and `build_images.sh` (Docker) mirror `r_polars_test`'s
container setup: a base image (R + MSstats/MSstatsConvert via Bioconductor + testthat/
callr + litellm/instructor/pydantic) plus one experiment image per doc condition. Unlike
`r_polars_test` (which bakes a single condition into `tasks/` at build time), these
images build *all* doc conditions and just set `CONDITION` as that image's runtime
default — `run.sh` already picks `docs_conditions/$CONDITION/` dynamically, so the same
image works for any condition via `--env CONDITION=...`. Not built or tested in this
environment — build and run on HPC:

```bash
# Apptainer
./build_sif.sh                    # base + both conditions
./build_sif.sh --skip-base        # reuse an existing msstats-bench-base.sif

apptainer run --cleanenv \
    --env MODEL=huggingface/<model> --env RUNS=5 --env API_BASE=http://<host>:port/v1 \
    --bind /path/to/submissions:/workspace/submissions \
    msstats-bench-exp_2a-pitfall_note.sif

# Docker
./build_images.sh                 # base + both conditions
docker run --rm \
    -e MODEL=huggingface/<model> -e RUNS=5 -e API_BASE=http://<host>:port/v1 \
    -v /path/to/submissions:/workspace/submissions \
    msstats-bench:exp_2a-pitfall_note
```

---

## 7. Value-based tier: does documentation change a real decision?

Devs flagged that MSstats' docs don't explain *when* to deviate from defaults.
Concretely: `dataProcess()`'s default `normalization = "equalizeMedians"` shifts each
run so their **medians line up**, which implicitly assumes *most proteins don't change
across conditions* — a normal, reasonable assumption. It breaks when a large fraction of
the proteome is expected to change, and when it breaks, it doesn't just lose power, it
actively creates false discoveries.

**The benchmark this is scored against**: a real spike-in dataset (`msstat-data/`, not
committed — see `.gitignore`) with a constant human proteome background and an *E. coli*
proteome spiked in at 5 known ratios (A=1x baseline, B=1.5x, C=2x, D=2.5x, E=3x). Because
the true answer is known by construction (every human protein should show log2FC=0,
every *E. coli* protein should show log2FC = log2(spike ratio)), correctness is
*mechanically* checkable via false discovery proportion (FDP) — not a structural
pass/fail, and not an LLM judge.

**Reproduced independently in this renv** (`reference/probe_normalization_output.txt`),
FDP with the default `equalizeMedians` vs. `normalization = FALSE`, across all 4
contrasts:

| Contrast (true spike ratio) | `equalizeMedians` (default) FDP | `normalization = FALSE` FDP |
|---|---|---|
| E-A (3x) | **49.8%** | 14.5% |
| D-A (2.5x) | **31.4%** | 14.4% |
| C-A (2x) | **24.4%** | 15.2% |
| B-A (1.5x) | 14.3% | 15.0% |

The gap is a clean dose-response: it scales with how much of the proteome is truly
shifting, and vanishes at the smallest spike level — direct mechanistic confirmation of
*why* the assumption fails, not just *that* it does.

### The task and how it's scored

`tasks/task_06_choose_normalization.R` asks the model to process this dataset given a
plain description of the experimental design (the spike-in structure above) — without
naming "median normalization" or the pitfall — and choose settings itself. It is
deliberately **not graded pass/fail** the way tasks 03/04 are: a single FDP threshold
isn't meaningful across every contrast, since the true effect size varies by design.
Instead, `tests/test_06_choose_normalization.R`:

- runs a harness-side (fixed, not model-written) `groupComparison()` across all 4
  contrasts using the known species labels, and logs FDP + log2FC bias per contrast;
- statically greps the submitted code for the `normalization =` value actually chosen,
  independent of the numeric outcome;
- keeps one loose sanity assertion (didn't crash, right object shape) so it still counts
  in `benchmark.R`'s pass-rate aggregate, but the real signal is the logged metrics.

`analyze_normalization_results.py` tabulates these across doc conditions/runs — the
"prove it with evidence" deliverable, not a vibe.

### The second task: a dataset where the default is the *right* call

`tasks/task_05_choose_normalization_talus.R` is the counterpart, built from
`High-DIA-Talus-Perturbations-DIANN` (a real THP-1 chromatin-proteomics drug-perturbation
experiment — DMSO control vs. two compounds, DbET6 and PF477736). Unlike the spike-in
data, this dataset has no constructed species-label ground truth, so FDP isn't
computable. Initial exploration here included a false start (recorded in `Notes.md`) —
comparing raw hit *counts* between normalization settings looked like the same failure
mode as the spike-in data, until checking standard error (SE) alongside the fold-change
estimate showed the opposite signature.

BRD2/BRD3/BRD4 are bromodomain proteins and the direct, well-established target of the
dBET6 degrader — the closest thing to a "known true effect" this dataset has. Reproduced
independently (`reference/probe_talus_normalization_output.txt`):

| Landmark protein | log2FC (equalizeMedians) | log2FC (FALSE) | SE (equalizeMedians) | SE (FALSE) |
|---|---|---|---|---|
| BRD2 | -1.029 | -0.980 | 0.090 | 0.213 (2.4x) |
| BRD3 | -2.022 | -1.985 | 0.103 | 0.183 (1.8x) |
| BRD4 | -1.184 | -1.140 | 0.131 | 0.260 (2.0x) |

Every protein: the point estimate barely moves, but the SE roughly doubles without
normalization — the signature of normalization *removing real technical noise* (real
per-run median intensities vary ~0.5-1 log2 unit even between replicates of the same
condition), not introducing bias. That's the opposite of the spike-in dataset, where the
point estimate itself shifted away from the known true value, and by more the bigger the
true effect was. `tests/test_05_choose_normalization_talus.R` logs SE and log2FC for the
landmark proteins (harness-side `groupComparison`, same isolation principle as task_06)
against these pinned reference values; `analyze_talus_results.py` tabulates them.

**Why both tasks matter together, not just individually**: `task_06` tests whether the
model correctly turns normalization *off* when the assumption is violated. `task_05`
tests the opposite failure mode — does it keep normalization *on* when the assumption
holds, or does reading the `pitfall_note` guidance cause it to reflexively recommend
`normalization = FALSE` on *everything*? A doc fix that only helps on one of these two
tasks isn't actually teaching calibrated judgment, just swapping one blanket default for
another.

### The second doc condition: the literal candidate doc fix

`extract_docs.R`'s **`pitfall_note`** condition is `one_example` plus a short paragraph
operationalizing exactly the fix a doc maintainer would propose — that `equalizeMedians`
assumes most proteins don't change, names when that fails, and suggests
`normalization = FALSE` (or checking the SE-vs-point-estimate diagnostic) as an
alternative. Comparing `pitfall_note` against `one_example` (docs + example, but no
mention of this edge case) is the direct test of whether adding this specific guidance
changes model behavior and, more importantly, the resulting FDP/SE.

### What a result here would mean

- If `pitfall_note` lowers FDP on `task_06` (closer to the `normalization = FALSE`
  reference numbers) **and** leaves `task_05`'s SE close to the `equalizeMedians`
  reference (not inflated toward the `FALSE` reference): this exact paragraph is worth
  adding to `dataProcess()`'s real docs — it's teaching calibrated judgment, not a
  reflexive rule, backed by numbers on both sides.
- If `pitfall_note` helps `task_06` but pushes `task_05` toward `normalization = FALSE`
  too (inflating its SE toward the `FALSE` reference): the guidance is over-generalizing
  — the model learned "avoid the default" rather than "check whether the assumption
  holds," which would itself be an important finding about the limits of a short prose
  warning.
- If it doesn't move either task: either the guidance needs to be more prominent (e.g. in
  the function signature's default itself, not prose) or the model isn't using the
  injected docs to inform this kind of judgment call at all.

### Explicitly out of scope for this tier

- The MBR (match-between-runs) variant of the spike-in dataset — using No-MBR only.
- Actually running LLM generation for either task (same blocker as the structural tier:
  no API/vLLM endpoint in this environment). Verified instead with hand-written reference
  solutions run through the real `benchmark.R` harness, reproducing the pinned reference
  tables exactly for both `task_05` and `task_06`.

---

## 8. Future scope

- Generate + score real runs for both tiers — no LLM has been queried yet (needs a vLLM
  endpoint and `litellm`/`instructor` installed wherever `generate.py` runs).
- Build the remaining structural pipeline-stage tasks: input-schema construction,
  converter selection, and FDR-based significance filtering.
- Add a `full`-dump doc condition (whole vignette + man page) and finer-grained
  ablations, once the one-example result above is in.
- Containerize the pipeline (Docker/Apptainer) for HPC runs, matching the sibling
  tinygrad/r-polars studies.
