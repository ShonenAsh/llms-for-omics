#!/usr/bin/env Rscript
# extract_docs.R -- inject MSstats Rd docs into task templates
#
# Simplified port of r_polars_test/extract_docs.R. MSstats Rd topics are
# flat, single-name files (dataProcess.Rd, groupComparison.Rd, ...) -- unlike
# r_polars's S3-method Rd files with double-underscore aliases -- so no
# alias-resolution machinery is needed here: an @requires token is looked up
# directly as a doc-db key.

suppressWarnings(suppressMessages(library(tools)))

# CONFIG
PKG       <- "MSstats"
TASKS_DIR <- "tasks"
OUT_DIR   <- "docs_conditions"
COMMENT   <- "# "   # injected docs are commented so the file always parses

# DOC EXTRACTION
load_doc_db <- function(pkg = PKG) {
  tryCatch(Rd_db(pkg), error = function(e)
    stop("Rd_db('", pkg, "') failed -- is the package installed? ",
         conditionMessage(e), call. = FALSE))
}

.rd_flatten <- function(x) {
  if (is.null(x)) return("")
  if (is.character(x)) return(paste(x, collapse = ""))
  if (is.list(x)) return(paste(vapply(x, .rd_flatten, character(1)), collapse = ""))
  as.character(x)
}

rd_section <- function(topic, tag) {
  hits <- Filter(function(e) identical(attr(e, "Rd_tag"), tag), topic)
  if (!length(hits)) return("")
  trimws(paste(vapply(hits, .rd_flatten, character(1)), collapse = "\n"))
}

build_doc_records <- function(db = load_doc_db()) {
  records <- list()
  for (nm in names(db)) {
    topic <- db[[nm]]
    fun <- sub("\\.Rd$", "", nm)
    records[[fun]] <- list(
      usage       = rd_section(topic, "\\usage"),
      description = rd_section(topic, "\\description"),
      examples    = rd_section(topic, "\\examples")
    )
  }
  records
}

# RESOLUTION
read_required <- function(task_file) {
  lines <- readLines(task_file, warn = FALSE)
  hit <- grep("@requires", lines, value = TRUE)
  if (!length(hit)) stop("No @requires line in ", task_file, call. = FALSE)
  raw  <- sub(".*@requires", "", hit[1])
  toks <- strsplit(raw, "[,[:space:]]+")[[1]]
  unique(toks[nzchar(toks)])
}

# CONDITION SPECS
# Narrowed to exactly 2 conditions (per the user's call): both already include
# usage+description+examples (the "one example helps" question is considered
# settled from prior tinygrad/r-polars work + this project's own tasks 03/04);
# what's under test now is only the incremental value of the callout on top of
# that, isolating "add this specific guidance" as the only variable between
# the two conditions.
spec_defaults <- list(usage = FALSE, description = FALSE, examples = FALSE,
                       pitfall_note = FALSE)
mk <- function(...) modifyList(spec_defaults, list(...))

CONDITIONS <- list(
  # "docs + examples"
  "one_example"  = mk(usage = TRUE, description = TRUE, examples = TRUE),
  # "docs + examples + callouts" -- the literal candidate doc fix under test
  # for tasks 05/06 (normalization-choice tier): one_example + a short
  # paragraph on when equalizeMedians' core assumption fails.
  "pitfall_note" = mk(usage = TRUE, description = TRUE, examples = TRUE, pitfall_note = TRUE)
)

# Free-text guidance, keyed by function name, injected only when a task
# @requires that function AND the active condition sets pitfall_note = TRUE.
# This is not derived from the installed package's Rd db -- it's the candidate
# documentation fix itself, made testable.
PITFALL_NOTES <- list(
  dataProcess = paste(
    paste(
      "Practical note on normalization: `equalizeMedians` (the default) assumes",
      "most proteins/features are NOT differentially abundant across conditions,",
      "and uses that assumption to align each run's median. This assumption fails",
      "when a large fraction of what's measured is expected to change (e.g.",
      "spike-in benchmarks, or experiments where a large share of the measured set",
      "is itself a direct target of the perturbation) -- in such cases median",
      "normalization can compress true fold-changes and inflate false positives",
      "among otherwise-stable proteins/features."
    ),
    paste(
      "A concrete way to check which situation you're in: run the comparison",
      "with normalization on and with normalization = FALSE, then compare",
      "fold-change estimates and standard errors between the two. If",
      "fold-change estimates barely move but standard errors shrink",
      "substantially with normalization on, the assumption holds and",
      "normalization is correctly removing technical noise -- keep it on. If",
      "fold-change estimates themselves shift noticeably between the two",
      "settings (and shift more for proteins/features with larger true",
      "effects), the assumption is likely violated -- consider",
      "`normalization = FALSE`, or another normalization strategy, instead."
    ),
    sep = "\n\n"
  )
)

# COMPOSITION
.comment <- function(txt) {
  if (is.null(txt) || !nzchar(txt)) return(character(0))
  paste0(COMMENT, strsplit(txt, "\n", fixed = TRUE)[[1]])
}

# one function's doc -> character vector of comment lines
compose_block <- function(fun, rec, spec) {
  parts <- character()
  if (spec$usage && nzchar(rec$usage))
    parts <- c(parts, paste0(COMMENT, "Signature:"), .comment(rec$usage))
  if (spec$description && nzchar(rec$description))
    parts <- c(parts, paste0(COMMENT, "Description:"), .comment(rec$description))
  if (spec$examples && nzchar(rec$examples))
    # MSstats \examples blocks are a single continuous narrative script per
    # function (unlike r_polars's many small snippets), so "one example" is
    # simply the whole block -- no first-chunk splitting needed.
    parts <- c(parts, paste0(COMMENT, "Example:"), .comment(rec$examples))
  if (isTRUE(spec$pitfall_note) && !is.null(PITFALL_NOTES[[fun]]))
    parts <- c(parts, paste0(COMMENT, "Note:"), .comment(PITFALL_NOTES[[fun]]))

  if (!length(parts)) return(character(0))
  c(paste0(COMMENT, "--- ", fun, " ---"), parts, COMMENT)
}

# union of required functions -> full top-of-file doc block (comment lines)
compose_top_block <- function(required, spec, records) {
  lines   <- character()
  missing <- character()
  for (fun in required) {
    rec <- records[[fun]]
    if (is.null(rec)) { missing <- c(missing, fun); next }
    lines <- c(lines, compose_block(fun, rec, spec))
  }
  if (length(lines)) lines <- c(paste0(COMMENT, "--- docs ---"), lines)
  list(lines = lines, missing = unique(missing))
}

# INJECTION + DRIVER
inject_docs <- function(task_file, doc_lines, out_file) {
  lines <- readLines(task_file, warn = FALSE)
  lines <- lines[!grepl("@requires", lines)]
  idx <- grep("<<DOCS>>", lines)
  if (!length(idx)) stop("No <<DOCS>> marker in ", task_file, call. = FALSE)
  i <- idx[1]
  before <- if (i > 1) lines[seq_len(i - 1)] else character(0)
  after  <- if (i < length(lines)) lines[(i + 1):length(lines)] else character(0)
  writeLines(c(before, doc_lines, after), out_file)
}

build_condition <- function(cond, records) {
  spec <- CONDITIONS[[cond]]
  cdir <- file.path(OUT_DIR, cond)
  dir.create(cdir, recursive = TRUE, showWarnings = FALSE)
  task_files <- list.files(TASKS_DIR, pattern = "\\.R$", full.names = TRUE)
  if (!length(task_files)) stop("No .R task files in ", TASKS_DIR, call. = FALSE)
  miss_log <- list()
  for (tf in task_files) {
    required <- read_required(tf)
    blk <- compose_top_block(required, spec, records)
    inject_docs(tf, blk$lines, file.path(cdir, basename(tf)))
    if (length(blk$missing))
      miss_log[[paste(cond, basename(tf))]] <- blk$missing
  }
  if (length(miss_log)) {
    message("Unresolved @requires tokens (check function name against Rd_db):")
    for (k in names(miss_log))
      message("  ", k, ": ", paste(miss_log[[k]], collapse = ", "))
  }
  message("built: ", cond)
}

build_all <- function() {
  records <- build_doc_records(load_doc_db())
  for (cond in names(CONDITIONS)) build_condition(cond, records)
  invisible(records)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1L && nzchar(args[1L])) {
    cond <- args[1L]
    if (!cond %in% names(CONDITIONS))
      stop("Unknown condition: ", cond, ". Available: ",
           paste(names(CONDITIONS), collapse = ", "), call. = FALSE)
    records <- build_doc_records(load_doc_db())
    build_condition(cond, records)
  } else {
    build_all()
  }
}
