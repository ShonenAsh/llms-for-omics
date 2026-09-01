#!/usr/bin/env Rscript
# benchmark.R -- testthat runner for MSstats task submissions.
# Near-verbatim port of r_polars_test/benchmark.R, minus polars-specific
# thread-pool plumbing, plus per-subprocess cwd isolation (see below).
suppressPackageStartupMessages(library(testthat))
suppressPackageStartupMessages(library(callr))

`%||%` <- function(a, b) if (is.null(a)) b else a

cli_args <- commandArgs(trailingOnly = TRUE)
args <- list()
for (i in seq_along(cli_args)) {
  if (grepl("^--", cli_args[i])) {
    key <- sub("^--", "", cli_args[i])
    val <- if (i < length(cli_args) && !grepl("^--", cli_args[i + 1])) {
      i <<- i + 1
      cli_args[i]
    } else { TRUE }
    args[[key]] <- val
  }
}

submission_dir <- normalizePath(args[["submission-dir"]] %||% "submissions", mustWork = FALSE)
test_dir_path  <- normalizePath(args[["test-dir"]] %||% "tests", mustWork = FALSE)
results_file   <- args[["results"]] %||% file.path(submission_dir, "results.md")

# Per-test-file wall-clock timeout (seconds). A generated submission with an
# infinite loop or a blocking call would otherwise hang the whole benchmark
# forever; each file runs in its own subprocess and is killed past this limit.
test_timeout   <- suppressWarnings(as.numeric(
  args[["timeout"]] %||% Sys.getenv("TEST_TIMEOUT", "120")))
if (is.na(test_timeout) || test_timeout <= 0) test_timeout <- 120

Sys.setenv(SUBMISSION_DIR = submission_dir)

stopifnot("test_dir not found" = dir.exists(test_dir_path))
dir.create(submission_dir, showWarnings = FALSE, recursive = TRUE)

# Count expected expect_*() calls in test files statically
expected_total <- 0L
expected_per_file <- list()
for (tf in list.files(test_dir_path, pattern = "\\.R$", full.names = TRUE)) {
  tlines <- readLines(tf, warn = FALSE)
  n <- sum(grepl("expect_", tlines))
  fn <- basename(tf)
  expected_per_file[[fn]] <- n
  expected_total <- expected_total + n
}

# Run each test file in an isolated subprocess with a hard timeout. This keeps
# console output visible, captures it for results.md, and guarantees a hung or
# crashing submission can never stall the benchmark: kill_tree() reaps the
# whole process tree. The concatenated per-file results reproduce what
# test_dir() returned, so the aggregation below is unchanged.
child_libpath <- .libPaths()
child_env     <- c(callr::rcmd_safe_env(), SUBMISSION_DIR = submission_dir,
                    PROJECT_ROOT = getwd())

run_test_file <- function(f) {
  logf <- tempfile()
  p <- callr::r_bg(
    function(file) {
      # dataProcess()/groupComparison() default to use_log_file = TRUE, which
      # writes an MSstats log file into the current working directory as a
      # side effect. A submission that doesn't pass use_log_file = FALSE
      # (plausible, and not something we should rely on a submission getting
      # right) would otherwise scatter log files into the repo root, racing
      # across parallel runs. Isolate cwd per subprocess to absorb this.
      wd  <- tempfile("msstats_bench_")
      dir.create(wd)
      old <- setwd(wd)
      on.exit({ setwd(old); unlink(wd, recursive = TRUE, force = TRUE) }, add = TRUE)

      suppressPackageStartupMessages(library(testthat))
      testthat::test_file(file, reporter = "summary", stop_on_failure = FALSE)
    },
    args = list(file = f),
    libpath = child_libpath, env = child_env,
    stdout = logf, stderr = "2>&1", poll_connection = FALSE
  )
  p$wait(timeout = test_timeout * 1000)
  if (p$is_alive()) {
    p$kill_tree()
    out <- tryCatch(readLines(logf, warn = FALSE), error = function(e) character())
    unlink(logf)
    return(list(results = NULL, output = c(out,
      sprintf("<<TIMEOUT: %s killed after %gs>>", basename(f), test_timeout))))
  }
  out <- tryCatch(readLines(logf, warn = FALSE), error = function(e) character())
  unlink(logf)
  res <- tryCatch(p$get_result(), error = function(e) e)
  if (inherits(res, "error")) {
    return(list(results = NULL, output = c(out,
      sprintf("<<ERROR in %s: %s>>", basename(f), conditionMessage(res)))))
  }
  list(results = res, output = out)
}

test_files <- list.files(test_dir_path, pattern = "\\.R$", full.names = TRUE)
results <- list()
test_output <- character()
failed_files <- character()  # files that timed out or errored before finishing
for (tf in test_files) {
  r <- run_test_file(tf)
  cat(r$output, sep = "\n"); cat("\n")            # keep console output visible
  test_output <- c(test_output, r$output, "")
  if (is.null(r$results)) failed_files <- c(failed_files, basename(tf))
  else results <- c(results, r$results)
}

per_file <- list()
for (tr in results) {
  fn <- basename(tr$file %||% "unknown")
  if (is.null(per_file[[fn]])) {
    per_file[[fn]] <- list(n = 0, passed = 0, failed = 0, skipped = 0)
  }
  for (expectation in tr$results) {
    per_file[[fn]]$n <- per_file[[fn]]$n + 1
    if (inherits(expectation, "expectation_success")) {
      per_file[[fn]]$passed <- per_file[[fn]]$passed + 1
    } else if (inherits(expectation, "expectation_failure") ||
               inherits(expectation, "expectation_error")) {
      per_file[[fn]]$failed <- per_file[[fn]]$failed + 1
    } else if (inherits(expectation, "expectation_skip")) {
      per_file[[fn]]$skipped <- per_file[[fn]]$skipped + 1
    }
  }
}

# Aggregate per-test_that results (group expectations by test name)
per_test <- list()
for (tr in results) {
  fn <- basename(tr$file %||% "unknown")
  for (expectation in tr$results) {
    test_full <- expectation$test %||% "unknown"
    test_short <- sub("\\s.*", "", test_full)  # first word = function name
    if (is.null(per_test[[fn]])) {
      per_test[[fn]] <- list()
    }
    if (is.null(per_test[[fn]][[test_short]])) {
      per_test[[fn]][[test_short]] <- list(passed = 0, failed = 0)
    }
    if (inherits(expectation, "expectation_success")) {
      per_test[[fn]][[test_short]]$passed <- per_test[[fn]][[test_short]]$passed + 1
    } else {
      per_test[[fn]][[test_short]]$failed <- per_test[[fn]][[test_short]]$failed + 1
    }
  }
}

test_lines <- character()
for (fn in names(per_test)) {
  for (tn in names(per_test[[fn]])) {
    s <- per_test[[fn]][[tn]]
    status <- if (s$failed > 0) "FAILED" else "PASSED"
    test_lines <- c(test_lines, sprintf("%s::%s %s", fn, tn, status))
  }
}
for (fn in failed_files) {
  test_lines <- c(test_lines, sprintf("%s TIMEOUT/ERROR (no results)", fn))
}

sink(results_file)
cat("# Benchmark Results\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("## Test Output\n\n```\n")
cat(test_output, sep = "\n")
cat("\n```\n\n")

cat("## Summary\n\n")
cat("| File | Expected | Passed | Failed | Skipped |\n")
cat("|------|----------|--------|--------|--------|\n")

total_passed <- total_failed <- total_skipped <- 0
for (fn in names(expected_per_file)) {
  exp_n <- expected_per_file[[fn]]
  s <- per_file[[fn]] %||% list(passed = 0, failed = 0, skipped = 0)
  total_passed <- total_passed + s$passed
  total_failed <- total_failed + (exp_n - s$passed)
  total_skipped <- total_skipped + s$skipped
  cat(sprintf("| %s | %d | %d | %d | %d |\n", fn, exp_n, s$passed, exp_n - s$passed, s$skipped))
}
cat(sprintf("| **Total** | **%d** | **%d** | **%d** | **%d** |\n",
            expected_total, total_passed, expected_total - total_passed, total_skipped))
cat("\n## Pass rate\n\n")
if (expected_total > 0) {
  rate <- round(total_passed / expected_total * 100, 1)
  cat(sprintf("**%.1f%%** (%d/%d)\n", rate, total_passed, expected_total))
} else {
  cat("No tests found.\n")
}
cat("\n## Individual Results\n\n")
cat(test_lines, sep = "\n")
cat("\n")
sink()
