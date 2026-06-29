#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(testthat))

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
data_dir       <- normalizePath(args[["data-dir"]] %||% "data", mustWork = FALSE)

Sys.setenv(SUBMISSION_DIR = submission_dir)
Sys.setenv(DATA_PATH = data_dir)

stopifnot("test_dir not found" = dir.exists(test_dir_path))
dir.create(submission_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== polars Benchmark ===\n")
cat(sprintf("Submissions: %s\n", submission_dir))
cat(sprintf("Tests:       %s\n", test_dir_path))
cat(sprintf("Results:     %s\n\n", results_file))

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

results <- test_dir(test_dir_path, reporter = "summary", stop_on_failure = FALSE)

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

sink(results_file)
cat("# Benchmark Results\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
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
sink()

cat(sprintf("\nDone. %d/%d passed (%.1f%%)\n",
    total_passed, expected_total, if (expected_total > 0) total_passed / expected_total * 100 else 0))
