library("testthat")
suppressPackageStartupMessages(library("MSstats"))

submission_path <- Sys.getenv("SUBMISSION_DIR", unset = normalizePath(file.path("..", "tasks")))
project_root <- Sys.getenv("PROJECT_ROOT", unset = normalizePath(".."))

# Strip any test_that(...) blocks from the sourced submission before evaluating it,
# in case a completion accidentally includes test code alongside the stub body.
safe_source <- function(file) {
  lines <- readLines(file, warn = FALSE)
  result <- character()
  depth <- 0L
  in_test <- FALSE
  for (line in lines) {
    if (grepl("^\\s*test_that\\(", line)) {
      in_test <- TRUE
      depth <- depth + nchar(gsub("[^{]", "", line)) - nchar(gsub("[^}]", "", line))
      if (depth <= 0L) in_test <- FALSE
      next
    }
    if (in_test) {
      depth <- depth + nchar(gsub("[^{]", "", line)) - nchar(gsub("[^}]", "", line))
      if (depth <= 0L) { in_test <- FALSE; depth <- 0L }
      next
    }
    result <- c(result, line)
  }
  source(textConnection(result), local = globalenv())
}

submission_file <- file.path(submission_path, "task_06_choose_normalization.R")
safe_source(submission_file)

# Static diagnostic: which `normalization` value did the submission actually
# choose? Cheap, no R semantics needed -- answers "did it even reconsider the
# default" independent of the numeric outcome below.
chosen_normalization <- local({
  code <- paste(readLines(submission_file, warn = FALSE), collapse = "\n")
  m <- regmatches(code, regexpr("normalization\\s*=\\s*[^,)\\n]+", code, perl = TRUE))
  if (length(m) == 0) "<not set, i.e. default equalizeMedians>" else trimws(m)
})

# Ground truth: constant human background, E. coli spiked at 5 known ratios.
# Pinned to reference/probe_normalization_output.txt -- never hand-typed.
EXPECTED_FC <- c("E-A" = 3, "D-A" = 2.5, "C-A" = 2, "B-A" = 1.5)

calculateFDP <- function(comparisonResult, comparison, fdp_cutoff = 0.05) {
  cr <- comparisonResult[comparisonResult$Label == comparison & is.na(comparisonResult$issue), ]
  ecoli <- cr[grepl("ECOLI", cr$Protein) & cr$adj.pvalue < fdp_cutoff, ]
  human <- cr[!grepl("ECOLI", cr$Protein) & cr$adj.pvalue < fdp_cutoff, ]
  n_sig <- nrow(ecoli) + nrow(human)
  list(FDP = if (n_sig > 0) nrow(human) / n_sig else NA_real_,
       n_ecoli_sig = nrow(ecoli), n_human_sig = nrow(human))
}

biasSummary <- function(comparisonResult, comparison) {
  cr <- comparisonResult[comparisonResult$Label == comparison & is.na(comparisonResult$issue), ]
  ecoli_fc <- cr[grepl("ECOLI", cr$Protein), "log2FC"]
  human_fc <- cr[!grepl("ECOLI", cr$Protein), "log2FC"]
  list(median_ecoli_log2FC = median(ecoli_fc, na.rm = TRUE),
       median_human_log2FC = median(human_fc, na.rm = TRUE))
}

test_that("task_06_choose_normalization: FDP/bias logged across the spike-in dose series", {
  raw <- readRDS(file.path(project_root, "reference", "normalization_fixture.rds"))
  result <- task_06_choose_normalization(raw)

  # Loose sanity check ONLY -- this task is deliberately NOT graded pass/fail on
  # structure, since a single FDP threshold isn't meaningful across every
  # contrast (the true effect size varies by design). The real signal is the
  # METRIC lines below, parsed by the results-analysis script.
  expect_type(result, "list")
  expect_true("ProteinLevelData" %in% names(result))

  groups <- sort(levels(result$ProteinLevelData$GROUP))
  comparison <- matrix(c(-1, 0, 0, 0, 1,
                          -1, 0, 0, 1, 0,
                          -1, 0, 1, 0, 0,
                          -1, 1, 0, 0, 0), nrow = 4, byrow = TRUE)
  row.names(comparison) <- c("E-A", "D-A", "C-A", "B-A")
  colnames(comparison) <- groups

  # Harness-side, fixed comparison step -- the submission does not control this
  # call, so grading isolates exactly the normalization/processing decision.
  cmp <- MSstats::groupComparison(contrast.matrix = comparison, data = result,
                                   use_log_file = FALSE, verbose = FALSE)

  for (contrast in c("E-A", "D-A", "C-A", "B-A")) {
    fdp <- calculateFDP(cmp$ComparisonResult, comparison = contrast)
    bias <- biasSummary(cmp$ComparisonResult, comparison = contrast)
    cat(sprintf(
      paste("METRIC contrast=%s chosen_normalization=%s FDP=%.4f",
            "n_ecoli_sig=%d n_human_sig=%d",
            "median_ecoli_log2FC=%.4f expected_ecoli_log2FC=%.4f median_human_log2FC=%.4f\n"),
      contrast, chosen_normalization, fdp$FDP, fdp$n_ecoli_sig, fdp$n_human_sig,
      bias$median_ecoli_log2FC, log2(EXPECTED_FC[[contrast]]), bias$median_human_log2FC
    ))
  }
})
