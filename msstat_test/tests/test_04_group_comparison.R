library("testthat")
suppressPackageStartupMessages(library("MSstats"))

submission_path <- Sys.getenv("SUBMISSION_DIR", unset = normalizePath(file.path("..", "tasks")))

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
safe_source(file.path(submission_path, "task_04_group_comparison.R"))

test_that("task_04_group_comparison returns MSstats' real ComparisonResult schema", {
  data(DIARawData, package = "MSstats")
  # Build the processed object independently of any task_03 submission, so this
  # test grades task_04 in isolation.
  processed <- MSstats::dataProcess(DIARawData, use_log_file = FALSE, verbose = FALSE)
  groups <- levels(processed$ProteinLevelData$GROUP)

  contrast_matrix <- matrix(c(1, -1), nrow = 1)
  colnames(contrast_matrix) <- groups
  row.names(contrast_matrix) <- paste(groups[1], groups[2], sep = "-")

  result <- task_04_group_comparison(processed, contrast_matrix)

  expect_s3_class(result, "data.frame")

  # The real ComparisonResult column names, pinned to reference/probe_output.txt --
  # this is the single highest-value hallucination target per CLAUDE.md.
  expect_equal(
    colnames(result),
    c("Protein", "Label", "log2FC", "SE", "Tvalue", "DF", "pvalue",
      "adj.pvalue", "issue", "MissingPercentage", "ImputationPercentage")
  )

  # One row per protein (2 proteins in DIARawData) for the single contrast.
  expect_equal(nrow(result), 2)
  expect_true(is.numeric(result$log2FC))
  expect_true(is.numeric(result$adj.pvalue))
})
