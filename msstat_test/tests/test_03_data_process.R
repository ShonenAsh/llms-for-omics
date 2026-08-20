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
safe_source(file.path(submission_path, "task_03_data_process.R"))

test_that("task_03_data_process returns MSstats' real processed-object structure", {
  data(DIARawData, package = "MSstats")
  result <- task_03_data_process(DIARawData)

  # Real dataProcess() output is a plain list (verified in reference/probe_output.txt)
  expect_type(result, "list")
  expect_true(all(c("FeatureLevelData", "ProteinLevelData", "SummaryMethod") %in% names(result)))

  # Protein-level table's real column names, pinned to reference/probe_output.txt --
  # never hand-typed from memory.
  expect_true(all(
    c("RUN", "Protein", "LABEL", "LogIntensities", "GROUP", "SUBJECT") %in%
      colnames(result$ProteinLevelData)
  ))

  # 2 proteins x 4 runs under default summarization -- a hallucinated normalization/
  # summaryMethod argument would change this row count.
  expect_equal(nrow(result$ProteinLevelData), 8)
})
