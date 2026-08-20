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

submission_file <- file.path(submission_path, "task_05_choose_normalization_talus.R")
safe_source(submission_file)

# Static diagnostic: which `normalization` value did the submission actually
# choose? Same helper as test_06_choose_normalization.R.
chosen_normalization <- local({
  code <- paste(readLines(submission_file, warn = FALSE), collapse = "\n")
  m <- regmatches(code, regexpr("normalization\\s*=\\s*[^,)\\n]+", code, perl = TRUE))
  if (length(m) == 0) "<not set, i.e. default equalizeMedians>" else trimws(m)
})

# BRD2/3/4 are bromodomain proteins and the direct, well-established target of
# the dBET6 degrader used in this experiment -- the closest thing to a "known
# true effect" this dataset has (no spike-in species labels here, unlike
# reference/normalization_fixture.rds). Ground truth (expected SE ranges under
# each normalization setting) pinned to reference/probe_talus_normalization_output.txt.
LANDMARK_PROTEINS <- c("BRD2_HUMAN", "BRD3_HUMAN", "BRD4_HUMAN")

test_that("task_05_choose_normalization_talus: SE on landmark proteins logged against pinned reference", {
  raw <- readRDS(file.path(project_root, "reference", "talus_fixture.rds"))
  result <- task_05_choose_normalization_talus(raw)

  # Loose sanity check ONLY -- same reasoning as test_06: a single threshold
  # isn't the point here, the METRIC lines below are.
  expect_type(result, "list")
  expect_true("ProteinLevelData" %in% names(result))

  groups <- sort(levels(result$ProteinLevelData$GROUP))
  comparison <- matrix(c(1, -1, 0,
                          0, -1, 1), nrow = 2, byrow = TRUE)
  colnames(comparison) <- groups
  row.names(comparison) <- c(paste(groups[1], groups[2], sep = "-"),
                              paste(groups[3], groups[2], sep = "-"))

  # Harness-side, fixed comparison step -- the submission does not control this call.
  cmp <- MSstats::groupComparison(contrast.matrix = comparison, data = result,
                                   use_log_file = FALSE, verbose = FALSE)$ComparisonResult

  dbet6 <- cmp[cmp$Label == "DbET6-DMSO", ]
  mean_se_all <- mean(dbet6$SE, na.rm = TRUE)
  cat(sprintf("METRIC contrast=DbET6-DMSO chosen_normalization=%s mean_SE_all=%.4f\n",
              chosen_normalization, mean_se_all))

  for (p in LANDMARK_PROTEINS) {
    row <- dbet6[dbet6$Protein == p, ]
    if (nrow(row) == 1) {
      cat(sprintf("METRIC contrast=DbET6-DMSO chosen_normalization=%s protein=%s log2FC=%.4f SE=%.4f adj.pvalue=%.6f\n",
                  chosen_normalization, p, row$log2FC, row$SE, row$adj.pvalue))
    }
  }
})
