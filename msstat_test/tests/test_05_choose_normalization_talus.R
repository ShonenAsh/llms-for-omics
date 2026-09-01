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

  # Build the contrast matrix by matching group NAMES explicitly, not by
  # sorted position -- sort() order for mixed-case strings like "DbET6" vs
  # "DMSO" depends on the active locale (confirmed: en_US.UTF-8 sorts
  # DbET6 < DMSO < PF477736, but C/POSIX locale -- the likely container
  # default -- sorts DMSO < DbET6 < PF477736). Indexing by position after
  # sort() silently flips which condition is +1 vs -1 depending on locale;
  # indexing by name is locale-independent.
  groups <- levels(result$ProteinLevelData$GROUP)
  comparison <- matrix(0, nrow = 2, ncol = length(groups),
                        dimnames = list(c("DbET6-DMSO", "PF477736-DMSO"), groups))
  comparison["DbET6-DMSO", "DbET6"] <- 1
  comparison["DbET6-DMSO", "DMSO"] <- -1
  comparison["PF477736-DMSO", "PF477736"] <- 1
  comparison["PF477736-DMSO", "DMSO"] <- -1

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
