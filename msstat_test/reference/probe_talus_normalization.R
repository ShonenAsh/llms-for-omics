# Ground-truth probe for the Talus normalization-choice tier (task_05).
# Run once: Rscript reference/probe_talus_normalization.R > reference/probe_talus_normalization_output.txt 2>&1
#
# Unlike the spike-in benchmark (reference/probe_normalization.R), this dataset
# has no constructed species-label ground truth, so FDP isn't computable here.
# Instead it uses a diagnostic verified this session: compare fold-change (FC)
# and standard error (SE) between normalization settings for a small set of
# known landmark proteins.
#
# BRD2/BRD3/BRD4 are bromodomain proteins and the direct, well-established
# target of the dBET6 degrader used in this experiment (vs DMSO control) --
# about the closest thing to a "known true effect" this dataset has. If
# normalization is appropriate here (as the devs recommend), turning it off
# should barely move these proteins' fold-change estimates but should
# noticeably inflate their standard errors (removing real technical noise
# costs precision, doesn't introduce bias). If normalization were instead
# distorting the result the way it does on the spike-in benchmark, turning it
# off would move the fold-change estimates themselves, not just the SE.
#
# Every number test_05's scoring is pinned against comes from this file's
# output, never from memory.

suppressPackageStartupMessages(library(MSstats))

hr <- function(title) cat("\n\n====", title, "====\n")

raw <- readRDS("reference/talus_fixture.rds")

LANDMARK_PROTEINS <- c("BRD2_HUMAN", "BRD3_HUMAN", "BRD4_HUMAN")

runFullAnalysis <- function(normalization) {
  hr(paste0("dataProcess(normalization = ", deparse(normalization), ")"))
  qd <- dataProcess(raw, use_log_file = FALSE, verbose = FALSE,
                     normalization = normalization, MBimpute = FALSE)

  groups <- sort(levels(qd$ProteinLevelData$GROUP))
  comparison <- matrix(c(1, -1, 0,
                          0, -1, 1), nrow = 2, byrow = TRUE)
  colnames(comparison) <- groups
  row.names(comparison) <- c(paste(groups[1], groups[2], sep = "-"),
                              paste(groups[3], groups[2], sep = "-"))

  cmp <- groupComparison(contrast.matrix = comparison, data = qd,
                          use_log_file = FALSE, verbose = FALSE)$ComparisonResult

  landmark <- cmp[cmp$Protein %in% LANDMARK_PROTEINS & cmp$Label == "DbET6-DMSO",
                  c("Protein", "log2FC", "SE", "adj.pvalue")]
  hr(paste0("Landmark proteins (DbET6-DMSO), normalization = ", deparse(normalization)))
  print(landmark[order(landmark$Protein), ], row.names = FALSE)

  overall_mean_se <- mean(cmp$SE[cmp$Label == "DbET6-DMSO"], na.rm = TRUE)
  cat("\nMean SE across all proteins (DbET6-DMSO):", overall_mean_se, "\n")

  list(landmark = landmark, mean_se = overall_mean_se)
}

res_norm <- runFullAnalysis("equalizeMedians")
res_nonorm <- runFullAnalysis(FALSE)

hr("SUMMARY: landmark protein SE, equalizeMedians (recommended) vs FALSE")
summary_df <- merge(res_norm$landmark, res_nonorm$landmark, by = "Protein",
                     suffixes = c("_equalizeMedians", "_FALSE"))
print(summary_df[, c("Protein", "log2FC_equalizeMedians", "log2FC_FALSE",
                      "SE_equalizeMedians", "SE_FALSE")], row.names = FALSE)

cat("\nMean SE across all proteins: equalizeMedians =", res_norm$mean_se,
    " FALSE =", res_nonorm$mean_se, "\n")

hr("probe complete")
