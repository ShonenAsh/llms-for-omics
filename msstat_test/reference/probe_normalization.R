# Ground-truth probe for the normalization-choice tier (task_06).
# Run once: Rscript reference/probe_normalization.R > reference/probe_normalization_output.txt 2>&1
# Reproduces (independently of msstat-data/'s own vignette) the median-normalization
# pitfall on the spike-in benchmark: constant human background, E. coli spiked at
# 5 increasing levels (A=1x baseline .. E=3x). Ground truth is known by construction:
# every human protein should show log2FC=0 across all contrasts; every E. coli
# protein should show log2FC = log2(spike ratio). This makes correctness
# mechanically checkable via false discovery proportion (FDP), not just structural.
#
# Every number test_06_choose_normalization.R's scoring is pinned against comes from
# this file's output, never from memory or the source vignette's claims.

suppressPackageStartupMessages(library(MSstats))

hr <- function(title) cat("\n\n====", title, "====\n")

raw <- readRDS("reference/normalization_fixture.rds")

# Expected true fold changes, by contrast (spike ratio relative to baseline A)
EXPECTED_FC <- c("E-A" = 3, "D-A" = 2.5, "C-A" = 2, "B-A" = 1.5)

calculateFDP <- function(comparisonResult, comparison, fdp_cutoff = 0.05) {
  cr <- comparisonResult[comparisonResult$Label == comparison & is.na(comparisonResult$issue), ]
  ecoli <- cr[grepl("ECOLI", cr$Protein) & cr$adj.pvalue < fdp_cutoff, ]
  human <- cr[!grepl("ECOLI", cr$Protein) & cr$adj.pvalue < fdp_cutoff, ]
  data.frame(
    contrast = comparison,
    n_ecoli_sig = nrow(ecoli),
    n_human_sig = nrow(human),
    FDP = nrow(human) / (nrow(ecoli) + nrow(human))
  )
}

biasSummary <- function(comparisonResult, comparison) {
  cr <- comparisonResult[comparisonResult$Label == comparison & is.na(comparisonResult$issue), ]
  ecoli_fc <- cr[grepl("ECOLI", cr$Protein), "log2FC"]
  human_fc <- cr[!grepl("ECOLI", cr$Protein), "log2FC"]
  data.frame(
    contrast = comparison,
    expected_ecoli_log2FC = log2(EXPECTED_FC[[comparison]]),
    median_ecoli_log2FC = median(ecoli_fc, na.rm = TRUE),
    median_human_log2FC = median(human_fc, na.rm = TRUE)  # expected 0
  )
}

runFullAnalysis <- function(normalization) {
  hr(paste0("dataProcess(normalization = ", deparse(normalization), ") -- running, ~75s"))
  QuantData <- dataProcess(
    raw, use_log_file = FALSE, verbose = FALSE,
    normalization = normalization,
    MBimpute = FALSE, featureSubset = "topN", n_top_feature = 50
  )

  groups <- sort(levels(QuantData$ProteinLevelData$GROUP))
  comparison <- matrix(c(-1,0,0,0,1,
                          -1,0,0,1,0,
                          -1,0,1,0,0,
                          -1,1,0,0,0), nrow = 4, byrow = TRUE)
  row.names(comparison) <- c("E-A", "D-A", "C-A", "B-A")
  colnames(comparison) <- groups

  hr(paste0("groupComparison(normalization = ", deparse(normalization), ") -- running"))
  result <- groupComparison(contrast.matrix = comparison, data = QuantData,
                             use_log_file = FALSE, verbose = FALSE)

  contrasts <- c("E-A", "D-A", "C-A", "B-A")
  fdp <- do.call(rbind, lapply(contrasts, function(cn)
    calculateFDP(result$ComparisonResult, comparison = cn)))
  bias <- do.call(rbind, lapply(contrasts, function(cn)
    biasSummary(result$ComparisonResult, comparison = cn)))

  hr(paste0("FDP by contrast (normalization = ", deparse(normalization), ")"))
  print(fdp, row.names = FALSE)
  hr(paste0("log2FC bias by contrast (normalization = ", deparse(normalization), ")"))
  print(bias, row.names = FALSE)

  list(fdp = fdp, bias = bias)
}

res_norm <- runFullAnalysis("equalizeMedians")
res_nonorm <- runFullAnalysis(FALSE)

hr("SUMMARY: FDP, equalizeMedians (default) vs FALSE, by contrast")
summary_fdp <- merge(res_norm$fdp, res_nonorm$fdp, by = "contrast",
                      suffixes = c("_equalizeMedians", "_FALSE"))
print(summary_fdp[order(-summary_fdp$FDP_equalizeMedians), c("contrast", "FDP_equalizeMedians", "FDP_FALSE")],
      row.names = FALSE)

hr("probe complete")
