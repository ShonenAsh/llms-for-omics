# One-time fixture prep for the normalization-choice tier (task_06).
# Run once: Rscript reference/prepare_normalization_fixture.R
#
# Converts the No-MBR Metamorpheus spike-in benchmark (msstat-data/, ~99MB raw CSV,
# not committed -- see .gitignore) into MSstats format ONE TIME and caches the ~4MB
# result to reference/normalization_fixture.rds. The LLM's task stub receives this
# pre-converted table directly (never the raw CSV), exactly like DIARawData in
# tasks 03/04 -- so per-run grading only pays for dataProcess(), not for reloading
# or reconverting 342k rows of raw search-engine output.
#
# MBR = FALSE is required for this No-MBR data -- confirmed by reproducing the
# vignette's own helper function verbatim (msstat-data/.../For_Ashish.Rmd) and
# hitting "A non-empty vector of column names for `by` is required." without it.
# Even the domain experts' own case-study script is missing this argument.

suppressPackageStartupMessages({
  library(MSstats)
  library(MSstatsConvert)
})

RAW_DIR <- "msstat-data/10-DDA-Control-Mixtures-Metamorpheus/No-MBR"
OUT_FILE <- "reference/normalization_fixture.rds"

stopifnot(
  "Raw data not found -- see msstat_test/msstat-data/ (not committed, ask for the source data)" =
    dir.exists(RAW_DIR)
)

input <- data.table::fread(file.path(RAW_DIR, "QuantifiedPeaks.csv"))
annot <- data.table::fread(file.path(RAW_DIR, "annotation.csv"))

msstats_fmt <- MetamorpheusToMSstatsFormat(
  input, annot,
  use_log_file = FALSE,
  removeFewMeasurements = FALSE,
  removeProtein_with1Feature = FALSE,
  MBR = FALSE
)

cat("Converted dim:", paste(dim(msstats_fmt), collapse = " x "), "\n")
cat("Columns:", paste(colnames(msstats_fmt), collapse = ", "), "\n")

species <- ifelse(grepl("ECOLI", msstats_fmt$ProteinName), "ECOLI",
            ifelse(grepl("HUMAN", msstats_fmt$ProteinName), "HUMAN", "OTHER"))
cat("Unique proteins by species:\n")
print(table(species[!duplicated(msstats_fmt$ProteinName)]))

saveRDS(msstats_fmt, OUT_FILE)
cat("Written ->", OUT_FILE, "\n")
