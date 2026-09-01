# One-time fixture prep for the Talus normalization-choice tier (task_05).
# Run once: Rscript reference/prepare_talus_fixture.R
#
# Converts the Talus DIA-NN chromatin-proteomics dataset (msstat-data/, not
# committed -- see .gitignore) into MSstats format ONE TIME and caches the
# ~300KB result to reference/talus_fixture.rds. The LLM's task stub receives
# this pre-converted table directly, never the raw DIA-NN report.
#
# This is the dataset the devs recommend KEEPING equalizeMedians on for
# (confirmed independently -- see reference/probe_talus_normalization.R):
# only 3 of 32 proteins in this targeted chromatin panel (BRD2/3/4) are real
# drug targets, not enough to distort the per-sample median, so normalization
# correctly removes real run-to-run technical noise instead of introducing
# bias. It's the counterpart to reference/normalization_fixture.rds (the
# spike-in benchmark, where normalization should be OFF).

suppressPackageStartupMessages({
  library(MSstats)
  library(MSstatsConvert)
})

RAW_DIR <- "msstat-data/High-DIA-Talus-Perturbations-DIANN"
OUT_FILE <- "reference/talus_fixture.rds"

stopifnot(
  "Raw data not found -- see msstat_test/msstat-data/ (not committed, ask for the source data)" =
    dir.exists(RAW_DIR)
)

input_data <- read.csv(file.path(RAW_DIR, "diann_report_may_institute.csv"))
annotation_file <- read.csv(file.path(RAW_DIR, "annot_may_institute.csv"))

# Same converter call as the vignette's own Monday-label_free_case_study.R.
msstats_format <- DIANNtoMSstatsFormat(
  input_data, annotation = annotation_file,
  global_qvalue_cutoff = 0.01, qvalue_cutoff = 0.01, pg_qvalue_cutoff = 0.01,
  useUniquePeptide = TRUE, removeFewMeasurements = TRUE,
  removeOxidationMpeptides = TRUE, removeProtein_with1Feature = TRUE,
  MBR = FALSE, use_log_file = FALSE, verbose = FALSE
)
msstats_format <- as.data.frame(msstats_format)

cat("Converted dim:", paste(dim(msstats_format), collapse = " x "), "\n")
cat("Unique proteins:", length(unique(msstats_format$ProteinName)), "\n")
cat("Unique runs:", length(unique(msstats_format$Run)), "\n")
cat("Conditions:", paste(sort(unique(msstats_format$Condition)), collapse = ", "), "\n")

saveRDS(msstats_format, OUT_FILE)
cat("Written ->", OUT_FILE, "\n")
