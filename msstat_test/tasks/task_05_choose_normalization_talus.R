# Task: Choose appropriate data-processing settings for a chromatin-proteomics
# drug-perturbation experiment
# @requires dataProcess

<<DOCS>>

# Your code here:
# --- STUB ---

#' Summarize this chromatin-proteomics drug-perturbation dataset to protein level.
#' `raw` is MSstats-format DIA data from an experiment in THP-1 cells: cells
#' were treated with one of three conditions (DMSO vehicle control, or one of
#' two small-molecule compounds, DbET6 or PF477736) for 4 hours, then
#' chromatin-bound proteins were enriched with an automated pulldown method
#' and quantified. Choose MSstats data-processing settings appropriate for
#' this experimental design and return the processed object. Do not write log
#' files or print progress (pass use_log_file = FALSE, verbose = FALSE).
#' @param raw an MSstats-format data.frame from the chromatin pulldown experiment
#' @return the object returned by MSstats' processing step
task_05_choose_normalization_talus <- function(raw) {

}
# --- END STUB ---
