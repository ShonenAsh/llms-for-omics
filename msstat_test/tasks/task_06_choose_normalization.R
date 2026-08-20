# Task: Choose appropriate data-processing settings for a spike-in benchmark
# @requires dataProcess

<<DOCS>>

# Your code here:
# --- STUB ---

#' Summarize this spiked-in control-mixture dataset to protein level.
#' `raw` is MSstats-format data from a benchmark experiment: a constant human
#' proteome background, with an E. coli proteome spiked in at increasing
#' concentrations across 5 conditions (A = 1x baseline, B = 1.5x, C = 2x,
#' D = 2.5x, E = 3x). Choose MSstats data-processing settings appropriate for
#' this experimental design and return the processed object. Do not write log
#' files or print progress (pass use_log_file = FALSE, verbose = FALSE).
#' @param raw an MSstats-format data.frame from the spiked-in benchmark
#' @return the object returned by MSstats' processing step
task_06_choose_normalization <- function(raw) {

}
# --- END STUB ---
