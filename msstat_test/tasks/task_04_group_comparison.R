# Task: Compare protein abundance between groups
# @requires groupComparison

<<DOCS>>

# Your code here:
# --- STUB ---

#' Compare protein abundance between groups.
#' Given a processed MSstats object `processed` (as returned by dataProcess())
#' and a contrast matrix `contrast_matrix`, run MSstats' group-comparison step
#' and return the resulting comparison results table (one row per protein per
#' contrast). Do not write log files or print progress (pass
#' use_log_file = FALSE, verbose = FALSE).
#' @param processed the object returned by dataProcess()
#' @param contrast_matrix a numeric contrast matrix (see MSstats::groupComparison)
#' @return a data.frame of comparison results
task_04_group_comparison <- function(processed, contrast_matrix) {

}
# --- END STUB ---
