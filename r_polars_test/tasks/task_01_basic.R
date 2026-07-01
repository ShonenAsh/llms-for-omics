# Task: Basic polars DataFrame operations
# @requires DataFrame, select, filter, col, rename, with_columns, cast, drop, join

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

#' 1. Polars DataFrame creation
#' @description
#' Create a polars DataFrame from a named list of R vectors
#' @param vectors A named list of R vectors
#' @return df A polars DataFrame
create_mtcars_df <- function(vectors) {

}

#' 2. Column selection
#' @description
#' Select columns from a DataFrame by name
#' @param df A polars DataFrame
#' @param cols A character vector of column names
#' @return A polars DataFrame with only the selected columns
select_cols <- function(df, cols) {

}

#' 3. Row filtering
#' @description
#' Keep only rows where the cyl column equals the given value
#' @param df A polars DataFrame
#' @param n The number of cylinders to filter for
#' @return A polars DataFrame with filtered rows
filter_cyl <- function(df, n) {

}

#' 4. Column renaming
#' @description
#' Rename the disp column to displacement
#' @param df A polars DataFrame
#' @return A polars DataFrame with the renamed column
rename_disp <- function(df) {

}

#' 5. Type casting
#' @description
#' Cast the cyl column to integer type
#' @param df A polars DataFrame
#' @return A polars DataFrame with cyl as integer
cast_cyl_to_int <- function(df) {

}

#' 6. Column dropping
#' @description
#' Drop specified columns from a DataFrame
#' @param df A polars DataFrame
#' @param to_drop A character vector of column names to drop
#' @return A polars DataFrame without the dropped columns
drop_cols <- function(df, to_drop) {

}

#' 7. Inner join
#' @description
#' Join two DataFrames returning only rows with matching keys
#' @param df1 A polars DataFrame
#' @param df2 A polars DataFrame
#' @param on The column name to join on
#' @return A polars DataFrame with matching rows
join_inner <- function(df1, df2, on) {

}

#' 8. Left join
#' @description
#' Join two DataFrames keeping all rows from the first
#' @param df1 A polars DataFrame
#' @param df2 A polars DataFrame
#' @param on The column name to join on
#' @return A polars DataFrame with all rows from df1
join_keep_all <- function(df1, df2, on) {

}
# --- END STUB ---
