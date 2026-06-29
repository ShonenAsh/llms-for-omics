# Task: Basic polars DataFrame operations
# @requires DataFrame, select, filter, col, rename, with_columns, cast, drop, join

# Main prompt:
# Implement the following polars DataFrame operations:
# 1. Create a polars DataFrame from a named list of R vectors
# 2. Select columns by name from a DataFrame
# 3. Filter rows where the cyl column equals a given value
# 4. Rename the disp column to displacement
# 5. Cast the cyl column to integer type
# 6. Drop specified columns from a DataFrame
# 7. Inner join two DataFrames on a key column
# 8. Left join two DataFrames on a key column

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

create_mtcars_df <- function(vectors) {
  
}

select_cols <- function(df, cols) {
  
}

filter_cyl <- function(df, n) {
  
}

rename_disp <- function(df) {
  
}

cast_cyl_to_int <- function(df) {
  
}

drop_cols <- function(df, to_drop) {
  
}

join_inner <- function(df1, df2, on) {
  
}

join_keep_all <- function(df1, df2, on) {
  
}
# --- END STUB ---
