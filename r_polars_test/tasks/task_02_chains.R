# Task: Chained polars operations
# @requires scan_parquet, scan_csv, select, join, with_columns, drop, null_count, get_column, to_r_vector, when, lit, cast, add, alias, unpivot, rank, over, col

# Main prompt:
# Implement the following polars operations:
# 1. Drop columns where the proportion of null values exceeds a threshold
# 2. Join NYC taxi trip data with zone lookup to add pickup borough
# 3. Prepend "Guzzler-" to car names where mpg is below threshold
# 4. Unpivot (melts) wide data to long format
# 5. Rank a column descending within groups

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

remove_high_null_cols <- function(df, threshold) {

}

trip_pickup_borough <- function(parquet_path, csv_path) {

}

label_gas_guzzlers <- function(df, mpg_col, name_col, threshold) {

}

unpivot_df <- function(df, id_cols, names_to, values_to) {

}

rank_by_group <- function(df, value_col, group_col, new_col) {

}
# --- END STUB ---
