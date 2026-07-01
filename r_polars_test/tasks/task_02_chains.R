# Task: Chained polars operations
# @requires scan_parquet, scan_csv, select, join, with_columns, drop, null_count, get_column, to_r_vector, when, lit, cast, add, alias, unpivot, rank, over, col

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

#' 1. Remove high-null columns
#' @description
#' Drop columns where the proportion of null values exceeds the threshold
#' @param df A polars DataFrame
#' @param threshold A numeric threshold between 0 and 1
#' @return A polars DataFrame with high-null columns removed
remove_high_null_cols <- function(df, threshold) {

}

#' 2. Join NYC taxi trip data with zone lookup
#' @description
#' Read data from a CSV containing NYC Zones and a parquet containing
#' Yellow Taxi trip data. Return a single polars DataFrame containing
#' all trip details along with the pickup borough in NYC.
#' @param csv_path path to CSV containing NYC Zone information
#'      Columns (4): "LocationID", "Borough", "Zone", "service_zone"
#' @param parquet_path path to the parquet file containing Yellow Taxi trip data
#'      Columns (19): "VendorID", "tpep_pickup_datetime",
#'      "tpep_dropoff_datetime", "passenger_count", "trip_distance",
#'      "RatecodeID", "store_and_fwd_flag", "PULocationID",
#'      "DOLocationID", "payment_type", "fare_amount", "extra",
#'      "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
#'      "total_amount", "congestion_surcharge", "Airport_fee"
#' @return A polars DataFrame with ALL trip data and their pickup "Borough"
trip_pickup_borough <- function(parquet_path, csv_path) {

}

#' 3. Conditional update with string manipulation
#' @description
#' Prepend "Guzzler-" to name_col values where mpg_col is below the threshold
#' @param df A polars DataFrame
#' @param mpg_col The name of the mpg column
#' @param name_col The name of the car name column
#' @param threshold The mpg threshold
#' @return A polars DataFrame with updated name_col values
label_gas_guzzlers <- function(df, mpg_col, name_col, threshold) {

}

#' 4. Pivot from wide to long
#' @description
#' Unpivot non-id columns into key-value pairs
#' @param df A polars DataFrame
#' @param id_cols Column name(s) to keep as identifiers
#' @param names_to Name for the new key column
#' @param values_to Name for the new value column
#' @return A polars DataFrame in long format
unpivot_df <- function(df, id_cols, names_to, values_to) {

}

#' 5. Window function rank
#' @description
#' Rank value_col descending within each group defined by group_col
#' @param df A polars DataFrame
#' @param value_col The column to rank
#' @param group_col The column to group by
#' @param new_col The name for the rank column
#' @return A polars DataFrame with the rank column appended
rank_by_group <- function(df, value_col, group_col, new_col) {

}
# --- END STUB ---
