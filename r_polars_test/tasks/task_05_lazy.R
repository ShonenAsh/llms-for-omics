# Task: LazyFrame pipelines with polars
# @requires DataFrame, LazyFrame, scan_parquet, scan_csv, sink_parquet, collect, collect_schema, filter, select, with_columns, group_by, agg, sort, head, join, col, alias, len, sum, dt_hour

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

#' 1. Lazy filter and sink to parquet
#' @description
#' Build a lazy pipeline that scans the input parquet, filters rows
#' whose total_amount is greater than or equal to min_amount, and sinks
#' the result to output_path as a parquet file. Do not materialise the
#' full DataFrame in memory (no intermediate collect()).
#' @param input_path Path to the source Yellow Taxi parquet file
#' @param output_path Path to write the filtered parquet file
#' @param min_amount Minimum total_amount threshold (inclusive)
#' @return Invisibly, output_path
lazy_filter_sink <- function(input_path, output_path, min_amount) {

}

#' 2. Total revenue per borough (lazy)
#' @description
#' Scan both the parquet trip data and the CSV zone lookup as
#' LazyFrames, join them lazily on PULocationID == LocationID, group
#' by Borough, sum total_amount per borough, sort descending by revenue,
#' and only then collect the result into a DataFrame.
#' @param parquet_path Path to the Yellow Taxi parquet
#' @param csv_path Path to the zone lookup CSV
#' @return A polars DataFrame with columns: Borough, total_revenue
lazy_borough_revenue <- function(parquet_path, csv_path) {

}

#' 3. Pickups per hour of day (lazy)
#' @description
#' Build a lazy pipeline that reads the trip parquet, derives the hour
#' of day from tpep_pickup_datetime, groups by hour, counts trips per
#' hour, sorts ascending by hour, and collects the final result.
#' @param input_path Path to the Yellow Taxi parquet
#' @return A polars DataFrame with columns: hour, n_trips
lazy_pickups_per_hour <- function(input_path) {

}

#' 4. Get schema without collecting
#' @description
#' Return the schema (a named list of column-name -> dtype) of the
#' parquet file without materialising any rows. Use scan_parquet and
#' collect_schema() rather than reading the data.
#' @param input_path Path to the Yellow Taxi parquet
#' @return A named list; names are column names, values are polars dtypes
lazy_get_schema <- function(input_path) {

}

#' 5. Top-k trips by fare (lazy, with column pruning)
#' @description
#' Build a lazy pipeline that scans the parquet, sorts trips by
#' fare_amount descending, keeps only the top k rows, projects to
#' just tpep_pickup_datetime, PULocationID, and fare_amount, and
#' collects the result.
#' @param input_path Path to the Yellow Taxi parquet
#' @param k Integer number of top-fare trips to return
#' @return A polars DataFrame with columns:
#'   tpep_pickup_datetime, PULocationID, fare_amount
lazy_top_fares <- function(input_path, k) {

}
# --- END STUB ---
