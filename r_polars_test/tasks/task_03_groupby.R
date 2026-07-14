# Task: GroupBy and aggregation with polars
# @requires DataFrame, scan_parquet, scan_csv, read_csv, group_by, agg, col, mean, sum, median, len, quantile, sort, head, with_columns, alias, join

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

#' 1. Trip count per pickup borough
#' @description
#' Read the Yellow Taxi trip parquet and the NYC zone-lookup CSV, join
#' each trip's PULocationID to its Borough, and count the trips per
#' Borough. The returned DataFrame should be sorted descending by count.
#' @param parquet_path Path to the Yellow Taxi trip parquet
#' @param csv_path Path to the NYC zone-lookup CSV
#'      (columns: LocationID, Borough, Zone, service_zone)
#' @return A polars DataFrame with columns: Borough (String), n_trips (integer),
#'      sorted descending by n_trips
trips_per_borough <- function(parquet_path, csv_path) {

}

#' 2. Average fare by passenger count
#' @description
#' Group by passenger_count and compute the mean fare_amount per group.
#' @param df A polars DataFrame containing at least passenger_count and fare_amount
#' @return A polars DataFrame with columns: passenger_count, avg_fare
avg_fare_by_passenger <- function(df) {

}

#' 3. Multiple aggregations per payment type
#' @description
#' Group by payment_type and compute four statistics in a single agg call:
#' the mean tip_amount, median trip_distance, sum of total_amount, and
#' number of trips.
#' @param df A polars DataFrame with columns: payment_type, tip_amount,
#'      trip_distance, total_amount
#' @return A polars DataFrame with columns: payment_type, mean_tip,
#'      median_distance, total_revenue, n_trips
agg_multi_stats_by_payment <- function(df) {

}

#' 4. Top-k pickup zones by revenue
#' @description
#' Join the trips DataFrame with the zone lookup CSV, group by Zone,
#' sum total_amount per zone, and return the k zones with the highest
#' revenue (sorted descending by revenue).
#' @param df A polars DataFrame of trips with columns PULocationID and total_amount
#' @param csv_path Path to the NYC zone-lookup CSV
#' @param k Integer number of zones to return
#' @return A polars DataFrame with columns: Zone, total_revenue
top_k_zones_by_revenue <- function(df, csv_path, k) {

}

#' 5. Fare quantile per borough
#' @description
#' Join trips with the zone-lookup CSV and compute the q-th quantile of
#' fare_amount per pickup Borough.
#' @param df A polars DataFrame of trips with columns PULocationID and fare_amount
#' @param csv_path Path to the NYC zone-lookup CSV
#' @param q A numeric quantile between 0 and 1
#' @return A polars DataFrame with columns: Borough, fare_quantile
quantile_fare_by_borough <- function(df, csv_path, q) {

}

#' 6. Share of trips per payment type
#' @description
#' Group by payment_type and return both the raw trip count per group
#' and its share of the total number of trips (share values sum to 1).
#' @param df A polars DataFrame with a payment_type column
#' @return A polars DataFrame with columns: payment_type, n_trips, share
payment_type_share <- function(df) {

}
# --- END STUB ---
