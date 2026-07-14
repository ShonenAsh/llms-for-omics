# Task: String and datetime operations with polars
# @requires DataFrame, with_columns, col, alias, filter, group_by, agg, len, sort, cast, is_in, dt_hour, dt_weekday, dt_total_minutes, dt_strftime, str_contains, str_split, list_first

<<DOCS>>

# Your code here:
# --- STUB ---
library(polars)

#' 1. Add a pickup hour column
#' @description
#' Extract the hour-of-day (0..23) from the tpep_pickup_datetime column
#' and add it as a new column named pickup_hour.
#' @param df A polars DataFrame with a tpep_pickup_datetime column
#' @return A polars DataFrame with an additional integer column pickup_hour
add_pickup_hour <- function(df) {

}

#' 2. Trip duration in minutes
#' @description
#' Compute the difference tpep_dropoff_datetime - tpep_pickup_datetime as
#' the total number of minutes and add it as a numeric column named
#' duration_min.
#' @param df A polars DataFrame with tpep_pickup_datetime and
#'      tpep_dropoff_datetime columns
#' @return A polars DataFrame with an additional numeric column duration_min
trip_duration_minutes <- function(df) {

}

#' 3. Filter to weekend pickups only
#' @description
#' Keep only rows whose tpep_pickup_datetime falls on a Saturday or
#' Sunday. The polars weekday function returns 1 for Monday through 7
#' for Sunday.
#' @param df A polars DataFrame with a tpep_pickup_datetime column
#' @return A polars DataFrame containing only weekend pickups
filter_weekend_pickups <- function(df) {

}

#' 4. Trip counts per pickup hour
#' @description
#' Group trips by the hour-of-day of tpep_pickup_datetime and return
#' the number of trips per hour, sorted ascending by hour.
#' @param df A polars DataFrame with a tpep_pickup_datetime column
#' @return A polars DataFrame with columns: hour (integer 0..23),
#'      n_trips (integer), sorted ascending by hour
hourly_trip_counts <- function(df) {

}

#' 5. Flag airport zones
#' @description
#' Add a boolean column is_airport that is TRUE whenever the Zone
#' column contains the substring "Airport" (case-sensitive).
#' @param df A polars DataFrame with a Zone string column
#' @return A polars DataFrame with an additional boolean column is_airport
flag_airport_zones <- function(df) {

}

#' 6. Extract primary zone from hierarchy
#' @description
#' Split each Zone value on the "/" character and keep only the first
#' segment as a new column named zone_primary. For example
#' "Allerton/Pelham Gardens" -> "Allerton"; "Manhattan" (no "/") stays
#' as "Manhattan".
#' @param df A polars DataFrame with a Zone string column
#' @return A polars DataFrame with an additional string column zone_primary
split_zone_hierarchy <- function(df) {

}

#' 7. Format pickup date as ISO string
#' @description
#' Format tpep_pickup_datetime as an ISO-style date string
#' ("YYYY-MM-DD") and add it as a new column named pickup_date_str.
#' @param df A polars DataFrame with a tpep_pickup_datetime column
#' @return A polars DataFrame with an additional string column pickup_date_str
format_pickup_date <- function(df) {

}
# --- END STUB ---
