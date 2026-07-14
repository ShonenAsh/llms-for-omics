library("testthat")
library("polars")

data_path <- Sys.getenv("DATA_PATH")
submission_path <- Sys.getenv("SUBMISSION_DIR", unset = normalizePath(file.path("..", "tasks")))
safe_source <- function(file) {
  lines <- readLines(file, warn = FALSE)
  result <- character()
  depth <- 0L
  in_test <- FALSE
  for (line in lines) {
    if (grepl("^\\s*test_that\\(", line)) {
      in_test <- TRUE
      depth <- depth + nchar(gsub("[^{]", "", line)) - nchar(gsub("[^}]", "", line))
      if (depth <= 0L) in_test <- FALSE
      next
    }
    if (in_test) {
      depth <- depth + nchar(gsub("[^{]", "", line)) - nchar(gsub("[^}]", "", line))
      if (depth <= 0L) { in_test <- FALSE; depth <- 0L }
      next
    }
    result <- c(result, line)
  }
  source(textConnection(result), local = globalenv())
}
safe_source(file.path(submission_path, "task_04_strings_dates.R"))

test_that("add_pickup_hour extracts hour-of-day", {
  df <- pl$DataFrame(
    tpep_pickup_datetime = as.POSIXct(
      c("2024-01-01 08:15:00", "2024-01-01 23:45:00", "2024-01-02 05:30:00"),
      tz = "UTC"
    )
  )
  result <- add_pickup_hour(df)
  expect_s3_class(result, "polars_data_frame")
  expect_true("pickup_hour" %in% result$columns)
  expect_equal(as.integer(as.data.frame(result)$pickup_hour), c(8L, 23L, 5L))
})

test_that("trip_duration_minutes computes correct duration", {
  df <- pl$DataFrame(
    tpep_pickup_datetime  = as.POSIXct(
      c("2024-01-01 08:00:00", "2024-01-01 09:00:00"), tz = "UTC"),
    tpep_dropoff_datetime = as.POSIXct(
      c("2024-01-01 08:15:00", "2024-01-01 10:30:00"), tz = "UTC")
  )
  result <- trip_duration_minutes(df)
  expect_true("duration_min" %in% result$columns)
  expect_equal(as.numeric(as.data.frame(result)$duration_min), c(15, 90))
})

test_that("filter_weekend_pickups keeps only Sat and Sun", {
  # 2024-01-05 Fri, 2024-01-06 Sat, 2024-01-07 Sun, 2024-01-08 Mon
  df <- pl$DataFrame(
    tpep_pickup_datetime = as.POSIXct(
      c("2024-01-05 12:00:00", "2024-01-06 12:00:00",
        "2024-01-07 12:00:00", "2024-01-08 12:00:00"), tz = "UTC"),
    id = 1:4
  )
  result <- suppressWarnings(filter_weekend_pickups(df))
  expect_s3_class(result, "polars_data_frame")
  expect_equal(sort(as.integer(as.data.frame(result)$id)), c(2L, 3L))
})

test_that("hourly_trip_counts aggregates trips per hour", {
  df <- pl$DataFrame(
    tpep_pickup_datetime = as.POSIXct(
      c("2024-01-01 08:00:00", "2024-01-01 08:30:00",
        "2024-01-01 09:00:00", "2024-01-01 09:45:00",
        "2024-01-01 09:55:00"), tz = "UTC")
  )
  result <- hourly_trip_counts(df)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("hour", "n_trips"))
  r <- as.data.frame(result)
  r <- r[order(as.integer(r$hour)), ]
  expect_equal(as.integer(r$hour), c(8L, 9L))
  expect_equal(as.integer(r$n_trips), c(2L, 3L))
})

test_that("flag_airport_zones detects Airport substring", {
  df <- pl$DataFrame(
    Zone = c("JFK Airport", "Newark Airport", "Alphabet City",
             "LaGuardia Airport", "Times Sq")
  )
  result <- flag_airport_zones(df)
  expect_true("is_airport" %in% result$columns)
  expect_equal(as.logical(as.data.frame(result)$is_airport),
               c(TRUE, TRUE, FALSE, TRUE, FALSE))
})

test_that("split_zone_hierarchy keeps first segment before /", {
  df <- pl$DataFrame(
    Zone = c("Allerton/Pelham Gardens", "Manhattan", "Foo/Bar/Baz", "SoHo")
  )
  result <- split_zone_hierarchy(df)
  expect_true("zone_primary" %in% result$columns)
  expect_equal(as.character(as.data.frame(result)$zone_primary),
               c("Allerton", "Manhattan", "Foo", "SoHo"))
})

test_that("format_pickup_date formats as YYYY-MM-DD", {
  df <- pl$DataFrame(
    tpep_pickup_datetime = as.POSIXct(
      c("2024-01-01 08:15:00", "2024-05-20 23:59:00"), tz = "UTC")
  )
  result <- format_pickup_date(df)
  expect_true("pickup_date_str" %in% result$columns)
  expect_equal(as.character(as.data.frame(result)$pickup_date_str),
               c("2024-01-01", "2024-05-20"))
})
