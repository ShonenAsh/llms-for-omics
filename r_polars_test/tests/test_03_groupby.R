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
safe_source(file.path(submission_path, "task_03_groupby.R"))

test_that("trips_per_borough counts trips per pickup borough", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  csv_path <- file.path(data_path, "taxi_zone_lookup.csv")
  result <- trips_per_borough(parquet_path, csv_path)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("Borough", "n_trips"))
  r <- as.data.frame(result)
  expect_true(all(diff(r$n_trips) <= 0))
  expect_equal(r$Borough[1], "Manhattan")
})

test_that("avg_fare_by_passenger computes correct group means", {
  df <- pl$DataFrame(
    passenger_count = c(1L, 1L, 2L, 2L, 3L),
    fare_amount = c(10.0, 20.0, 5.0, 15.0, 30.0)
  )
  result <- avg_fare_by_passenger(df)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("passenger_count", "avg_fare"))
  r <- as.data.frame(result)
  r <- r[order(r$passenger_count), ]
  expect_equal(r$avg_fare, c(15.0, 10.0, 30.0))
})

test_that("agg_multi_stats_by_payment produces all four statistics", {
  df <- pl$DataFrame(
    payment_type = c(1L, 1L, 2L, 2L, 2L),
    tip_amount = c(1.0, 3.0, 2.0, 4.0, 6.0),
    trip_distance = c(1.0, 3.0, 2.0, 4.0, 6.0),
    total_amount = c(10.0, 20.0, 30.0, 40.0, 50.0)
  )
  result <- agg_multi_stats_by_payment(df)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns,
                  c("payment_type", "mean_tip", "median_distance",
                    "total_revenue", "n_trips"))
  r <- as.data.frame(result)
  r <- r[order(r$payment_type), ]
  expect_equal(r$mean_tip, c(2.0, 4.0))
  expect_equal(r$median_distance, c(2.0, 4.0))
  expect_equal(r$total_revenue, c(30.0, 120.0))
  expect_equal(as.integer(r$n_trips), c(2L, 3L))
})

test_that("top_k_zones_by_revenue returns k zones sorted by revenue", {
  # LocationID 1 -> "Newark Airport", 2 -> "Jamaica Bay",
  # 3 -> "Allerton/Pelham Gardens", 4 -> "Alphabet City"
  trips <- pl$DataFrame(
    PULocationID = c(1L, 1L, 2L, 3L, 4L, 4L),
    total_amount = c(100.0, 50.0, 20.0, 10.0, 5.0, 5.0)
  )
  csv_path <- file.path(data_path, "taxi_zone_lookup.csv")
  result <- top_k_zones_by_revenue(trips, csv_path, 3)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("Zone", "total_revenue"))
  r <- as.data.frame(result)
  expect_equal(nrow(r), 3)
  expect_equal(r$Zone[1], "Newark Airport")
  expect_equal(r$total_revenue[1], 150.0)
  expect_true(all(diff(r$total_revenue) <= 0))
})

test_that("quantile_fare_by_borough computes quantile per borough", {
  # 1 -> EWR, 2 -> Queens, 4 -> Manhattan. Odd-length groups so median
  # is unambiguous across interpolation modes.
  trips <- pl$DataFrame(
    PULocationID = c(1L, 1L, 1L, 2L, 2L, 2L, 4L, 4L, 4L),
    fare_amount = c(10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 4.0, 8.0, 12.0)
  )
  csv_path <- file.path(data_path, "taxi_zone_lookup.csv")
  result <- quantile_fare_by_borough(trips, csv_path, 0.5)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("Borough", "fare_quantile"))
  r <- as.data.frame(result)
  medians <- setNames(r$fare_quantile, r$Borough)
  expect_equal(unname(medians["EWR"]), 20.0)
  expect_equal(unname(medians["Queens"]), 15.0)
  expect_equal(unname(medians["Manhattan"]), 8.0)
})

test_that("payment_type_share returns counts and shares summing to 1", {
  df <- pl$DataFrame(
    payment_type = c(1L, 1L, 1L, 1L, 2L, 2L, 3L, 4L, 4L, 4L)
  )
  result <- payment_type_share(df)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("payment_type", "n_trips", "share"))
  r <- as.data.frame(result)
  expect_equal(sum(as.integer(r$n_trips)), 10L)
  expect_equal(sum(r$share), 1.0, tolerance = 1e-9)
  by_type <- setNames(r$share, r$payment_type)
  expect_equal(unname(by_type["1"]), 0.4, tolerance = 1e-9)
  expect_equal(unname(by_type["4"]), 0.3, tolerance = 1e-9)
})
