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
safe_source(file.path(submission_path, "task_05_lazy.R"))

test_that("lazy_filter_sink writes filtered parquet without collecting all rows", {
  src <- tempfile(fileext = ".parquet")
  dst <- tempfile(fileext = ".parquet")
  on.exit(unlink(c(src, dst)), add = TRUE)
  pl$DataFrame(
    total_amount = c(5.0, 20.0, 50.0, 100.0, 200.0),
    id = 1:5
  )$write_parquet(src)

  lazy_filter_sink(src, dst, 50)
  expect_true(file.exists(dst))
  out <- as.data.frame(pl$read_parquet(dst))
  expect_true(all(out$total_amount >= 50))
  expect_equal(sort(as.integer(out$id)), c(3L, 4L, 5L))
})

test_that("lazy_borough_revenue returns per-borough revenue sorted desc", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  csv_path <- file.path(data_path, "taxi_zone_lookup.csv")
  result <- lazy_borough_revenue(parquet_path, csv_path)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("Borough", "total_revenue"))
  r <- as.data.frame(result)
  expect_equal(r$Borough[1], "Manhattan")
  expect_true(all(diff(r$total_revenue) <= 0))
})

test_that("lazy_pickups_per_hour returns 24 hour buckets summing to row count", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  result <- lazy_pickups_per_hour(parquet_path)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns, c("hour", "n_trips"))
  r <- as.data.frame(result)
  expect_equal(nrow(r), 24)
  expect_true(all(diff(as.integer(r$hour)) == 1L))
  expect_equal(min(as.integer(r$hour)), 0L)
  expect_equal(max(as.integer(r$hour)), 23L)
  total_rows <- pl$scan_parquet(parquet_path)$select(pl$len())$collect()
  expect_equal(sum(as.numeric(r$n_trips)),
               as.numeric(as.data.frame(total_rows)[[1]][1]))
})

test_that("lazy_get_schema returns schema without materialising rows", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  result <- lazy_get_schema(parquet_path)
  expect_true("VendorID" %in% names(result))
  expect_true("tpep_pickup_datetime" %in% names(result))
  expect_true("total_amount" %in% names(result))
  expect_equal(length(result), 19)
})

test_that("lazy_top_fares returns k top-fare trips with projected columns", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  result <- lazy_top_fares(parquet_path, 5)
  expect_s3_class(result, "polars_data_frame")
  expect_setequal(result$columns,
                  c("tpep_pickup_datetime", "PULocationID", "fare_amount"))
  expect_equal(result$height, 5)
  r <- as.data.frame(result)
  expect_true(all(diff(r$fare_amount) <= 0))
})
