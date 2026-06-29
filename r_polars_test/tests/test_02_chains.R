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
safe_source(file.path(submission_path, "task_02_chains.R"))

test_that("remove_high_null_cols drops columns exceeding null threshold", {
  df <- pl$DataFrame(
    a = c(1, NA, NA, NA, NA),
    b = c(1, 2, 3, 4, 5),
    c = c(NA, NA, NA, 1, 2),
    d = c(1, 2, 3, 4, 5)
  )
  result <- remove_high_null_cols(df, 0.4)
  expect_equal(result$columns, c("b", "d"))
})

test_that("trip_pickup_borough test understanding of joins", {
  parquet_path <- file.path(data_path, "yellow_tripdata_2024-01.parquet")
  csv_path <- file.path(data_path, "taxi_zone_lookup.csv")
  result <- trip_pickup_borough(parquet_path, csv_path)
  expect_true("Borough" %in% result$columns)
  expect_true("PULocationID" %in% result$columns)
  expect_false("LocationID" %in% result$columns)
  expect_false("Zone" %in% result$columns)
})

test_that("label_gas_guzzlers prepends to low-mpg car names", {
  df <- as_polars_df(mtcars)$with_columns(
    pl$lit(rownames(mtcars))$alias("car")
  )
  result <- label_gas_guzzlers(df, "mpg", "car", 20)

  low <- as.data.frame(result$filter(pl$col("mpg") < 20))$car
  expect_true(all(grepl("^Guzzler-", low)))

  high <- as.data.frame(result$filter(pl$col("mpg") >= 20))$car
  expect_true(!any(grepl("^Guzzler-", high)))
})

test_that("unpivot_df converts wide to long", {
  df <- pl$DataFrame(
    id = 1:3,
    a = c(10, 20, 30),
    b = c(40, 50, 60),
    c = c(70, 80, 90)
  )
  result <- unpivot_df(df, "id", "variable", "value")
  expect_equal(result$columns, c("id", "variable", "value"))
  expect_equal(result$shape[1], 9)
  expect_equal(sort(unique(as.data.frame(result)$variable)), c("a", "b", "c"))
})

test_that("rank_by_group produces correct ranks within groups", {
  df <- pl$DataFrame(
    group = c("x", "x", "x", "y", "y"),
    val   = c(10, 30, 20, 5, 15)
  )
  result <- rank_by_group(df, "val", "group", "rnk")

  x_vals <- result$filter(pl$col("group") == "x")$sort("val", descending = TRUE)
  expect_equal(as.data.frame(x_vals)$rnk, c(1, 2, 3))

  y_vals <- result$filter(pl$col("group") == "y")$sort("val", descending = TRUE)
  expect_equal(as.data.frame(y_vals)$rnk, c(1, 2))
})
