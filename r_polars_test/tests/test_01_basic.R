library("testthat")
library("polars")

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
safe_source(file.path(submission_path, "task_01_basic.R"))

test_that("create_mtcars_df returns correct DataFrame", {
  vectors <- as.list(mtcars)
  result <- create_mtcars_df(vectors)
  expect_s3_class(result, "polars_data_frame")
  expect_equal(result$shape, c(32, 11))
  expect_true(result$equals(as_polars_df(mtcars)))
})

test_that("select_cols returns only requested columns", {
  df <- as_polars_df(mtcars)
  result <- select_cols(df, c("mpg", "hp", "wt"))
  expect_equal(result$columns, c("mpg", "hp", "wt"))
  expected <- mtcars[, c("mpg", "hp", "wt")]
  row.names(expected) <- NULL
  expect_equal(as.data.frame(result), expected, ignore_attr = TRUE)
})

test_that("filter_cyl keeps only rows with matching cyl", {
  df <- as_polars_df(mtcars)
  result <- filter_cyl(df, 6)
  expected <- mtcars[mtcars$cyl == 6, ]
  row.names(expected) <- NULL
  expect_equal(as.data.frame(result), expected, ignore_attr = TRUE)
})

test_that("rename_disp renames disp to displacement", {
  df <- as_polars_df(mtcars)
  result <- rename_disp(df)
  expect_true("displacement" %in% result$columns)
  expect_false("disp" %in% result$columns)
  expect_equal(result$width, 11)
})

test_that("cast_cyl_to_int changes cyl type to integer", {
  df <- as_polars_df(mtcars)
  result <- cast_cyl_to_int(df)
  types <- result$dtypes
  cyl_idx <- which(result$columns == "cyl")
  expect_true(grepl("Int32|Int64", format(types[[cyl_idx]])))
})

test_that("drop_cols removes specified columns", {
  df <- as_polars_df(mtcars)
  result <- drop_cols(df, c("carb", "drat"))
  expect_false("carb" %in% result$columns)
  expect_false("drat" %in% result$columns)
  expect_equal(result$width, 9)
})

test_that("join_inner returns only matching rows", {
  left <- pl$DataFrame(key = 1:3, val_left = c("a", "b", "c"))
  right <- pl$DataFrame(key = 2:4, val_right = c("x", "y", "z"))
  result <- join_inner(left, right, "key")
  expect_equal(as.data.frame(result)$key, c(2, 3))
  expect_equal(result$shape[1], 2)
})

test_that("join_keep_all preserves all rows from left", {
  left <- pl$DataFrame(key = 1:3, val_left = c("a", "b", "c"))
  right <- pl$DataFrame(key = 2:4, val_right = c("x", "y", "z"))
  result <- join_keep_all(left, right, "key")
  keys <- as.data.frame(result)$key
  expect_equal(keys, c(1, 2, 3))
  expect_equal(result$shape[1], 3)
  expect_true(is.na(as.data.frame(result)$val_right[1]))
})
