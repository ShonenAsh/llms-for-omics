#!/usr/bin/env Rscript
# build_docs.R -- inject polars Rd docs into task templates

suppressWarnings(suppressMessages(library(tools)))

# CONFIG
PKG        <- "polars"
TASKS_DIR  <- "tasks"          # *.R task templates, each with @requires + <<DOCS>>
OUT_DIR    <- "docs_conditions"
CAPTURE_OUTPUT    <- TRUE       # run doc examples to capture printed I/O?
EXAMPLE_SPLIT     <- "\n[ \t]*\n"   # blank-line split for the "one example" rung
COMMENT           <- "# "       # injected docs are commented so the file parses

NAMESPACE_TOKENS <- c("str","dt","list","struct","bin","cat","arr",
                      "name","meta","bool")
DTYPE_TOKENS <- c("Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32",
                  "UInt64","Float32","Float64","Boolean","String","Utf8",
                  "Categorical","Date","Datetime","Duration","Time","Struct",
                  "List","Binary","Null")

# DOC EXTRACTION
load_doc_db <- function(pkg = PKG) {
  tryCatch(Rd_db(pkg), error = function(e)
    stop("Rd_db('", pkg, "') failed -- is the package installed? ",
         conditionMessage(e), call. = FALSE))
}

.rd_flatten <- function(x) {
  if (is.null(x)) return("")
  if (is.character(x)) return(paste(x, collapse = ""))
  if (is.list(x)) return(paste(vapply(x, .rd_flatten, character(1)), collapse = ""))
  as.character(x)
}

rd_section <- function(topic, tag) {
  hits <- Filter(function(e) identical(attr(e, "Rd_tag"), tag), topic)
  if (!length(hits)) return("")
  trimws(paste(vapply(hits, .rd_flatten, character(1)), collapse = "\n"))
}

topic_aliases <- function(topic) {
  hits <- Filter(function(e) identical(attr(e, "Rd_tag"), "\\alias"), topic)
  trimws(vapply(hits, .rd_flatten, character(1)))
}

build_doc_records <- function(db = load_doc_db()) {
  records <- list()
  for (nm in names(db)) {
    topic <- db[[nm]]
    rec <- list(
      usage       = rd_section(topic, "\\usage"),
      description = rd_section(topic, "\\description"),
      arguments   = rd_section(topic, "\\arguments"),
      value       = rd_section(topic, "\\value"),
      examples    = rd_section(topic, "\\examples")
    )
    for (al in topic_aliases(topic)) records[[al]] <- rec
  }
  records
}

# RESOLUTION
read_required <- function(task_file) {
  lines <- readLines(task_file, warn = FALSE)
  hit <- grep("@requires", lines, value = TRUE)
  if (!length(hit)) stop("No @requires line in ", task_file, call. = FALSE)
  raw  <- sub(".*@requires", "", hit[1])
  toks <- strsplit(raw, "[,[:space:]]+")[[1]]
  unique(toks[nzchar(toks)])
}

resolve_alias <- function(fun, records) {
  if (!is.null(records[[fun]])) return(fun)
  rec_names <- names(records)
  rec_lower <- tolower(rec_names)
  fun_lower <- tolower(fun)

  if (!grepl("_", fun_lower)) {
    # Bare token: match __fun or exact fun
    pattern <- paste0("(^|__)", fun_lower, "$")
    return(rec_names[grep(pattern, rec_lower, value = FALSE)])
  }

  # Token contains underscore: could be qualified (Class_method) or bare
  # (e.g. cum_sum). Distinguish by checking whether the prefix is a known
  # class/namespace in the alias database.
  parts <- strsplit(fun_lower, "_", fixed = TRUE)[[1]]
  prefix <- parts[1]
  suffix <- paste(parts[-1], collapse = "_")

  is_known_prefix <- any(grepl(paste0("^", prefix, "__"), rec_lower))

  if (is_known_prefix) {
    # Qualified: prefix__suffix
    pattern <- paste0("^", prefix, "__", suffix, "$")
    return(rec_names[grep(pattern, rec_lower, value = FALSE)])
  } else {
    # Bare token with underscore in the name (e.g. cum_sum)
    pattern <- paste0("(^|__)", fun_lower, "$")
    return(rec_names[grep(pattern, rec_lower, value = FALSE)])
  }
}

collect_dollar <- function(e, acc = character()) {
  if (is.call(e)) {
    if (identical(e[[1]], as.name("$"))) {
      acc <- c(acc, as.character(e[[3]]))
      acc <- collect_dollar(e[[2]], acc)
    } else for (i in seq_along(e)) acc <- collect_dollar(e[[i]], acc)
  }
  acc
}
used_funs <- function(code_text)
  setdiff(unique(unlist(lapply(parse(text = code_text), collect_dollar))),
          NAMESPACE_TOKENS)

# OPTIONAL I/O CAPTURE
.output_cache <- new.env(parent = emptyenv())

capture_example_output <- function(alias, code) {
  if (!CAPTURE_OUTPUT || !nzchar(code)) return("")
  if (!is.null(.output_cache[[alias]])) return(.output_cache[[alias]])
  if (!requireNamespace("callr", quietly = TRUE)) {
    warning("callr missing; skipping output capture."); return("")
  }
  out <- tryCatch(
    callr::r(function(src) {
      suppressMessages(library(polars))
      buf <- character()
      con <- textConnection("buf", "w", local = TRUE)
      sink(con); on.exit({ sink(); close(con) }, add = TRUE)
      eval(parse(text = src), envir = new.env())
      paste(buf, collapse = "\n")
    }, args = list(src = code), timeout = 30),
    error = function(e) "")
  .output_cache[[alias]] <- out
  out
}

# DTYPE / SCHEMA STRIPPING  (2b ablation)
strip_dtypes <- function(txt) {
  if (!nzchar(txt)) return(txt)
  lines <- strsplit(txt, "\n", fixed = TRUE)[[1]]
  drop  <- logical(length(lines))
  for (i in seq_along(lines)) {
    if (grepl("shape:\\s*\\(", lines[i])) drop[i] <- TRUE
    if (grepl("^[^[:alnum:]]*---[^[:alnum:]]*$", lines[i])) {  # the `---` row
      drop[i] <- TRUE
      if (i < length(lines)) drop[i + 1L] <- TRUE             # dtype row beneath
    }
  }
  kept <- paste(lines[!drop], collapse = "\n")
  for (tok in DTYPE_TOKENS)
    kept <- gsub(paste0("\\b", tok, "\\b"), "____", kept)
  kept
}

# CONDITION SPECS
spec_defaults <- list(
  usage = FALSE, description = FALSE, arguments = FALSE, value = FALSE,
  examples = "none", output = FALSE, strip_dtypes = FALSE
)
mk <- function(...) modifyList(spec_defaults, list(...))

FULL <- mk(usage = TRUE, description = TRUE, arguments = TRUE, value = TRUE,
           examples = "all", output = TRUE)

CONDITIONS <- list(
  # 1  -- no docs baseline (baseline / control)
  "none"                   = list(usage = FALSE, description = FALSE,
                                  arguments = FALSE, value = FALSE,
                                  examples = "none", output = FALSE,
                                  strip_dtypes = FALSE),
  # 2a -- compounding ladder
  "2a_1_signatures"        = mk(usage = TRUE),
  "2a_2_sig_description"   = mk(usage = TRUE, description = TRUE),
  "2a_3_one_example"       = mk(usage = TRUE, description = TRUE, examples = "one"),
  "2a_4_examples_io"       = mk(usage = TRUE, description = TRUE,
                                examples = "all", output = TRUE),
  "2a_5_full"              = FULL,
  # 2b -- full minus one factor
  "2b_1_no_dtypes"         = modifyList(FULL, list(strip_dtypes = TRUE)),
  "2b_2_no_examples"       = modifyList(FULL, list(examples = "none", output = FALSE))
)

# COMPOSITION
first_example <- function(examples_txt) {
  if (!nzchar(examples_txt)) return("")
  trimws(strsplit(examples_txt, EXAMPLE_SPLIT)[[1]][1])
}

# prefix every line of a text block as a comment; "" -> character(0)
.comment <- function(txt) {
  if (is.null(txt) || !nzchar(txt)) return(character(0))
  paste0(COMMENT, strsplit(txt, "\n", fixed = TRUE)[[1]])
}

# one function's doc -> character vector of comment lines
compose_block <- function(fun, rec, alias, spec) {
  maybe_strip <- function(t) if (spec$strip_dtypes) strip_dtypes(t) else t
  parts <- character()

  if (spec$usage && nzchar(rec$usage))
    parts <- c(parts, paste0(COMMENT, "Signature:"), .comment(rec$usage))
  if (spec$description && nzchar(rec$description))
    parts <- c(parts, paste0(COMMENT, "Description:"), .comment(maybe_strip(rec$description)))
  if (spec$arguments && nzchar(rec$arguments))
    parts <- c(parts, paste0(COMMENT, "Arguments:"), .comment(maybe_strip(rec$arguments)))
  if (spec$value && nzchar(rec$value))
    parts <- c(parts, paste0(COMMENT, "Returns:"), .comment(maybe_strip(rec$value)))

  if (spec$examples != "none" && nzchar(rec$examples)) {
    ex <- if (spec$examples == "one") first_example(rec$examples) else rec$examples
    parts <- c(parts, paste0(COMMENT, "Examples:"), .comment(ex))
    if (spec$output) {
      out <- capture_example_output(alias, ex)
      out <- maybe_strip(out)
      if (nzchar(out)) parts <- c(parts, paste0(COMMENT, "Output:"), .comment(out))
    }
  }

  if (!length(parts)) return(character(0))
  c(paste0(COMMENT, "--- ", fun, " ---"), parts, COMMENT)
}

# union of required functions -> full top-of-file doc block (comment lines)
compose_top_block <- function(required, spec, records) {
  lines   <- character()
  missing <- character()
  seen    <- character()   # content signatures, to drop duplicate records

  for (fun in required) {
    keys <- resolve_alias(fun, records)
    if (!length(keys)) { missing <- c(missing, fun); next }
    for (k in keys) {
      rec <- records[[k]]
      sig <- paste(rec$usage, rec$description)   # cheap dedupe key
      if (sig %in% seen) next
      seen  <- c(seen, sig)
      lines <- c(lines, compose_block(fun, rec, k, spec))
    }
  }

  # Add a clean header to the doc block if there's any content
  if (length(lines)) {
    lines <- c(paste0(COMMENT, "--- docs ---"), lines)
  }

  list(lines = lines, missing = unique(missing))
}

# INJECTION + DRIVER
inject_docs <- function(task_file, doc_lines, out_file) {
  lines <- readLines(task_file, warn = FALSE)
  # Strip @requires directive line (matches same pattern as read_required)
  lines <- lines[!grepl("@requires", lines)]
  idx <- grep("<<DOCS>>", lines)
  if (!length(idx)) stop("No <<DOCS>> marker in ", task_file, call. = FALSE)
  i <- idx[1]
  before <- if (i > 1) lines[seq_len(i - 1)] else character(0)
  after  <- if (i < length(lines)) lines[(i + 1):length(lines)] else character(0)
  writeLines(c(before, doc_lines, after), out_file)
}

# Build one condition's processed task files into OUT_DIR/<cond>/
build_condition <- function(cond, records) {
  spec <- CONDITIONS[[cond]]
  cdir <- file.path(OUT_DIR, cond)
  dir.create(cdir, recursive = TRUE, showWarnings = FALSE)
  task_files <- list.files(TASKS_DIR, pattern = "\\.R$", full.names = TRUE)
  if (!length(task_files)) stop("No .R task files in ", TASKS_DIR, call. = FALSE)
  miss_log <- list()
  for (tf in task_files) {
    required <- read_required(tf)
    blk <- compose_top_block(required, spec, records)
    inject_docs(tf, blk$lines, file.path(cdir, basename(tf)))
    if (length(blk$missing))
      miss_log[[paste(cond, basename(tf))]] <- blk$missing
  }
  if (length(miss_log)) {
    message("\nUnresolved @requires tokens (check alias scheme):")
    for (k in names(miss_log))
      message("  ", k, ": ", paste(miss_log[[k]], collapse = ", "))
    invisible(miss_log)
  }
  message("built: ", cond)
}

build_all <- function() {
  records <- build_doc_records(load_doc_db())
  for (cond in names(CONDITIONS)) build_condition(cond, records)
  invisible(records)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1L && nzchar(args[1L])) {
    cond <- args[1L]
    if (!cond %in% names(CONDITIONS))
      stop("Unknown condition: ", cond, ". Available: ",
           paste(names(CONDITIONS), collapse = ", "), call. = FALSE)
    records <- build_doc_records(load_doc_db())
    build_condition(cond, records)
  } else {
    build_all()
  }
}
