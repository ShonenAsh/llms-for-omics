# Environment probe for the MSstats documentation -> LLM hallucination study.
# Run once: Rscript reference/probe.R > reference/probe_output.txt 2>&1
# This is the source of truth for every structural test assertion in tasks/.
# Never hand-write a column/argument name from memory -- pull it from this output.

suppressPackageStartupMessages(library(MSstats))

hr <- function(title) cat("\n\n====", title, "====\n")

hr("MSstats package version")
cat(as.character(packageVersion("MSstats")), "\n")

hr("Built-in datasets (data(package = \"MSstats\"))")
print(data(package = "MSstats")$results[, c("Item", "Title")])

hr("ls(\"package:MSstats\") -- full exported symbol list")
print(ls("package:MSstats"))

hr("formals(dataProcess)")
print(formals(dataProcess))

hr("formals(groupComparison)")
print(formals(groupComparison))

hr("DIARawData: str()")
data(DIARawData)
str(DIARawData)

hr("DIARawData: colnames()")
print(colnames(DIARawData))

hr("DIARawData: head()")
print(head(DIARawData))

hr("dataProcess(DIARawData) with defaults -- running, this may take a moment")
set.seed(1)
QuantData <- dataProcess(DIARawData, use_log_file = FALSE, verbose = FALSE)

hr("class(QuantData)")
print(class(QuantData))

hr("names(QuantData) / structure")
print(names(QuantData))
str(QuantData, max.level = 2)

hr("names(QuantData$ProteinLevelData)")
print(colnames(QuantData$ProteinLevelData))

hr("head(QuantData$ProteinLevelData)")
print(head(QuantData$ProteinLevelData))

hr("names(QuantData$FeatureLevelData)")
print(colnames(QuantData$FeatureLevelData))

hr("groupComparison(QuantData) with defaults -- running")
levels(QuantData$ProteinLevelData$GROUP)
groups <- levels(QuantData$ProteinLevelData$GROUP)
cat("Groups found:", paste(groups, collapse = ", "), "\n")

# Build a simple pairwise comparison matrix across all groups found, since the
# task stubs need to know the exact contrast-matrix shape MSstats expects.
if (length(groups) >= 2) {
  comparison <- matrix(0, nrow = 1, ncol = length(groups))
  colnames(comparison) <- groups
  comparison[1, 1] <- 1
  comparison[1, 2] <- -1
  row.names(comparison) <- paste(groups[1], groups[2], sep = "-")

  print(comparison)

  testResultOneComparison <- groupComparison(
    contrast.matrix = comparison,
    data = QuantData,
    use_log_file = FALSE,
    verbose = FALSE
  )

  hr("class(testResultOneComparison)")
  print(class(testResultOneComparison))

  hr("names(testResultOneComparison)")
  print(names(testResultOneComparison))

  hr("colnames(testResultOneComparison$ComparisonResult) -- THE key hallucination-target schema")
  print(colnames(testResultOneComparison$ComparisonResult))

  hr("head(testResultOneComparison$ComparisonResult)")
  print(head(testResultOneComparison$ComparisonResult))

  hr("str(testResultOneComparison$ComparisonResult)")
  str(testResultOneComparison$ComparisonResult)
} else {
  cat("Fewer than 2 groups found in DIARawData -- cannot build contrast matrix.\n")
}

hr("*toMSstatsFormat converters exported")
print(grep("toMSstatsFormat", ls("package:MSstats"), value = TRUE))

hr("probe complete")
