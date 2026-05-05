# Popper Replication Driver Script (clean version)
# Loads replication datasets, fits SEM Model 1, and writes key results.

suppressPackageStartupMessages({
  library(lavaan)
  library(psych)
  library(dplyr)
  library(tidyr)
})

# ---------------- Helper to locate data files ----------------
locate_file <- function(fname) {
  candidates <- c(
    file.path("/app/data", fname),
    file.path("/workspace/data", fname),
    file.path("/workspace/replication_data/Popper Replication Data Files", fname),
    file.path("replication_data/Popper Replication Data Files", fname),
    file.path("Popper Replication Data Files", fname),
    fname
  )
  for (p in candidates) {
    if (file.exists(p)) return(p)
  }
  stop(paste("File", fname, "not found in any candidate locations."))
}

# ---------------- Load data ----------------
cat("Locating datasets...\n")
corr_path <- locate_file("Popper Data for Correlations.csv")
cfa_path  <- locate_file("Popper_Data for CFA and SEM.csv")
cat("Correlation data:", corr_path, "\n")
cat("CFA/SEM data:", cfa_path, "\n")

Cordata <- read.csv(corr_path, stringsAsFactors = FALSE)
Popper_Data.for.CFA <- read.csv(cfa_path, stringsAsFactors = FALSE)

# Basic cleaning: drop rows with NA in focal parcels
focal_vars <- c("Open_Par1", "Open_Par2", "Open_Par3", "Lead_Par1", "Lead_Par2", "Lead_Par3")
Popper_Data.for.CFA <- Popper_Data.for.CFA %>% drop_na(all_of(focal_vars))

# ---------------- Specify SEM Model 1 ----------------
Model1 <- 'ATTACH =~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY =~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN =~ Open_Par1 + Open_Par2 + Open_Par3
LEAD =~ Lead_Par1 + Lead_Par2 + Lead_Par3

ANXIETY ~ a*ATTACH
OPEN ~ c*ATTACH
LEAD ~ b*ANXIETY + d*OPEN

ab := a*b
cd := c*d'

# ---------------- Fit SEM ----------------
cat("Fitting SEM Model 1...\n")
fitModel1 <- sem(Model1, data = Popper_Data.for.CFA, std.ov = TRUE)

# ---------------- Extract focal coefficient ----------------
std_sol <- standardizedSolution(fitModel1)
row_ol <- filter(std_sol, lhs == "LEAD", op == "~", rhs == "OPEN")

beta <- if ("std.all" %in% names(row_ol)) row_ol$std.all else if ("est.std" %in% names(row_ol)) row_ol$est.std else row_ol$est
se_val <- row_ol$se
p_val <- row_ol$pvalue

# ---------------- Fit indices ----------------
fit_vals <- fitMeasures(fitModel1, c("cfi", "rmsea", "srmr", "chisq", "df", "pvalue"))

# ---------------- Correlation (totals) ----------------
corr_beta <- NA_real_
if (all(c("Oppenness_Total", "Leader_Exp_Total") %in% names(Cordata))) {
  corr_beta <- cor(Cordata$Oppenness_Total, Cordata$Leader_Exp_Total, use = "pairwise.complete.obs")
}

# ---------------- Assemble results text ----------------
results_txt <- paste0(
  "Open -> Lead standardized beta: ", round(beta, 3), "\n",
  "SE: ", round(se_val, 3), ", p-value: ", signif(p_val, 3), "\n",
  "Correlation (total scores): ", ifelse(is.na(corr_beta), "NA", round(corr_beta, 3)), "\n\n",
  "Model Fit Indices:\n",
  paste(names(fit_vals), round(fit_vals, 3), sep = ": ", collapse = "\n"),
  "\n"
)

# ---------------- Write results ----------------
result_path <- "/app/data/results.txt"
cat("Writing results to", result_path, "\n")
tryCatch({
  writeLines(results_txt, result_path)
}, error = function(e) {
  cat("Failed to write to", result_path, "- writing to local results.txt\n")
  writeLines(results_txt, "results.txt")
})

cat("Done.\n")
