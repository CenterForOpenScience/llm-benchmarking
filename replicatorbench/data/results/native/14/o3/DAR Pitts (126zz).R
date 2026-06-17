# Replication script for Pitts et al. focal model (LeavingAgency ~ JobSat + controls)
# This version is adapted for execution inside the Docker container.

# Load required packages
library(data.table)
library(sandwich)
library(lmtest)
library(miceadds)   # provides glm.cluster

# Path to the data relative to working directory (/workspace) when container runs
csv_path <- "replication_data/Estimation Data - Pitts (126zz).csv"

# Read data
FinDT <- data.table::fread(csv_path, data.table = FALSE)

# Listwise deletion on variables used in the model
model_vars <- c("LeavingAgency", "JobSat", "Over40", "NonMinority", "SatPay", 
                "SatAdvan", "PerfCul", "Empowerment", "RelSup", "Relcow", 
                "Over40xSatAdvan", "Agency")
FinDT <- FinDT[complete.cases(FinDT[ , model_vars ]), ]

# Fit logistic regression with agency‐clustered SEs
myLogit <- miceadds::glm.cluster(formula = LeavingAgency ~ JobSat + Over40 + NonMinority + 
                                   SatPay + SatAdvan + PerfCul + Empowerment + 
                                   RelSup + Relcow + Over40xSatAdvan,
                                 data = FinDT,
                                 cluster = "Agency",
                                 family = binomial)

# Show summary in console
print(summary(myLogit))

# Extract coefficient information for JobSat# Extract coefficient information for JobSat
coefs <- summary(myLogit)  # summary returns a matrix for glm.cluster in miceadds
if (is.list(coefs) && !is.null(coefs$coefficients)) {
  coef_matrix <- coefs$coefficients
} else if (is.matrix(coefs)) {
  coef_matrix <- coefs
} else {
  stop("Unexpected structure of summary(myLogit)")
}

job_row <- coef_matrix["JobSat", ]

# Prepare output lines
out_lines <- c(
  "Replication results for JobSat in logistic model predicting LeavingAgency",
  paste0("Coefficient (log-odds): ", round(job_row["Estimate"], 4)),
  paste0("Std. Error (cluster-robust): ", round(job_row["Std. Error"], 4)),
  paste0("z value: ", round(job_row["z value"], 3)),
  paste0("Pr(>|z|): ", formatC(job_row["Pr(>|z|)"], format = "e", digits = 3))
)

# Write to designated output file (mounted volume inside container)
output_path <- "/app/data/replication_output.txt"
writeLines(out_lines, con = output_path)

cat("Results written to", output_path, "\n")
