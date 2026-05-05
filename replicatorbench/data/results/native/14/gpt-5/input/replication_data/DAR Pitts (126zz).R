rm(list=ls())
suppressPackageStartupMessages({
  library(data.table)  # for fast data loading
  library(sandwich)    # for robust variance estimators
})

# Locate data file under /workspace/replication_data (preferred) or /app/data
DATA_FILE <- 'Estimation Data - Pitts (126zz).csv'
if (file.exists(file.path('/workspace/replication_data', DATA_FILE))) {
  setwd('/workspace/replication_data')
} else if (file.exists(file.path('/app/data', DATA_FILE))) {
  setwd('/app/data')
} else {
  stop(paste0('Data file not found in either /workspace/replication_data or /app/data: ', DATA_FILE))
}

# Load dataset
FinDT <- as.data.frame(read.csv(DATA_FILE))

# Listwise deletion on variables used in model
vars_used <- c('LeavingAgency','JobSat','Over40','NonMinority','SatPay','SatAdvan','PerfCul','Empowerment','RelSup','Relcow','Over40xSatAdvan','Agency')
FinDT <- FinDT[complete.cases(FinDT[, vars_used]), ]

# Fit logistic regression (binomial logit)
myLogit <- glm(
  formula = LeavingAgency ~ JobSat + Over40 + NonMinority + SatPay + SatAdvan + PerfCul + Empowerment + RelSup + Relcow + Over40xSatAdvan,
  data = FinDT,
  family = binomial(link = 'logit')
)

# Cluster-robust variance-covariance matrix by Agency
cl <- as.factor(FinDT$Agency)
V_cl <- sandwich::vcovCL(myLogit, cluster = cl)

# Compute robust coefficient table
coefs <- coef(myLogit)
se <- sqrt(diag(V_cl))
z <- coefs / se
p <- 2 * pnorm(abs(z), lower.tail = FALSE)
res_tab <- data.frame(Estimate = coefs, Robust_SE = se, z = z, p_value = p, row.names = names(coefs))

# Output results to console and file under /app/data (fallback to CWD if /app/data missing)
out_dir <- '/app/data'
if (!dir.exists(out_dir)) { out_dir <- getwd() }
res_file <- file.path(out_dir, 'replication_results_pitts_126zz.txt')
sink(res_file)
cat('Logistic regression with cluster-robust SE by Agency (sandwich::vcovCL)\n')
print(res_tab)
cat('\nN (complete cases):', nrow(FinDT), '\n')
sink()

cat('Logistic regression with cluster-robust SE by Agency (sandwich::vcovCL)\n')
print(res_tab)
cat('\nN (complete cases):', nrow(FinDT), '\n')
