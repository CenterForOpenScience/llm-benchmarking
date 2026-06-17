# Replication analysis for Yang et al. (2013) focal claim: Lottery vs Gift framing reduces WTP
# Container-ready script: reads data from /app/data, handles encoding, computes t-test and effect sizes.

suppressPackageStartupMessages({
  library(psych)   # describeBy
  library(effsize) # cohen.d
})

DATA_PATH <- "/app/data/Data_Cleaned_22102020.csv"
OUTPUT_SUBSET <- "/app/data/dataS.csv"

# Robust CSV reader that tries multiple encodings and a raw-bytes fallback
safe_read_csv <- function(path) {
  encs <- c("UTF-8-BOM", "UTF-8", "latin1", "windows-1252")
  for (enc in encs) {
    df <- try(read.csv(path, header = TRUE, na.strings = c("", "NA", " "), fileEncoding = enc), silent = TRUE)
    if (!inherits(df, "try-error") && is.data.frame(df) && ncol(df) > 0 && nrow(df) > 0) return(df)
  }
  # Fallback: read raw bytes, strip BOM, normalize newlines, convert to UTF-8
  con <- file(path, open = "rb")
  on.exit(try(close(con), silent = TRUE), add = TRUE)
  raw_bytes <- readBin(con, what = "raw", n = file.info(path)$size)
  # Strip UTF-8 BOM if present (EF BB BF)
  if (length(raw_bytes) >= 3 && all(raw_bytes[1:3] == as.raw(c(0xEF, 0xBB, 0xBF)))) {
    raw_bytes <- raw_bytes[-c(1:3)]
  }
  txt <- rawToChar(raw_bytes)
  txt <- iconv(txt, from = "", to = "UTF-8", sub = "")
  txt <- gsub("\r\n?", "\n", txt)
  df <- read.csv(text = txt, header = TRUE, na.strings = c("", "NA", " "))
  if (!is.data.frame(df) || nrow(df) == 0) stop("Parsed CSV has no rows after sanitization")
  df
}

# Helper: coerce numeric columns if they contain stray symbols (e.g., $)
clean_numeric <- function(x) {
  if (is.numeric(x)) return(x)
  as.numeric(gsub("[^0-9.-]", "", as.character(x)))
}

# Read data
raw <- tryCatch(safe_read_csv(DATA_PATH), error = function(e) {
  stop(paste0("Failed to read data from ", DATA_PATH, ": ", e$message))
})

# Clean and construct variables
raw$Lot_WTPc  <- clean_numeric(raw$Lot_WTPc)
raw$Gift_WTPc <- clean_numeric(raw$Gift_WTPc)
lot_chk <- clean_numeric(raw$Lot_check)
gft_chk <- clean_numeric(raw$Gift_check)

# Condition: 0 = lottery, 1 = gift
Cond <- rep(NA_integer_, nrow(raw))
Cond[!is.na(lot_chk) & is.na(gft_chk)] <- 0L
Cond[is.na(lot_chk) & !is.na(gft_chk)] <- 1L
both <- which(!is.na(lot_chk) & !is.na(gft_chk))
if (length(both) > 0) {
  use_lot <- which(!is.na(raw$Lot_WTPc[both]) & is.na(raw$Gift_WTPc[both]))
  use_gft <- which(is.na(raw$Lot_WTPc[both]) & !is.na(raw$Gift_WTPc[both]))
  if (length(use_lot) > 0) Cond[both[use_lot]] <- 0L
  if (length(use_gft) > 0) Cond[both[use_gft]] <- 1L
}
raw$Cond <- factor(Cond, levels = c(0, 1), labels = c("lottery", "gift"))

# WTP by condition
raw$WTP <- NA_real_
raw$WTP[raw$Cond == "lottery"] <- raw$Lot_WTPc[raw$Cond == "lottery"]
raw$WTP[raw$Cond == "gift"]    <- raw$Gift_WTPc[raw$Cond == "gift"]

# Comprehension check aligned with condition
raw$check <- NA_real_
raw$check[raw$Cond == "lottery"] <- lot_chk[raw$Cond == "lottery"]
raw$check[raw$Cond == "gift"]    <- gft_chk[raw$Cond == "gift"]

# Drop rows missing condition or WTP
dat <- subset(raw, !is.na(Cond) & !is.na(WTP))

# Critical subset: correct comprehension (code == 3 for lowest $10)
datS <- subset(dat, check == 3)

# Save subset
try(write.csv(datS, OUTPUT_SUBSET, row.names = FALSE), silent = TRUE)

cat("\nDescriptives by condition (correct comprehension only):\n")
print(describeBy(datS$WTP, datS$Cond))

cat("\nWelch two-sample t-test (WTP ~ Cond) on correct subset:\n")
print(t.test(WTP ~ Cond, data = datS))

cat("\nEffect size (Cohen's d; lottery vs gift):\n")
print(cohen.d(datS$WTP, datS$Cond, hedges.correction = TRUE))

cat("\nMann-Whitney (Wilcoxon rank-sum) test:\n")
print(wilcox.test(WTP ~ Cond, data = datS, exact = FALSE))

cat("\nNs per group (correct subset):\n")
print(table(datS$Cond))

cat("\nAnalysis completed. Subset written to:", OUTPUT_SUBSET, "\n")
