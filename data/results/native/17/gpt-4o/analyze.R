library("lmtest")
library("sandwich")
set.seed(2020)

## Uses the output from the data build.

main <- function(mode = "test", drop_wave_7 = T){
  data <- read.csv("/app/data/data.csv")
  
  if (drop_wave_7) data <- data[!(data$wave == 7), ]
  drop <- ifelse(drop_wave_7, "drop_7", "keep_7")
  
  if (mode == "test"){
    # Randomize data in each column independently for testing purposes.
    for (i in 3:ncol(data)){
      data[, i] <- sample(data[, i])
    }
  }
  
  # Baseline "Long" Model
  # https://www.r-econometrics.com/methods/hcrobusterrors/
  # https://stats.stackexchange.com/questions/117052/replicating-statas-robust-option-in-r (HC1 matches Stata robust option)
  model <- lm(gov_consumption ~ sd_gov + mean_gov +
                africa + laam + asiae +
                col_uka + col_espa + col_otha +
                federal + oecd +
                log_gdp_per_capita + trade_share + age_15_64 + age_65_plus, 
              data = data)
  out <- coeftest(model, vcov = vcovHC(model, type = "HC1")) 
  
  
  # Clean and produce table
  out <- rbind(out, c(length(model$residuals), "", "", ""))
  out <- rbind(out, c(summary(model)$r.squared, "", "", ""))
  rownames(out) <- c(rownames(out)[-c((nrow(out)-1):nrow(out))], "Obs.", "R-squared")
  write.table(out, file = sprintf("/app/data/%s_%s.txt", mode, drop))
}

# Test analysis code
main(mode = "test", drop_wave_7 = T)
main(mode = "test", drop_wave_7 = F)

# Run on real data w/o wave 7
main(mode = "real", drop_wave_7 = T)

# Run on real data w/ wave 7
main(mode = "real", drop_wave_7 = F)