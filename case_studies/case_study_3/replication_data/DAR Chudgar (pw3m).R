#rm(list=ls())
#setwd('***')
data_dir <- Sys.getenv("DATA_DIR", unset = "/app/data")
csv_path <- file.path(data_dir, "Estimation Data - Chudgar (pw3m).csv")
DT <- as.data.frame(read.csv(csv_path, stringsAsFactors = FALSE))
#DT <- as.data.frame(read.csv('Estimation Data - Chudgar (pw3m)'))
colnames(DT) <- c('Country','CountryID',
                  'CountryNUM','YEAR',
                  'GINI','Variance','Flag')
DT <- subset(DT, DT$Flag==1)
cor.test(DT$GINI,DT$Variance)


#DT2 <- DT[sample(c(1:nrow(DT)), ceiling(nrow(DT)*0.05)),]
#cor.test(DT2$GINI,DT2$Variance)







