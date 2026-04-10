#!/usr/bin/env Rscript
# Auto-install required packages if missing
required <- c("rio", "dplyr", "mlogit", "rmarkdown", "knitr", "ggplot2", "openssl")
for(pkg in required){
  if(!requireNamespace(pkg, quietly = TRUE)){
    install.packages(pkg, repos="https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# Render the R Markdown analysis file (source Rmd)
rmarkdown::render("/workspace/replication_data/Analysis_updated_src.Rmd", output_format = "html_document", output_dir = "/workspace/replication_data")
