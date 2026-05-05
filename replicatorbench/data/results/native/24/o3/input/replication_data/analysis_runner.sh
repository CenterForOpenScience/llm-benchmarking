#!/bin/bash
# Script to render analysis_report.Rmd to HTML and stdout key results
set -e
Rscript -e "rmarkdown::render('analysis_report.Rmd', output_file='analysis_report.html')"