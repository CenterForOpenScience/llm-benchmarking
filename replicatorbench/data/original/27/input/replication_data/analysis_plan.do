log using analysis_plan.log, replace

/* Reproduce analysis in Zhang 2009 using group averages in Fig 3 */
clear all
use zhang_avgprice
signrank avgprice=1.94

/* Generate synthetic data to demonstrate analysis to be used */
/* The IPO simulation will compute and record the market price each round for each group */
clear all
set obs 15
gen group = _n
expand 20
sort group
by group: gen round = _n
sort group round

set seed 234567
gen price = runiform(0, 4.5)
gen bankrupt = runiform()<.005

collapse (mean) avgprice=price (max) anybankrupt=bankrupt, by(group)
signrank avgprice=1.94 if anybankrupt==0

log close
