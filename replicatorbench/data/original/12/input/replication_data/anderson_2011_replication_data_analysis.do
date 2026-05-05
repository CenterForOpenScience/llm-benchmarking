*Nathaniel Porter
*2022-05-03
*SCORE Anderson_AmEcoJourn_2011_bLe8_329k Data Analysis


*change to local directory and load data
cd "d:/aris_reds_data"

log using "anderson_2011_replication_final_analysis", replace
*log using "anderson_2011_replication_analysis"

use "analysis_data.dta", clear

/*
*Preregistered analysis (5% sample)
set seed 234
sample 5

*Regression (Table 3) with all (both types of locaste)
*state code and caste interacted (because numbering differs by state)
*alternative would be to recode state-specific castes into new castes
qui regress raw_inc_per_acre literate_hh land_owned locaste_land_v stcode##caste, vce(cluster vill_id)
*output (line 1 is b, line 2 is SE, line 3 is test stat, line 4 is p-value)
etable, keep(literate_hh land_owned locaste_land_v) cstat(_r_b) cstat(_r_se) cstat(_r_z) cstat(_r_p) mstat(N) mstat(r2) mstat(r2_a) mstat(aic) mstat(bic) mstat(ll)
*/

*final analysis
use "analysis_data.dta", clear
qui regress raw_inc_per_acre literate_hh land_owned locaste_land_v stcode##caste, vce(cluster vill_id)
*output (line 1 is b, line 2 is SE, line 3 is test stat, line 4 is p-value)
etable, keep(literate_hh land_owned locaste_land_v) cstat(_r_b) cstat(_r_se) cstat(_r_z) cstat(_r_p, nformat(%6.4f)) mstat(N) mstat(r2) mstat(r2_a) mstat(aic) mstat(bic) mstat(ll)

*Exploratory analysis using net income per acre
qui regress net_inc_per_acre literate_hh land_owned locaste_land_v stcode##caste, vce(cluster vill_id)
etable, keep(literate_hh land_owned locaste_land_v) cstat(_r_b) cstat(_r_se) cstat(_r_z) cstat(_r_p, nformat(%6.4f)) mstat(N) mstat(r2) mstat(r2_a) mstat(aic) mstat(bic) mstat(ll)

*Alternative analysis using only subset of cases in UP/B (following original study)
keep if inlist(stcode,2,15)
qui regress raw_inc_per_acre literate_hh land_owned locaste_land_v stcode##caste, vce(cluster vill_id)
etable, keep(literate_hh land_owned locaste_land_v) cstat(_r_b) cstat(_r_se) cstat(_r_z) cstat(_r_p, nformat(%6.4f)) mstat(N) mstat(r2) mstat(r2_a) mstat(aic) mstat(bic) mstat(ll)

log close