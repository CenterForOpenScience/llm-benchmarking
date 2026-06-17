*****************************************
* This do-file replicates a research claim from Cohen et al. (2015) in American Economic Review
* "Price Subsidies, Diagnostic Tests, and Targeting of Malaria Treatment: Evidence from a Randomized Controlled Trial"
* H*: ACT [artemisinin combination therapies] subsidies induce take-up of ACT.
*
*****************************************
version 15.1
clear all

cd "..." /*Change directory*/

*Start a log file
local log_name "Cohen-et-al_Replication" /*Give a name to the log where session is recorded*/
log using `log_name', replace

use ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta /*Import provided replication dataset*/

************************ Dependent variable: Took ACT ************************
*"y_eh is the outcome of interest for illness episode e in household h." (page 627)
*"Column 1 of Table 2 reports results on overall ACT access" (page 628)

rename drugs_taken_AL took_ACT  
 
************** Focal independent variable: any ACT voucher subsidy **************
/*"Panel A of Table 2 presents a specification where we pool all three ACT subsidies and compare outcomes
to the control group, while panel B presents a specification where we separately
estimate the impact of the three different subsidy levels. In both cases, the
omitted category is the “no ACT subsidy” (control) group" (page 627)*/
gen act_subsidy = (maltest_chw_voucher_given==1)
*Replace "act_subsidy" as missing for maltest_chw_voucher_given=98
replace act_subsidy=. if maltest_chw_voucher_given==98

************************ Other controls ************************
*1) Household ID
*“Robust standard errors clustered at the household level in parentheses” (page 627)
/*According to the "Sample Size" Section, each observation in each wave corresponds
to a unique household, i.e. only 1 random fever is selected in each sampled household.
However, it is possible that the same household or febrile individual may have been 
surveyed more than once across the survey waves. There is no way to determine this 
in the data, but the data source reports that this is “expected to be rare” because 
the random starting household and the sampling interval are both different in each wave.
Consequently, it is assumed that each observation is a different household.*/ 
gen hh_id=_n

*2) Sampling Weight:
*Variable: weight
*Sampling weight.
*Description: continuous variable that indicates the CU-level weights (by wave)
*It corresponds to the weight variable, no further changes are needed.

*3) Strata:
/*\lambda_strata are strata fixed effects (page 627)*/
gen strata = cu_code

*4)Where malaria test was taken
*Variable: maltest_where
*Sample parameter variable.
/*Description: As it is explained in the “Study Design” and "Sample size" sections,
by design, the ACT subsidy is given to individuals who were tested through the 
community health worker (CHW) program and got a positive result. Consequently, 
the estimate is restricted to the subsample where maltest_where=1 (CHW)*/

*5)Data collection period
*Variable: wave
*Sample parameter variable.
/*Description: As it is explained in the "Sample size" section, the focal dependent 
and independent variables are obtained from the 3 post-baseline surveys, i.e., WAGE!=0.*/

*Test for statistically significant differences between the control group
/*Table 1 in Cohen et al (2015) presents baseline household characteristics and 
tests for balance across treatment groups. To test balance across the experimental
groups, they regressed each dependent variable in Table 1 on a dummy variable for 
each of the three ACT subsidy levels and a dummy variable for the RDT subsidy. 
Moreover, they include a full set of strata dummies in the regression. (pages 622-624)*/

*Declare survey data
svyset hh_id [pweight=weight], strata(strata)

*Household assets
*Description: Does your household have the following items?
*1) Electricity
gen electricity= strpos(ses_hh_items, "1")
replace electricity=. if ses_hh_item==""
*2) Television
gen television= strpos(ses_hh_items, "2")
replace television=. if ses_hh_item==""
*3) Refrigerator
gen refrigerator= strpos(ses_hh_items, "3")
replace refrigerator=. if ses_hh_item==""
*4) Radio
gen radio= strpos(ses_hh_items, "4")
replace radio=. if ses_hh_item==""
*5) Mobile phone
gen mobile= strpos(ses_hh_items, "5")
replace mobile=. if ses_hh_item==""
*6) Motorcycle
gen motorcycle= strpos(ses_hh_items, "6")
replace motorcycle=. if ses_hh_item==""
*7) Car/Truck
gen car= strpos(ses_hh_items, "7")
replace car=. if ses_hh_item==""
*8) Bank account
gen bank_account= strpos(ses_hh_items, "8")
replace bank_account=. if ses_hh_item==""
*9) No assets
gen no_assets= strpos(ses_hh_items, "9")
replace no_assets=. if ses_hh_item==""

svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy electricity i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy television i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy refrigerator i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy radio i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy mobile i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy motorcycle i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy car i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy bank_account i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy no_assets i.strata
*At a 5% significance level, households in the treatment group are more likely to have a refrigerator, and a mobile phone.

*Number of cows
gen num_cows=ses_no_cows
replace num_cows=. if ses_no_cows==.
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy num_cows i.strata
*No significant difference between the groups in terms of number of cows present in the household.

*Number of sheep
gen num_sheep=ses_no_sheep
replace num_sheep=. if ses_no_sheep==.
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy num_sheep i.strata
*At a 5% significance level, households in the treatment group have a higher number of sheep.

*Number of goats
gen num_goats=ses_no_goats
replace num_goats=. if ses_no_goats==.
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy num_goats i.strata
*No significant difference between the groups in terms of number of goats present in the household.

*Type of toilet
*Description: What kind of toilet does your household have?
gen flush_toilet = (ses_toilet_type==1) /*Flush or pour flush toilet*/
replace flush_toilet=. if ses_toilet_type==.

gen vip_toilet = (ses_toilet_type==2) /*VIP/Ventilated improved pit*/
replace vip_toilet=. if ses_toilet_type==.

gen latrine_with_slab = (ses_toilet_type==3) /*Pit latrine WITH slab*/
replace latrine_with_slab=. if ses_toilet_type==.

gen latrine_without_slab = (ses_toilet_type==4) /*Pit latrine without slab*/
replace latrine_without_slab=. if ses_toilet_type==.

gen composting_toilet = (ses_toilet_type==5) /*Composting toilet*/
replace composting_toilet=. if ses_toilet_type==.

gen bucket_toilet = (ses_toilet_type==6) /*Bucket toilet*/
replace bucket_toilet=. if ses_toilet_type==.

gen no_facility_toilet = (ses_toilet_type==7) /*No facility/bush/field*/
replace no_facility_toilet=. if ses_toilet_type==.

gen other_toilet = (ses_toilet_type==8) /*Other*/
replace other_toilet=. if ses_toilet_type==.

svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy flush_toilet i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy vip_toilet i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy latrine_with_slab i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy latrine_without_slab i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy composting_toilet i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy bucket_toilet i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy no_facility_toilet i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy other_toilet i.strata
/*At a 5% significance level, households in the treatment group are more likely 
to have a "VIP/Ventilated improved pit", a "Composting toilet" or "Other type of toilet".*/ 

*Type of fuel
*Description: What type of fuel does your household mainly use for cooking?
/*Note that the households in the control and treatment group do not contain any
type of fuel information in the replication dataset.*/ 
tab ses_fuel_type if maltest_where==1 & wave!=0
*no observations

*Type of floor material
*Description: Main material of the floor in your house?
gen earthen_floor = (ses_floor_material==1) /*Earthen*/
replace earthen_floor=. if ses_floor_material==.

gen cement_floor = (ses_floor_material==2) /*Cement*/
replace cement_floor=. if ses_floor_material==.

gen floor_tiles = (ses_floor_material==3) /*Floor tiles*/
replace floor_tiles=. if ses_floor_material==.

gen wood_planks = (ses_floor_material==4) /*Wood planks*/
replace wood_planks=. if ses_floor_material==.

gen polished_wood = (ses_floor_material==5) /*Polished wood*/
replace polished_wood=. if ses_floor_material==.

gen other_floor = (ses_floor_material==6)
replace other_floor=. if ses_floor_material==.

svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy earthen_floor i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy cement_floor i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy floor_tiles i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy wood_planks i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy polished_wood i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy other_floor i.strata
*No significant differences between the groups in types of floor material.

*Type of Wall material
*Description: Main material of the walls in your house?
gen stone_wall = (ses_wall_material==1) /*Stone*/
replace stone_wall=. if ses_wall_material==.

gen brick_wall = (ses_wall_material==2) /*Brick*/
replace brick_wall=. if ses_wall_material==.

gen timber_wall = (ses_wall_material==3) /*Timber*/
replace timber_wall=. if ses_wall_material==.

gen iron_wall = (ses_wall_material==4) /*Iron Sheet*/
replace iron_wall=. if ses_wall_material==.

gen mud_wall = (ses_wall_material==5) /*Mud*/
replace mud_wall=. if ses_wall_material==.

gen wood_wall = (ses_wall_material==6) /*Wood*/
replace wood_wall=. if ses_wall_material==.

gen cement_wall = (ses_wall_material==7) /*Cement*/
replace cement_wall=. if ses_wall_material==.

gen other_wall = (ses_wall_material==8) /*Other*/
replace other_wall=. if ses_wall_material==.

svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy stone_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy brick_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy timber_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy iron_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy mud_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy wood_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy cement_wall i.strata
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy other_wall i.strata
/*At a 5% significance level, households in the treatment group are more likely 
to have "Stone" and "Cement" walls.*/

*Wealth Index (raw score), pooled
*Description: Based on DHS wealth index, using polychoric correlation and principal components anlaysis (PCA)
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy ses_DHS_score_pooled i.strata
*No significant differences between the groups in wealth index (raw score).

*Wealth Index (quintile), pooled
*Description: Wealth index by quintile
svy, subpop(if maltest_where==1 & wave!=0): reg act_subsidy ses_DHS_percentile_pooled i.strata
*No significant differences between the groups in wealth index quintiles.

************************ Test of the SCORE claim, H*  ************************
*Control variables
/* \lamda_strata are strata fixed effects, and x_h controls for age of the household head (p. 627)*/
local covariates "i.cu_code refrigerator mobile vip_toilet composting_toilet other_toilet stone_wall cement_wall num_sheep" 
*OLS regression model for the impact on ACT access
svy, subpop(if maltest_where==1 & wave!=0): reg took_ACT act_subsidy `covariates'

*Close log
log close
*Create PDF from log
translate `log_name'.smcl `log_name'.pdf, replace

display `End of Do-file'
