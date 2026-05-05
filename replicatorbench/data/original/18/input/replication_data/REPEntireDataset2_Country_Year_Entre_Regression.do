clear
set more off

capture log using /Users/Victor/Downloads/OriginaldataVictorVolkmanVer/Log/Table2_Country_Year_Entre_RegressionREP.log, replace

use /Users/Victor/Downloads/OriginaldataVictorVolkmanVer/Data/GEM_Country_Year.dta, clear

import delimited REPdata using /Users/Victor/Downloads/replication_data_mkk9.csv, clear

drop if median_age == "NA"
destring median_age, replace


local dep_var    = "entrepreneurship"

local opt_weight = "[aw=cy_cell]"
local opt_std    = ", cluster(country)"

/* Table 2. Column 3*/
eststo: quietly reg `dep_var' median_age i.year `opt_weight' `opt_std' 
estout using LiangTestCompleteVersion2.csv, cells(b(star fmt(3)) t(par fmt(2))) replace
eststo clear
asdoc reg `dep_var' median_age i.year `opt_weight' `opt_std' , save(LiangTestCompleteFull.doc) replace


log close
