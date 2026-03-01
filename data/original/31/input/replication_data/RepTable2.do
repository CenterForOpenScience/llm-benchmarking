log using "/Users/Victor/Downloads/Original data 3/Stata-log-files/Table2R2.log", replace

clear
set mem 100m
set more off



***loading variables for Experiment 1 

use "/Users/Victor/Downloads/Original data 3/REPExperiment1DataR2.dta"



//Ethnicity-Salience Treatment Effects



***Table 2C: Effect of race-salience treatment on risk premia of Asians

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if asian == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2CAsianR2.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if asian == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2CAsianLogR2.doc) replace


***Table 2D: Effect of race-salience treatment on risk-premia of whites

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if white == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DWhiteR2.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if white == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DWhiteLogR2.doc) replace

***Table 2D: Effect of race-salience treatment on risk-premia of blacks

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if black == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DBlackR2.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if black == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DBlackLogR2.doc) replace


***Table 2A: Effect of race-salience treatment on discount rates of Asians

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if asian == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2AAsianR2.doc) replace


***Table 2B: Effect of race-salience treatment on discount rate of whites

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if white == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2BWhiteR2.doc) replace

***Table 2Bii: Effect of race-salience treatment on discount rate of blacks

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if black == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2BBlackR2.doc) replace



log close
