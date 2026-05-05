log using "/Users/Victor/Downloads/Original data 3/Stata-log-files/Table2BothRounds.log", replace

clear
set mem 100m
set more off



***loading variables for Experiment 1 

use "/Users/Victor/Downloads/Original data 3/REPExperiment1DataBothRounds.dta"



//Ethnicity-Salience Treatment Effects



***Table 2C: Effect of race-salience treatment on risk premia of Asians

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if asian == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2CAsianBothRounds.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if asian == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2CAsianLogBothRounds.doc) replace


***Table 2D: Effect of race-salience treatment on risk-premia of whites

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if white == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DWhiteBothRounds.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if white == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DWhiteLogBothRounds.doc) replace

***Table 2D: Effect of race-salience treatment on risk-premia of blacks

asdoc intreg lowerboundrisk upperboundrisk givenprimingquestionnaire if black == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DBlackBothRounds.doc) replace
asdoc intreg lowerboundlogrisk upperboundlogrisk givenprimingquestionnaire if black == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2DBlackLogBothRounds.doc) replace


***Table 2A: Effect of race-salience treatment on discount rates of Asians

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if asian == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2AAsianBothRounds.doc) replace


***Table 2B: Effect of race-salience treatment on discount rate of whites

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if white == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2BWhiteBothRounds.doc) replace

***Table 2Bii: Effect of race-salience treatment on discount rate of blacks

asdoc intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if black == 1, cluster(id) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable2BBlackBothRounds.doc) replace



log close
