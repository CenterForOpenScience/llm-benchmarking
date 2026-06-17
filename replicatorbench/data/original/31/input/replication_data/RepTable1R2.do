log using "/Users/Victor/Downloads/Original data 3/Stata-log-files/REPTable1R2.log", replace


clear
set mem 100m
set more off



***loading variables for Experiment 1

use "/Users/Victor/Downloads/Original data 3/REPExperiment1DataR2.dta"



//Percent of Impatient or Safe Choices



***summarizing percentage of impatient choices

**Asians
asdoc su impatient if asian == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianControlPatR2.doc) replace
asdoc su impatient if asian == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianPrimedPatR2.doc) replace
asdoc ttest impatient if asian == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianEthnicPatR2.doc) replace

**Whites
asdoc su impatient if white == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteControlPatR2.doc) replace
asdoc su impatient if white == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhitePrimedPatR2.doc) replace
asdoc ttest impatient if white == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteEthnicPatR2.doc) replace

**Blacks
asdoc su impatient if black == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackControlPatR2.doc) replace
asdoc su impatient if black == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackPrimedPatR2.doc) replace
asdoc ttest impatient if black == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackEthnicPatR2.doc) replace

***summarizing percentage of safe choices

**Asians
asdoc su risksafee if asian == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianControlRiskR2.doc) replace
asdoc su risksafee if asian == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianPrimedRiskR2.doc) replace
asdoc ttest risksafee if asian == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianEthnicRiskR2.doc) replace

**Whites
asdoc su risksafee if white == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteControlRiskR2.doc) replace
asdoc su risksafee if white == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhitePrimedRiskR2.doc) replace
asdoc ttest risksafee  if white == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteEthnicRiskR2.doc) replace

**Blacks
asdoc su risksafee if black == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackControlRiskR2.doc) replace
asdoc su risksafee if black == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackPrimedRiskR2.doc) replace
asdoc ttest risksafee  if black == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackEthnicRiskR2.doc) replace

log close
