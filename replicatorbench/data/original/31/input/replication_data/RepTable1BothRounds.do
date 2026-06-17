log using "/Users/Victor/Downloads/Original data 3/Stata-log-files/REPTable1R2.log", replace


clear
set mem 100m
set more off



***loading variables for Experiment 1

use "/Users/Victor/Downloads/Original data 3/REPExperiment1DataBothRounds.dta"



//Percent of Impatient or Safe Choices



***summarizing percentage of impatient choices

**Asians
asdoc su impatient if asian == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianControlPatBothRounds.doc) replace
asdoc su impatient if asian == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianPrimedPatBothRounds.doc) replace
asdoc ttest impatient if asian == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianEthnicPatBothRounds.doc) replace

**Whites
asdoc su impatient if white == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteControlPatBothRounds.doc) replace
asdoc su impatient if white == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhitePrimedPatBothRounds.doc) replace
asdoc ttest impatient if white == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteEthnicPatBothRounds.doc) replace

**Blacks
asdoc su impatient if black == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackControlPatBothRounds.doc) replace
asdoc su impatient if black == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackPrimedPatBothRounds.doc) replace
asdoc ttest impatient if black == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackEthnicPatBothRounds.doc) replace

***summarizing percentage of safe choices

**Asians
asdoc su risksafee if asian == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianControlRiskBothRounds.doc) replace
asdoc su risksafee if asian == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianPrimedRiskBothRounds.doc) replace
asdoc ttest risksafee if asian == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1AsianEthnicRiskBothRounds.doc) replace

**Whites
asdoc su risksafee if white == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteControlRiskBothRounds.doc) replace
asdoc su risksafee if white == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhitePrimedRiskBothRounds.doc) replace
asdoc ttest risksafee  if white == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1WhiteEthnicRiskBothRounds.doc) replace

**Blacks
asdoc su risksafee if black == 1 & givenprimingquestionnaire == 0, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackControlRiskBothRounds.doc) replace
asdoc su risksafee if black == 1 & givenprimingquestionnaire == 1, save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackPrimedRiskBothRounds.doc) replace
asdoc ttest risksafee  if black == 1, by(givenprimingquestionnaire) save(/Users/Victor/Downloads/Original data 3/ReplicationResults/RepTable1BlackEthnicRiskBothRounds.doc) replace

log close
