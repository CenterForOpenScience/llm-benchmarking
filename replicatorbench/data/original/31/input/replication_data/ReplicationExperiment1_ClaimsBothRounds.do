


clear
set mem 100m
set more off

log using "/Users/Victor/Downloads/Original data 3/Stata-log-files/REPExperiment1_ClaimsBothRounds.log", replace



****loading variables for Experiment 1 

import delimited using "/Users/victor/Documents/BenRepBothRoundsIR.csv", clear

////Section III



///PartA


//Body

****Claim: "Participants were Harvard College undergraduates, 71 of Asian descent and 66 of white descent." 

***Variable definition: raceA == 1 for Asian, raceW == 1 for white

generate white = nonasian*nonblack


tab asian white, missing 

generate id = v1



///Part C



****Claim: "Running separate regressions for each intertemporal choice type (immediate payment amount x time horizon) reveals that this treatment effect is statistically significant at the 1% level and of similar magnitude for all four types."

***reshaping the dataset, so each observation corresponds to a subject and intertemporal choice type

rename lowerboundpat lnNOratelo1
rename upperboundpat lnNOratehi1
rename lowerboundpat3 lnNOratelo2
rename upperboundpat3 lnNOratehi2

rename lowerboundpat2 lnOTratelo1
rename upperboundpat2 lnOTratehi1
rename lowerboundpat4 lnOTratelo2
rename upperboundpat4 nOTratehi2

rename patim1 impatientNO1
rename patim3 impatientNO2
rename patim2 impatientOT1
rename patim4 impatientOT2

reshape long lnNOratelo lnNOratehi lnOTratelo lnOTratehi impatientNO impatientOT, i(id) j(stakes)
gen largestakes = (stakes == 2)

rename lnNOratelo lndiscratelo1
rename lnNOratehi lndiscratehi1
rename impatientNO impatient1

rename lnOTratelo lndiscratelo2
rename lnOTratehi lndiscratehi2
rename impatientOT impatient2

reshape long lndiscratehi lndiscratelo impatient, i(id stakes) j(term)
gen longterm = (term == 2)
gen largelong = largestakes*longterm

intreg lndiscratelo lndiscratehi givenprimingquestionnaire if asian == 1 & largestakes == 0 & longterm == 0
intreg lndiscratelo lndiscratehi givenprimingquestionnaire if asian == 1 & largestakes == 1 & longterm == 0
intreg lndiscratelo lndiscratehi givenprimingquestionnaire if asian == 1 & largestakes == 0 & longterm == 1
intreg lndiscratelo lndiscratehi givenprimingquestionnaire if asian == 1 & largestakes == 1 & longterm == 1


//Footnote 10
****Claim: "In a regression that includes an interaction between identity salience and an indicator for having lived in the U.S. for fewer than 2 generations, the interaction term is insignificantly negative"

***including interaction term between ethnicity-salience treatment and immigrant (under 2 generations in US) in baseline discounting regression for Asians

intreg lndiscratelo lndiscratehi givenprimingquestionnaire treateimmigrant largestakes longterm largelong if asian == 1, cluster(id)

save "/Users/victor/Downloads/Original data 3/REPExperiment1DataBothRounds.dta", replace




log close
