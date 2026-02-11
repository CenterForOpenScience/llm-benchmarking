cd "C:\Users\fedor\OneDrive\Documents\DOKUMENTUMOK\Reproducibility Project\SCORE\Malik2020_replication_using.existing.datasets\data\replication_from.new.data"

* Start log file
log using "results_new.log"

* Import data
import delimited "replicationDataset_Malik2020_with.year.csv", varnames(1) case(preserve)   clear 

* The "date" variable is a string -> make it to a date type variable called date2
generate date2=date(date,"MDY")

* Take 5% random sample of the observations
sample 5 

* Focal analysis: Multilevel mixed-effects linear regression model to estimate the effect of time and governmental social distancing measures on mobility.
xtmixed CMRT_transit date2 lockdown ||city:, var

* Additional analysis: Multilevel mixed-effects linear regression model to estimate the effect of time and governmental social distancing measures on people staying at home. 
xtmixed CMRT_residential date2 lockdown ||city:, var

* Close log file
log close
