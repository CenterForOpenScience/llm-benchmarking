***********
** Replication analysis A plumbers
** 2022-05-10
** Nicole Schwitter nicole.schwitter.1@warwick.ac.uk
***********
/*
Claim: The average plumbing firm whose name begins with A or a number receives... more service complaints than other firms.
*/

//Read in and describe data
use final_data.dta, clear
browse

tab chicago first_A 
tab multiple_names first_A
sum complaints_2008, detail
tab complaints_2008

gen complaints_2008_bin = complaints_2008 > 0 //create binary variable

//tests of association
ttest complaints_2008, by(first_A) 
tab complaints_2008_bin first_A, chi

****Multivariate models
//Logit
logit complaints_2008_bin first_A 
logit complaints_2008_bin first_A ad_spend_k firm_age chicago emp_size multiple_names

//Poisson
poisson complaints_2008 first_A 
poisson complaints_2008 first_A ad_spend_k firm_age chicago emp_size multiple_names
estat gof //Overdispersion, standard poisson model might be inappropriate

//Zero inflated model
zip complaints_2008 first_A, inflate(first_A)
zip complaints_2008 first_A ad_spend_k firm_age chicago emp_size multiple_names, inflate(first_A ad_spend_k firm_age chicago emp_size multiple_names)
margins, at(first_A=(0 1)) atmeans
 
//Negative binomial
nbreg complaints_2008 first_A 
nbreg complaints_2008 first_A ad_spend_k firm_age chicago emp_size multiple_names
margins, at(first_A=(0 1)) atmeans 

