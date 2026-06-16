***********
** Replication analysis A plumbers
** Task 2
** 2022-05-25
** Nicole Schwitter nicole.schwitter.1@warwick.ac.uk
***********
/*
Claim: The average plumbing firm whose name begins with A or a number receives... more service complaints than other firms.

Instructions
Your analysis should produce a single, main result in terms of statistical families of z-, t-, F-, or χ² tests (or their alternative or non-parametric versions).
Use only the data of Illinois Plumbing Firms in your model. Do not restrict your sample to those firms that serve the metro Chicago area. Do not include the reviews received via yelp.com, Angie's List, and Consumer's Checkbook into your analysis. You should use the overall complaints number instead of complaints per employee.
If your first analysis happened to satisfy all these instructions, you can use that for completing Task 2.
*/

//Read in and describe data
use final_data.dta, clear

//tests of association
ttest complaints_2008, by(first_A) 

//Negative binomial
nbreg complaints_2008 first_A 
display e(df_m)
display %18.0g _b[first_A ]/_se[first_A ] //get more decimal points