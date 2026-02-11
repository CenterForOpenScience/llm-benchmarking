clear
* cd "\\file\Usersw$\wrr15\Home\My Documents\My Files\COS DATA-FINDING PROJECT\RESULTS\DATA ANALYSIS PROJECTS\Gelfand_covid_R3eV_6708"
log using "Gelfand_Results", replace
import delimited "gelfand_replication_data.csv"
describe
destring, replace ignore(NA)
describe
encode country, generate(countryname)
describe
gsort country year month day
tabulate country



/*
FROM THE DATA FINDER, SECTION 12B

The discrepancy between 57 countries in the preprint and 63 countries in the csv 
is likely accounted for by the exclusion of 6 countries that the corresponding 
author noted via email: Belgium, France, New Zealand, Norway, Pakistan, and Venezuela. 

For these six countries, the data collection that produced the tightness measure 
was conducted earlier than for the rest of the countries. The corresponding author 
recommended excluding the six countries from the replication attempt, as they 
did in the original analysis.
*/


// As a result of the comment above, I drop these countries from the dataset.
drop if country == "Belgium" | country == "France" | country == "New Zealand" | country == "Norway" | country == "Pakistan" | country == "Venezuela"
codebook country

* the ECDC dataset had some missing dates - these were typically days at the beginning, when no cases were recorded - we now add these and impose zero cases on those dates

gen date1=date(date,"YMD")
tsset countryname date1

* tsfill adds empty rows for missing dates
tsfill 
* if cases are zero, it means totals remain the same as days before
replace total_covid_per_million= l.total_covid_per_million if total_covid_per_million==.
replace gdp= l.gdp if gdp==.



// This section creates country ids that I will use in matching
gen t = _n
tsset t
gen id = 1
replace id = cond(gdp == L.gdp, L.id, L.id+1) in 2/l

// This section keeps those observations that have more 1 or more cases per million population
keep if total_covid_per_million > 1

// I then take the log of total cases.
gen ltotalcases = log(total_covid_per_million)
sum ltotalcases

// This creates a time variable for each country
gsort countryname date1
by countryname: gen time = _n
codebook time

// This drops observations beyond 30 days
drop if time > 30

// Note that all countries (57) have the same number of days
tabulate time

// This section replaces missing values of gini_val with the alternative gini values
// The new variable is gini, and it has no missing values.
gen gini = gini_val
replace gini = alternative_gini if gini_val == .

preserve

// This creates county specific dummy variables so I can estimate
// country specific exponential growth regressions
*tabulate country, gen(countryid)
tabulate countryname, gen(countryid)

// This estimates country-specific exponential growth regression
matrix coeffs = J(57,1,.)
matrix names = J(57,1,.)
forvalues i = 1/57 {
	reg ltotalcases time if countryid`i' == 1
	matrix coeffs[`i',1] = _b[time]
	matrix names[`i',1] = `i'
}

// This turns the vector of estimated coefficients and the vector of country IDsinto
// into the variables "coeffs1" and "names1"
svmat coeffs
svmat names
sum coeffs1 names1

keep coeffs1 names1
// This next line gets rid of all obs after 57 (because they are filled with missing values)
drop if _n > 57
// I save the estimated coefficients into a separate file for a later merge
save estimatedcoefficients, replace

restore

// This saves one observation for each country
by countryname: gen number = _n
keep if number == 1

// To get the total number of cases in the original study take mean and multiply by 57
sum obs_count_original
scalar totalobs_original_study = r(mean)*57

// To get the total number of cases in the replication take 30 and multiply by 57
scalar totalobs_replication = 30*57
scalar list totalobs_original_study totalobs_replication


// By changing the name of id to names1, I can use it for merging
rename id  names1

// This mergest the file of estimated coefficients with the main dataset
merge 1:1 names1 using estimatedcoefficients
summ

// This creates the interaction term
gen eff_tight = efficiency*tightness


regress coeffs1 eff_tight gdp gini median_age efficiency tightness

log close