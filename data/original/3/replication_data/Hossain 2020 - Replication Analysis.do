*****************************************
* This do-file replicates a research claim from Mohammad Alamgir Hossain (2020)
* "Is the spread of COVID-19 across countries influenced by environmental, economic and social factors?"
* H*:  At the country level, the democracy index will be positively associated with 
* the total number of confirmed infections per one million people.
*
*****************************************
version 15.1
clear all

cd "..." /*Change directory*/

*Start a log file
local log_name "Hossain_Replication" /*Give a name to the log where session is recorded*/
log using `log_name', replace

use "COVID replication.dta" /*Import provided replication dataset*/

local dataset = 1
*Value dataset = 1 COVID cases at time frame of the original study (i.e. cases of infection per one million people on April 03 2020)
*Value dataset = 2 COVID cases at time frame after the original study was conducted (i.e. cases of infection per one million people after April 03 2020)
*Value dataset = 3 COVID cases at whole frame (i.e. cases of infection per one million people between December 31 2019 and August 11 2020)
*Value dataset = 4 COVID cases at time frames not used in the original study 
*(i.e. cases of infection per one million people between December 31 2019 and August 11 2020; and cases of infection per one million people after April 03 2020)
*Value dataset = 5 all available time frames

if `dataset'==1 {
	*COVID_12_31_04_03 is confirmed COVID cases 12.31.2019 to 04.03.2020
	rename COVID_12_31_04_03 total_cases 
	}
else if `dataset'==2 {
	*COVID_04_04_08_11 is confirmed COVID cases 04.04.2020 to 08.11.2020
	rename COVID_04_04_08_11 total_cases 
	}
else if `dataset'==3 {
	*COVID_12_31_08_11 is confirmed COVID cases 12.31.2019 to 08.11.2020
	rename COVID_12_31_08_11 total_cases 
	}
else if `dataset'==4 {
	save temp.dta, replace
	*COVID_04_04_08_11 is confirmed COVID cases 04.04.2020 to 08.11.2020
	gen total_cases=COVID_04_04_08_11
	append using temp.dta
	*COVID_12_31_08_11 is confirmed COVID cases 12.31.2019 to 08.11.2020
	replace total_cases=COVID_12_31_08_11 if total_cases==.
	erase temp.dta
}	
else {
	save temp.dta, replace
	*COVID_12_31_04_03 is confirmed COVID cases 12.31.2019 to 04.03.2020
	gen total_cases=COVID_12_31_04_03
	append using temp.dta
	*COVID_04_04_08_11 is confirmed COVID cases 04.04.2020 to 08.11.2020
	replace total_cases=COVID_04_04_08_11 if total_cases==.
	append using temp.dta
	**COVID_12_31_08_11 is confirmed COVID cases 12.31.2019 to 08.11.2020
	replace total_cases=COVID_12_31_08_11 if total_cases==.
	erase temp.dta
}

********* Dependent variable: Cases of infection per one million people *********
*Page 10: "Total cases of infection are converted to cases per one million population 
*to capture the population effect. Cases of infection per one million people on 
*03 April 2020 by countries is denoted by Y and used as the dependent variable in our experimentation."

*Page 11: "Y is the total number of cases of confirmed infection per one million
*people in a country on a day (03 April 2020)"

*While in the original study the time frame is April 03 2020, in the replication it depends on the type of analysis chosen

*Use total COVID cases (total_cases) and population (popData2019) to construct total number of cases per million people
gen cases_per_million = total_cases/popData2019*1000000

*Page 11: "X1 is yearly average temperature of countries, 
* X2 is yearly average precipitation of countries, 
* X3 is openness measured by international trade as a percentage of GDP of countries
* X4 is democracy index of countries and 
* X5 is population density of countries in 2018.
*Page 13: "We apply Least Squares method on model (1) and find 
*that precipitation and population density have no significant effect on 
*the number of infection cases per one million people (Y). Those variables 
*are then excluded, and the model is re-estimated"

********************** Focal independent variable: Democracy **********************
*Use Democracy for the democracy index.
rename Democracy democracy
*Note The Economist publishes the index with a scale from 0 to 10. However, Gapminder (the data source for democracy index) 
*has converted the index from 0 to 100 to make it easier to communicate as a percentage.
*See https://www.gapminder.org/data/documentation/democracy-index/
*Because the author of the original study used the scores of democracy index compiled by Economist 
*Intelligence Unit (page 9), the data obtained in Gapminder is divided by 10
replace democracy = democracy/10

************************ Other controls ************************
*1) Yearly average temperature of countries:
*Variable: Annual_temp
*Annual temperature (averaged 1961 to 1999)
*Continuous variable 
rename Annual_temp temperature

*2) Openness measured by international trade as a percentage of GDP of countriess:
*Variable: trade_2016
*International trade (imputed)
*Continuous variable 
rename trade_2016 openness

************************ Test of the SCORE claim, H*  ************************
*Page 11: "We first simply regress total number of cases of infection per one 
*million people by countries reported on a recent day (03 April 2020) on our 
*selected explanatory variables. Specifically, we estimate the following regression:

* Y = \delta + \mu_{1}X1 + \mu_{2}X2 + \mu_{3}X3 + \mu_{4}X4 + \mu_{5}X5 + \nu (1)

*Y is the total number of cases of confirmed infection per one million people in 
*a country on a day (03 April 2020), X1 is yearly average temperature of countries,
*X2 is yearly average precipitation of countries, X3 is openness measured by 
*international trade as a percentage of GDP of countries, X4 is democracy index 
*of countries and X5 is population density of countries in 2018.

*Page 13: "We apply Least Squares method on model (1) and find that precipitation 
*and population density have no significant effect on the number of infection cases 
*per one million people (Y). Those variables are then excluded, and the model is re-estimated"

*Control variables
*1)Temperature
*2)Openness
local covariates "temperature openness"

*Regression model
reg cases_per_million democracy `covariates'

*Close log
log close
*Create PDF from log
translate `log_name'.smcl `log_name'.pdf, replace

display `End of Do-file'
