* Import the data file from the folder (alter file path to your specific machine)
import delimited "C:\Users\Christopher\Dropbox\Replications\Kollmeyer\data\finaldata_noNA.csv"
* Create a STATA-required non-string value for countries
encode country, gen(countrynum)

* Setup the panel ID
xtset countrynum

* Set the time variable for the panel
xtset countrynum year, yearly

* Define National Affluence as in the paper
gen NAff = gdp/pop

* Define Imports from South as in the paper
gen IMS = totalimport/(gdp*10000)

* Define Exports to South as in the paper
gen EXS = totalexport/(gdp*10000)

* Detect outliers using Hadi outlier detection as in the paper
hadimvo NAff IMS EXS unemp, gen(bad)

* Command drops observations tagged as outliers 
drop if bad == 1

* Retain only the columns necessary for estimation 
drop country countryyear gdp pop totalimport totalexport bad

* Include new 5-year time dummies to account for new observations added before 1970
gen DUM70to74 = 0
replace DUM70to74 = 1 if year >= 1970 & year <= 1974

* Generate 5-year time dummies as in the paper
gen DUM75to79 = 0
replace DUM75to79 = 1 if year >= 1975 & year <= 1979
gen DUM80to84 = 0
replace DUM80to84 = 1 if year >= 1980 & year <= 1984
gen DUM85to89 = 0
replace DUM85to89 = 1 if year >= 1985 & year <= 1989
gen DUM90to94 = 0
replace DUM90to94 = 1 if year >= 1990 & year <= 1994
gen DUM95to99 = 0
replace DUM95to99 = 1 if year >= 1995 & year <= 1999

* Include new 5-year time dummies to account for new observations added after 2003
gen DUM00to04 = 0
replace DUM00to04 = 1 if year >= 2000 & year <= 2004
gen DUM05to09 = 0
replace DUM05to09 = 1 if year >= 2005 & year <= 2009
gen DUM10to14 = 0
replace DUM10to14 = 1 if year >= 2010 & year <= 2014
gen DUM15to18 = 0
replace DUM15to18 = 1 if year >= 2015 & year <= 2018

* Re-order panel according to year - required to enable the lag operator "L.x"
sort countrynum year

** Uncomment the following set of commands to estimate the following FGLS model (without controls) and then verify the presence of serial autocorrelation in the residuals, spatial correlation, and groupwise heteroskedasticity:
*xtgls NAff L.IMS L.EXS L.unemp i.countrynum DUM70to74 DUM75to79 DUM80to84 DUM85to89 DUM90to94 DUM95to99 DUM00to04 DUM05to09 DUM10to14 DUM15to18
*xtserial
*xttest2
*xttest3

* Re-estimate the model controlling for autocorrelation w/in panels, cross-sectional correlation, and heteroskedasticity across panels
xtgls NAff L.IMS L.EXS L.unemp i.countrynum DUM70to74 DUM75to79 DUM80to84 DUM85to89 DUM90to94 DUM95to99 DUM00to04 DUM05to09 DUM10to14 DUM15to18, panels(hetero) corr(psar1) force
 