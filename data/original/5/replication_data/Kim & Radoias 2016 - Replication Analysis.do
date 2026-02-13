*****************************************
* This do-file replicates a research claim from Kim & Radoias (2016)
* "Education, individual time preferences, and asymptomatic disease detection"
* H*: Among the sample of respondents in poor general health who were found to
* be hypertensive during a screening, the probability of being undiagnosed 
* decreases with education.
*
*****************************************
version 15.1
clear all

cd "..." /*Change directory*/

*Start a log file
local log_name "Kim-Radoias_Replication" /*Give a name to the log where session is recorded*/
log using `log_name', replace

use replication_data.dta /*Import provided replication dataset*/

************************ Dependent variable: Being under-diagnosed ************************
*“Our dependent variable is a dummy equal to one for those respondents who were 
*found to be hypertensive during the IFLS screenings, but were not previously 
*diagnosed by a doctor.” (p. 18)

/*"As part of the survey, trained nurses measured respondents' blood
pressure three different times. The first measurement was dropped
because many people get nervous at first which can cause false high
measurements. We then used the average of the other two measurements
to construct the hypertension variable. Following WHO
standards, a person is considered hypertensive if his systolic is
greater than 140 or his diastolic is greater than 90. IFLS also asked
respondents if they had ever been diagnosed with hypertension
before. Therefore, the under-diagnosed respondents were those
who were found to be hypertensive during IFLS survey measurements,
but had not been previously diagnosed by doctors." (page 18)*/

*First, generate a dummy variable equal to 1 if hypertension and 0 otherwise
*Generate an average of the last two blood pressure measurements:
gen systolic = (us07b1 + us07c1)/2 /*systolic*/
gen diastolic = (us07b2 + us07c2)/2 /*diastolic*/
gen hypertension = 0
replace hypertension =1 if systolic>140 | diastolic>90
*Replace hypertension as missing if the blood preassure measurement is not available
replace hypertension = . if us07b1==. | us07c1==. | us07b2==. | us07c2==.

*Generate a dummy equal to one for those respondents who were 
*found to be hypertensive during the IFLS screenings, but were not previously 
*diagnosed by a doctor 
gen under_diag=0
replace under_diag=1 if hypertension ==1 & cd05==3
*Replace under_diag as missing if hypertension is not available, or if the 
*diagnosis variable is missing or the respondent don't know  (=8)
replace under_diag=. if hypertension ==. | cd05==. | cd05==8
 
********************** Focal independent variable: Years of Education **********************
*"Explanatory variables include respondents' education (measured in years of formal education)" (Page 18)
gen yrs_school=.
*Replace years of education equal to zero if respondent never attended school (=3).
replace yrs_school=0 if dl04==3

*Use variable "Highest level of education attended" (dl06) and dl07 
*"What is the highest grade completed at that school?" to construct years of education
*for people who attended an educational institution
*Missing values
replace dl07=. if dl07==98

*To understand indonesian educational system 
*https://wenr.wes.org/2019/03/education-in-indonesia-2#:~:text=Elementary%20education%20(pendidikan%20dasar)%20lasts,%2C%20arts%2C%20and%20physical%20education.
*The information coincides with information from:
*https://en.wikipedia.org/wiki/Education_in_Indonesia

*"Elementary education (pendidikan dasar) lasts for six years (grades one to six)"

*"Lower or junior secondary education lasts for three years (grades seven to nine)"

*"Senior secondary education is presently neither compulsory nor free. It lasts three years (grades 10 to 12) "

*1. Elementary
*Elementary (six years)
*Replace six years for people that completed elementary education
replace dl07=6 if dl07==7 & dl06==2
*Generate years of school for Elementary
replace yrs_school = dl07 if dl06==2

*Adult Education A (one year)
replace dl07=1 if dl07==7 & dl06==11
replace yrs_school = dl07 if dl06==11

*School for Disabled. According to documentation in Page 82 "School for disabled"
*could be either for elementary, junior high school or senior high school. 
*Given that is not possible to identify which education category level corresponds, 
*treat "School for disabled" as missing

*Islamic Elementary School (Madrasah Elementary) (six years)
replace dl07=6 if dl07==7 & dl06==72
replace yrs_school = dl07 if dl06==72

*Other. According to documentation in Page 82 "Other"
*could be either for elementary, junior high school, senior high school, 
*or D1, D2, D3//University.  Given that is not possible to identify 
*which education category level corresponds, treat "Other" as missing

*2. Junior High
*Junior High (three years)
replace dl07=3 if dl07==7 & dl06==3
replace yrs_school = dl07 + 6 if dl06==3 /*Add 6 years from Elementary*/

*Junior High Vocational (three years)
replace dl07=3 if dl07==7 & dl06==4
replace yrs_school = dl07 + 6 if dl06==4 /*Add 6 years from Elementary*/

*Adult Education B (four years)
replace dl07=4 if dl07==7 & dl06==12
replace yrs_school = dl07 + 1 if dl06==12 /*Add 1 year from Adult Education A*/

*Islamic Junior/High School (Madrasah Senior High School) (three years) 
replace dl07=3 if dl07==7 & dl06==73
replace yrs_school = dl07 + 6 if dl06==73 /*Add 6 years from Elementary*/

*Recall that "School for Disabled" and "Other" cannot be identified

*3. Senior High
*Senior High (three years)
replace dl07=3 if dl07==7 & dl06==5
replace yrs_school = dl07 + 9 if dl06==5 /*Add 9 years from Elementary plus Junior High*/

*Senior High Vocational (four years)
replace dl07=4 if dl07==7 & dl06==6
replace yrs_school = dl07 + 9 if dl06==6 /*Add 9 years from Elementary plus Junior High*/

*Adult Education C (three years)
replace dl07=3 if dl07==7 & dl06==15
replace yrs_school = dl07 + 5 if dl06==15 /*Add 5 years from Adult Education A and B*/

*Islamic Senior/High School (Madrasah Aaliyah) (three years) 
replace dl07=3 if dl07==7 & dl06==74
replace yrs_school = dl07 + 9 if dl06==74 /*Add 9 years from Elementary plus Junior High*/

*Recall that "School for Disabled" and "Other" cannot be identified

*4. D1, D2, D3, University
*College  (three years)
replace dl07=3 if dl07==7 & dl06==60
replace yrs_school = dl07 + 12 if dl06==60 /*Add 12 years from Elementary plus Junior High plus Senior High*/

*University S1 (Bachelor)  (four years)
replace dl07=4 if dl07==7 & dl06==61
replace yrs_school = dl07 + 12 if dl06==61 /*Add 12 years from Elementary plus Junior High plus Senior High*/

*University S2 (Master) (three years)
replace dl07=3 if dl07==7 & dl06==62
replace yrs_school = dl07 + 16 if dl06==62 /*Add 16 years from Elementary plus Junior High plus Senior High plus Bachelor*/

*University S3 (Doctorate) (five years)
replace dl07=5 if dl07==7 & dl06==63
replace yrs_school = dl07 + 16 if dl06==63 /*Add 16 years from Elementary plus Junior High plus Senior High plus Bachelor*/

*Open University (Six years)
replace dl07=6 if dl07==7 & dl06==13
replace yrs_school = dl07 + 12 if dl06==13 /*Add 12 years from Elementary plus Junior High plus Senior High*/

*Replace yrs_school=0 for Kindergarten
replace yrs_school=0 if dl06==90

*Replace "yrs_school" as missing for dl06=14 (Islamic School (pesantren)), dl06=98 ("Don't Know") and dl06=99 ("MISSING")
replace yrs_school=. if inlist(dl06, 14, 98, 99)

************************ Other controls ************************
/* Explanatory variables include respondents' education (measured
in years of formal education), respondents' age and age squared
(to allow for possible non-linear effects), respondents' individual
risk and time preferences, the distance from the closest health
center (to proxy for the ease of access to medical care), household
per capita expenditures (PCE), and a sex dummy. (p. 18)*/

*1) Age:
*Variable: ar09
*Respondents' age
*Description: continuous variable. 
drop age
gen age = ar09
*Replace as missing the maximum value of 998 years 
replace age=. if ar09==998
*Replace age as missing if ar09 is missing 
replace age=. if ar09==.

*2) Age squared:
*Variable: ar09
*Respondents' age squared
*Description: continuous variable. The maximum value of 998 is considered as a missing value 
gen agesqrt = ar09^2
*Replace as missing the maximum value of 998 years 
replace agesqrt=. if ar09==998
*Replace agesqrt as missing if ar09 is missing 
replace agesqrt=. if ar09==.

*3) Risk preferences
*Variable: si01, si02, si03, si04, si05
*Series of connected and branching questions on lottery choices.

/*For the time and risk preference parameters, we follow Ng (2013) and group
respondents in four distinct groups from the most patient to the
most impatient, respectively from the least risk averse to the most
risk averse. (p.18)

According to Ng "Risk and Time Preferences in Indonesia: The Role of
Demographics, Cognition, and Interviewers"

In Section 2.2, Ng explains how to construct a risk aversion measure based on 
respondents' certainty equivalent at the termination of the lottery questions. 

"The terminal node therefore represent an ordinal ranking of risk aversion among the respondents.
Respondents with risk aversion = 4 are the most risk averse, and those with risk aversion = 1
are least risk averse" (page 9 Ng). 

See Figure 1 (page 28 Ng) for details on the construction of Risk A and Risk B

Ng constructs two measures of risk aversion, that he calls Risk A and Risk B. 
Each measure depends on the two sets of questions that were asked. 
The two sets differ in the magnitude of the payoffs and the variance of their expected payoffs.
*/

*Risk A
gen risk_A=.
/*Ng includes a category of "Gamble averse" participants: participants who 
always choose the safe option. No willingness to take on risk*/
*Most risk averse (Category 4):
replace risk_A = 4 if (si01==1 | si02==2) & si03==1 & si04==1
*Category 3:
replace risk_A = 3 if (si01==1 | si02==2) & si03==1 & si04==2
*Category 2:
replace risk_A = 2 if (si01==1 | si02==2) & si03==2 & si05==1
*Least risk averse (Category 1):
replace risk_A = 1 if (si01==1 | si02==2) & si03==2 & si05==2

*Replace as missing if si01==8, si02==8, si03==8, si04==8, si05==8 (Don't Know)
replace risk_A=. if si01==8 | si02==8 | si03==8 | si04==8 | si05==8

*Risk B
gen risk_B=.
*Most risk averse (Category 4):
replace risk_B = 4 if (si11==1 | si12==2) & si13==1 & si14==1
*Category 3:
replace risk_B = 3 if (si11==1 | si12==2) & si13==1 & si14==2
*Category 2:
replace risk_B = 2 if (si11==1 | si12==2) & si13==2 & si15==1
*Least risk averse (Category 1):
replace risk_B = 1 if (si11==1 | si12==2) & si13==2 & si15==2

*Replace as missing if si11==8, si12==8, si13==8, si14==8, si15==8 (Don't Know)
replace risk_B=. if si11==8 | si12==8 | si13==8 | si14==8 | si15==8

/*Although Kim & Radoias use Ng method to elicit risk preferences, it is not clear 
how they merge Risk A and Risk B variables in a single Risk Preferences measure consisting of four groups. 
To group respondents in four distinct groups take the average between Risk A and Risk B and round to the closest integer*/
gen risk_preference = (risk_A + risk_B)/2
replace risk_preference =round(risk_preference)

*4) Time Preferences
*Variable: si01, si02, si03, si04, si05
*Series of connected and branching questions on intertemporal choices.

/*For the time and risk preference parameters, we follow Ng (2013) and group
respondents in four distinct groups from the most patient to the
most impatient, respectively from the least risk averse to the most
risk averse. (p.18)

According to Ng "Risk and Time Preferences in Indonesia: The Role of
Demographics, Cognition, and Interviewers"

In Section 2.3, Ng explains how to construct a time preference measure based on
respondents' choices at the termination of the intertemporal choice questions. 

"The terminal node therefore represent an ordinal ranking of risk aversion among the respondents.
Respondents with risk aversion = 4 are the most risk averse, and those with risk aversion = 1
are least risk averse" (page 9 Ng). 

At the end he constructs two measures of time preferences, that he calls Time A and Time B. 
Each measure depends on the two sets of questions were asked.

See Figure 2 (page 29 Ng) for details on the construction of Time A and Time B
*/
/*Ng includes a category of "Negative time discounters" participants: participants who 
chose to defer receiving money without compensation*/
gen time_A=.
*Least patient (Category 4):
replace time_A = 4 if (si21a==1 | si21e==3) & si21b==1 & si21c==1
*Category 3:
replace time_A = 3 if (si21a==1 | si21e==3) & si21b==1 & si21c==2
*Category 2:
replace time_A = 2 if (si21a==1 | si21e==3) & si21b==2 & si21d==1
*Most patient (Category 1):
replace time_A = 1 if (si21a==1 | si21e==3) & si21b==2 & si21d==2

*Replace as missing if si21a==9, si21e==9, si21b==9, si21c==9, si21d==9 (Don't Know)
replace time_A=. if si21a==9 | si21e==9 | si21b==9 | si21c==9 | si21d==9

gen time_B=.
*Least patient (Category 4):
replace time_B = 4 if (si22a==1 | si22e==3) & si22b==1 & si22c==1
*Category 3:
replace time_B = 3 if (si22a==1 | si22e==3) & si22b==1 & si22c==2
*Category 2:
replace time_B = 2 if (si22a==1 | si22e==3) & si22b==2 & si22d==1
*Most patient (Category 1):
replace time_B = 1 if (si22a==1 | si22e==3) & si22b==2 & si22d==2

*Replace as missing if si22a==9, si22e==9, si22b==9, si22c==9, si22d==9 (Don't Know)
replace time_B=. if si22a==9 | si22e==9 | si22b==9 | si22c==9 | si22d==9

/*Although Kim & Radoias use Ng method to elicit time preferences, it is not clear 
how they merge Time A and Time B variables in a single Time Preferences measure consisting of four groups. 
To group respondents in four distinct groups take the average between Time A and Time B and round to the closest integer*/
gen time_preference = (time_A + time_B)/2
replace time_preference =round(time_preference)

*5) Distance to Health Center
*Variable: rj11
*Distance to medical facility
/*NOTE: The variable might have serious limitations (See Section 12 in Preregistration)
*For example, this was only asked of respondents who had visited a medical provider 
in the last four weeks. Further, only respondents who knew the distance have a 
value for this variable.*/
gen distance = rj11
*Replace as missing if rj11x==8 (Don't know)
replace distance =. if rj11x==8
*Replace distance as missing if rj11 is missing 
replace distance=. if rj11==.

*6) Log of household per capita expenditures (PCE)
*During the past week...
*Stable Foods
*Replace as missing if ==7 or ==8
replace ks02_ks1type_A=. if ks02x_ks1type_A==8
replace ks02_ks1type_B=. if ks02x_ks1type_B==7 | ks02x_ks1type_B==8
replace ks02_ks1type_C=. if ks02x_ks1type_C==8
replace ks02_ks1type_D=. if ks02x_ks1type_D==8
replace ks02_ks1type_E=. if ks02x_ks1type_E==8
gen staple_food = ks02_ks1type_A + ks02_ks1type_B + ks02_ks1type_C + ks02_ks1type_D + ks02_ks1type_E
*Vegetables
*Replace as missing if ==8
replace ks02_ks1type_F=. if ks02x_ks1type_F==8
replace ks02_ks1type_G=. if ks02x_ks1type_G==8
replace ks02_ks1type_H=. if ks02x_ks1type_H==8
gen vegetables = ks02_ks1type_F + ks02_ks1type_G + ks02_ks1type_H
*Dried foods
*Replace as missing if ==8
replace ks02_ks1type_I=. if ks02x_ks1type_I==8
replace ks02_ks1type_J=. if ks02x_ks1type_J==8
gen dried = ks02_ks1type_I + ks02_ks1type_J
*Meat and Fish
*Replace as missing if ==8
replace ks02_ks1type_K=. if ks02x_ks1type_K==8
replace ks02_ks1type_L=. if ks02x_ks1type_L==8
replace ks02_ks1type_M=. if ks02x_ks1type_M==8
replace ks02_ks1type_N=. if ks02x_ks1type_N==8
gen meat_fish = ks02_ks1type_K + ks02_ks1type_L + ks02_ks1type_M + ks02_ks1type_N
*Other dishes
*Replace as missing if ==8
replace ks02_ks1type_OA=. if ks02x_ks1type_OA==8
replace ks02_ks1type_OB=. if ks02x_ks1type_OB==8
gen other_dishes = ks02_ks1type_OA + ks02_ks1type_OB
*Milk/Eggs 
*Replace as missing if ==5 or ==8
replace ks02_ks1type_P=. if ks02x_ks1type_P==8
replace ks02_ks1type_Q=. if ks02x_ks1type_Q==5 | ks02x_ks1type_Q==8
gen milk_eggs = ks02_ks1type_P + ks02_ks1type_Q
*Spices
*Replace as missing if ==8
replace ks02_ks1type_R=. if ks02x_ks1type_R==8
replace ks02_ks1type_S=. if ks02x_ks1type_S==8
replace ks02_ks1type_T=. if ks02x_ks1type_T==8
replace ks02_ks1type_U=. if ks02x_ks1type_U==8
replace ks02_ks1type_V=. if ks02x_ks1type_V==8
replace ks02_ks1type_W=. if ks02x_ks1type_W==8
replace ks02_ks1type_X=. if ks02x_ks1type_X==8
replace ks02_ks1type_Y=. if ks02x_ks1type_Y==8
gen spices = ks02_ks1type_R + ks02_ks1type_S + ks02_ks1type_T + ks02_ks1type_U + ks02_ks1type_V + ks02_ks1type_W + ks02_ks1type_X + ks02_ks1type_Y
*Beverages
*Replace as missing if ==5 or ==8
replace ks02_ks1type_Z=. if ks02x_ks1type_Z==8
replace ks02_ks1type_AA=. if ks02x_ks1type_AA==8
replace ks02_ks1type_BA=. if ks02x_ks1type_BA==8
replace ks02_ks1type_CA=. if ks02x_ks1type_CA==8
replace ks02_ks1type_DA=. if ks02x_ks1type_DA==8
replace ks02_ks1type_EA=. if ks02x_ks1type_EA==8
replace ks02_ks1type_FA=. if ks02x_ks1type_FA==8
replace ks02_ks1type_GA=. if ks02x_ks1type_GA==8
replace ks02_ks1type_HA=. if ks02x_ks1type_HA==8
replace ks02_ks1type_IA=. if ks02x_ks1type_IA==5 | ks02x_ks1type_IA==8
replace ks02_ks1type_IB=. if ks02x_ks1type_IB==5 | ks02x_ks1type_IB==8
gen beverages = ks02_ks1type_Z + ks02_ks1type_AA + ks02_ks1type_BA + ks02_ks1type_CA + ///
ks02_ks1type_DA + ks02_ks1type_EA + ks02_ks1type_FA + ks02_ks1type_GA + ks02_ks1type_HA + ///
ks02_ks1type_IA + ks02_ks1type_IB

*During the past month
*Electricity
*Replace as missing if ==8
gen electricity = ks06_ks2type_A1
replace electricity=. if ks06x_ks2type_A1==8
*Water
*Replace as missing if ==8
gen water = ks06_ks2type_A2
replace electricity=. if ks06x_ks2type_A2==8
*Fuel
*Replace as missing if ==8
gen fuel = ks06_ks2type_A3
replace fuel=. if ks06x_ks2type_A3==8
*Telephone (including vouchers and mobile starter pack)
*Replace as missing if ==8
gen telephone = ks06_ks2type_A4
replace telephone=. if ks06x_ks2type_A4==8
*Personal toiletries
*Replace as missing if ==8
gen toiletries = ks06_ks2type_B
replace toiletries=. if ks06x_ks2type_B==8
*Household items
*Replace as missing if ==8
gen HH_items = ks06_ks2type_C
replace HH_items=. if ks06x_ks2type_C==8
*Domestic services and servants' wages
*Replace as missing if ==8
gen domestic_serv = ks06_ks2type_C1
replace domestic_serv=. if ks06x_ks2type_C1==8
*Recreation and Entertainment
*Replace as missing if ==8
gen recreation = ks06_ks2type_D
replace recreation=. if ks06x_ks2type_D==8
*Transportation
*Replace as missing if ==8
gen transportation = ks06_ks2type_E
replace transportation=. if ks06x_ks2type_E==8
*Sweepstakes and the like
*Replace as missing if ==8
gen sweepstakes = ks06_ks2type_F1
replace sweepstakes=. if ks06x_ks2type_F1==8
*Arisan
*Replace as missing if ==8
gen arisan = ks06_ks2type_F2
replace arisan=. if ks06x_ks2type_F2==8

*During the past one year
*Clothing for children and adults
*Replace as missing if ==8
gen clothing = ks08_ks3type_A
replace clothing=. if ks08x_ks3type_A==8
*Household supplies and furniture
*Replace as missing if ==8
gen HH_supplies = ks08_ks3type_B
replace HH_supplies=. if ks08x_ks3type_B==8
*Medical Costs
*Replace as missing if ==7 or ==8
gen medical_costs = ks08_ks3type_C
replace medical_costs=. if ks08x_ks3type_C==7 | ks08x_ks3type_C==8
*Ritual ceremonies, charities and gifts
*Replace as missing if ==7 or ==8
gen ritual = ks08_ks3type_D
replace ritual=. if ks08x_ks3type_D==7 | ks08x_ks3type_D==8
*Taxes
*Replace as missing if ==6 or ==8
gen taxes = ks08_ks3type_E
replace taxes=. if ks08x_ks3type_E==6 | ks08x_ks3type_E==8
*Other expenditures not specified above
*Replace as missing if ==8
gen other_exp = ks08_ks3type_F
replace other_exp=. if ks08x_ks3type_F==8

/*The expenditures categories for transfers given to other parties outside the 
household are not included in the estimation of total household expenditures. 
The reason is that it is not clear that it captures additional spending over and
above the other expenditures categories. This corresponds to the variables 
ks06_ks2type_G, ks06x_ks2type_G, ks08_ks3type_G, and ks08x_ks3type_G.*/

*Generate Monthly Expenditure per household
*Transform expenditures to monthly frequency. 
*However, not clear in the paper what the frequency of their per capita expenditures.
*4.34524 weeks in a month 
*12 months in a year
gen HH_Expenditure = 4.34524*(staple_food + vegetables + dried + meat_fish + other_dishes + ///
milk_eggs + spices + beverages) + (electricity + water + fuel + telephone + toiletries + ///
HH_items + domestic_serv + recreation + transportation + sweepstakes + arisan) + ///
(1/12)*(clothing + HH_supplies + medical_costs + ritual + taxes + other_exp)

*Transform to Logarithm
gen log_PCE = log(HH_Expenditure/hh_size)

*7) Female 
*Variable: ar07
*Respondent is a female (=1)
*Description: 1=male, 3=female
gen female = (ar07==3) 
*Replace female as missing if ar07 is missing 
replace female=. if ar07==.

************************ Sample Parameters ************************
*1) Poor Health:
*Variable: kk01
*Description: Generally how is your health?
*1 == Very healthy, 2 == Somewhat healthy, 3== Somewhat unhealthy, 4 == Very unhealthy
/*Respondents were asked to evaluate their general
health status (GHS) on a scale from 1 to 4. Depending on the answers
provided, we split the sample in two groups: a healthy
group containing respondents who characterized their general
health status as being either “very healthy” or “somewhat
healthy”, and an unhealthy group containing respondents who
claimed they were either ”unhealthy” or “somewhat unhealthy”. (page 18)*/
gen poor_health = 0
replace poor_health=1 if kk01==3 | kk01==4
*Replace poor_health as missing if kk01 is missing 
replace poor_health=. if kk01==.

************************ Test of the SCORE claim, H*  ************************
*Control variables
/* Explanatory variables include respondents' education (measured
in years of formal education), respondents' age and age squared
(to allow for possible non-linear effects), respondents' individual
risk and time preferences, the distance from the closest health
center (to proxy for the ease of access to medical care), household
per capita expenditures (PCE), and a sex dummy. (p. 18)*/

*Probit regression including DISTANCE as a covariate
local covariates_distance "log_PCE risk_preference time_preference distance female age agesqrt"
*Probit regression model for the probability of being under-diagnosed
probit under_diag yrs_school `covariates_distance' if poor_health==1
*Obtain Marginal effects
mfx
*Verify that only "3:Not Book Proxy" participats are used in the probit regressions
tab proxy if e(sample)==1

*Probit regression excluding DISTANCE as a covariate
local covariates_no_distance "log_PCE risk_preference time_preference female age agesqrt"
*Probit regression model for the probability of being under-diagnosed
probit under_diag yrs_school `covariates_no_distance' if poor_health==1
*Obtain Marginal effects
mfx
*Verify that only "3:Not Book Proxy" participats are used in the probit regressions
tab proxy if e(sample)==1

*Close log
log close
*Create PDF from log
translate `log_name'.smcl `log_name'.pdf, replace

display `End of Do-file'
