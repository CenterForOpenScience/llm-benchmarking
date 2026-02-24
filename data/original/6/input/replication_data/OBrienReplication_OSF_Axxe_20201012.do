/* This do-file recreates O'Brien and Noy's 2015 ASR findings using more recent versions of the GSS (2012, 2014, 2016, 2018).
The data for this replication were created using the R Markdown file "OBrien.code" created by  Marco Ramljak. I edited that original file so that the data were exported in a Stata format. I also removed a variable they listed (natsci) and replaced it with (advfront). Based on table 2 of the original paper, I believe that switch is correct. The original Markdown file can be found here: https://osf.io/rzyx9/

The following information was found on this spreadsheet: https://docs.google.com/spreadsheets/d/1s0nSjqmDz_8r6pPdnRUl29ReCSZFFXwUf7AGTlupgVM/edit#gid=549689836

The claim I will be testing:
Although the post-secular perspective entails high levels of science knowledge as well as favorable views of science and religion, when scientific and religious perspectives conflict (e.g. evolution), the post-secular latent class almost unanimously aligned their views with particular religious accounts.

More specifically:
Members of the post-secular category were significantly less likely than members of the traditional group to respond that humans evolved from other animals (3 percent, significant at p < 0.05 on a two-tailed test, see Table 2, rightmost column).

How this is done:
Participants’ responses to the General Social Survey (GSS) were submitted to a latent class analysis that resulted in a three-class solution characterized as representing traditional, modern, and post-secular perspectives on science and religion. Following this assignment, two-tailed t-tests were used to compare responses between the three groups; for the purposes of the SCORE project, the focal test is the comparison between the Traditional and Post-Secular groups on the question concerning evolution (‘Human beings developed from earlier species of animals’, yes or no).

I run Latent Class Analysis (LCA) using the following variables: hotcore, radioact, boyorgrl, lasers, electron, viruses, earthsun, condrift, bigbang, evolved, odds1, odds2, expdesgn, scistudy, nextgen, toofast, advfront, scibnfts, bible, and reliten.

Erick Axxe -- October 12th 2020 -- Center for Open Science replication */

********************************************************************************
*********************************** Data cleaning ******************************
********************************************************************************
global clean "C:\Users\axxe.1\Documents\CenterOpenScience\OBrien_AmSocioRev_2015_AxxeReplication\GSSreplication_clean.dta"

** First, I call in the dataset. This data was created by the R file by Marco Ramljak. See above for notes on the edits I made to their R Markdown file.
use "C:\Users\axxe.1\Documents\CenterOpenScience\OBrien_AmSocioRev_2015_AxxeReplication\GSSreplication.dta", clear

** Change all the variable names to lowercase.
rename *, lower

** Variables will need to be edited to account for missing values. Binary questions will be recoded so that no answer (9) and not applicable (0) are missing. Don't know (8) is recoded to the "wrong" response.
global binary "hotcore radioact boyorgrl lasers electron viruses earthsun condrift bigbang evolved expdesgn odds1 odds2"

foreach i in $binary{
recode `i' 1=. 5=., gen(`i'_clean)
}

** Variables are coded on whether or not the respondent listed the correct answer. I code those as 1. Wrong responses are coded as 0.
global true "hotcore boyorgrl electron earthsun condrift bigbang evolved odds2"
global false "radioact lasers viruses expdesgn odds1"

foreach i in $true{
recode `i'_clean 2=1 3=0 4=0	
}

foreach i in $false{
recode `i'_clean 3=1 2=0 4=0
}

// I run a cross tab to confirm it was created correctly.
foreach i in $binary {
tab `i' `i'_clean, m
}

** Scientific study has 3 possible responses. Don't know (8), No answer (9), and not applicable (0) are coded as missing. 
recode scistudy 1=. 5=. 6=., gen(scistudy_clean)

// I run a cross tab to confirm it was created correctly.
tab scistudy scistudy_clean, m


** Scale response variables need to be recoded to account for missing. Don't know (8), no answer (9), and not applicable (0) are recoded as missing.  TOOFAST must also be reverse coded.
global scales "nextgen toofast advfront scibnfts"
foreach i in nextgen toofast advfront {
recode `i' 1=. 2=1 3=2 4=3 5=4 6=. 7=., gen(`i'_clean)
}

foreach i in scibnfts {
recode `i' 1=. 2=4 3=2 4=0  5=. 6=., gen(`i'_clean)
}

//Reverse code 'toofast'
recode toofast_clean 4=1 3=2 2=3 1=4

// I run a cross tab to confirm it was created correctly.
foreach i in $scales {
tab `i' `i'_clean
}

** "Bible" is recoded into three categories: whether people believe the bible is the actual word of God, is inspire by the word of God, or is a book of myths and fables. Missing responses to the question remain missing in the new variable.
recode bible 1=. 2=1 3=2 4=3 5=. 6=. 7=., gen(bible_clean)

** Finally, strength of religious affiliation needs to be recoded such that don't know (8), no answer (9), and not applicable (0) are coded as missing. 
recode reliten 1=. 2=4 3=3 4=2 5=1 6=. 7=., gen(reliten_clean)

// I run a cross tab to confirm it was created correctly.
tab reliten reliten_clean

save $clean, replace

********************************************************************************
************************** Test for sample sizes *******************************
********************************************************************************
** Here, I check whether the sample sizes from the original study correspond with mine.
use $clean, clear

keep if year == 2006 | year == 2008 | year == 2010

global vars_clean "hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean scistudy_clean nextgen_clean toofast_clean advfront_clean scibnfts_clean bible_clean reliten_clean"

egen nmis=rmiss($vars_clean)
tab nmis, m

keep if nmis == 0
tab year

*** The sample sizes from the original study: 1,563 from 2006; 988 from 2008; and 350 from 2010

*** The sample sizes from my code scheme: 1,608 from 2006, 1,016 from 2008, and 367 from 2010

log using OBrienReplication_Axxe_20201012.txt, text replace

********************************************************************************
********************************** Analysis ************************************
********************************************************************************

** I will be preparing three analyses: one that uses observations not in the original study, one that combines all observations, and that uses observations that were used in the original study. 

*************** 1. Observations not in the original study: *********************
use $clean, clear

// Only keep years not used in the original analysis.
keep if year > 2010

//Listwise deletion
egen nmis=rmiss($vars_clean)
tab nmis, m
keep if nmis == 0

// Latent Class Models -- I run three models with a varying number of latent classes: first has 2, second has 4, third has 3
quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 2) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc2

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit), lclass(C 4) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc4

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 3) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc3

estimates stats lc2 lc3 lc4

// Report the predicted value for each item in each class
predict classpost*, classposteriorpr

// Using those results, create a variable identifying the three classes. To do so, I place respondents into categories based off which category had the highest predictive probability.
egen max = rowmax(classpost*)
gen predclass = 1 if classpost1 == max
replace predclass = 2 if classpost2 == max
replace predclass = 3 if classpost3 ==max

//Ocular examination of the means for each class and variable
foreach i in $vars_clean {
mean `i', over(predclass)
}

tab bible_clean if predclass == 1
tab bible_clean if predclass == 2
tab bible_clean if predclass == 3

//This section depends on the results from the LC analysis. I review the conditional means of the results and label them accordingly.
label define classes 1"Traditional" 2"Modern" 3"Post-secular"
label values predclass classes

//I create a new variable in order to run the t-test with 1 as Post-secularists and 0 as traditionalists.
gen PostsecVsTrad = 1 if predclass == 3
replace PostsecVsTrad = 0 if predclass == 1
label define posttrad 0"Traditional" 1"Post-secular"
label values PostsecVsTrad posttrad

// Result of interest for GSS years not in the original study:
ttest evolved_clean, by(PostsecVsTrad)


************************ 2. All observations: **********************************
use $clean, clear

//Listwise deletion
egen nmis=rmiss($vars_clean)
tab nmis, m
keep if nmis == 0

// Latent Class Models -- I run three models with a varying number of latent classes: first has 2, second has 4, third has 3
quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 2) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc2

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit), lclass(C 4) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc4

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 3) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc3

estimates stats lc2 lc3 lc4

// Report the predicted value for each item in each class
predict classpost*, classposteriorpr

// Using those results, create a variable identifying the three classes. To do so, I place respondents into categories based off which category had the highest predictive probability.
egen max = rowmax(classpost*)
gen predclass = 1 if classpost1 == max
replace predclass = 2 if classpost2 == max
replace predclass = 3 if classpost3 ==max

//Ocular examination of the means for each class and variable
foreach i in $vars_clean {
mean `i', over(predclass)
}

tab bible_clean if predclass == 1
tab bible_clean if predclass == 2
tab bible_clean if predclass == 3

//This section depends on the results from the LC analysis. I review the conditional means of the results and label them accordingly.
label define classes 1"Modern" 2"Traditionalist" 3"Post-secular"
label values predclass classes

//I create a new variable in order to run the t-test with 1 as Post-secularists and 0 as traditionalists.
gen PostsecVsTrad = 1 if predclass == 3
replace PostsecVsTrad = 0 if predclass == 2
label values PostsecVsTrad posttrad

// Result of interest for all available GSS years:
ttest evolved_clean, by(PostsecVsTrad)


***************** 3. Observations in the original study: ***********************
use $clean, clear

keep if year <= 2010

//Listwise deletion
egen nmis=rmiss($vars_clean)
tab nmis, m
keep if nmis == 0

// Latent Class Models -- I run three models with a varying number of latent classes: first has 2, second has 4, third has 3
quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 2) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc2

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit), lclass(C 4) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc4

quietly gsem (hotcore_clean radioact_clean boyorgrl_clean lasers_clean electron_clean viruses_clean earthsun_clean condrift_clean bigbang_clean evolved_clean expdesgn_clean odds1_clean odds2_clean <- ,  logit) /// 
( bible_clean  <- , mlogit) ///
( nextgen_clean toofast_clean advfront_clean scibnfts_clean reliten_clean scistudy_clean <- , ologit) [pweight = wtss], lclass(C 3) nonrtolerance startvalues(randomid, draws(15) seed(12345)) nodvheader nocapslatent em(iter(5))

est store lc3

estimates stats lc2 lc3 lc4


// Report the predicted value for each item in each class
predict classpost*, classposteriorpr

// Using those results, create a variable identifying the three classes. To do so, I place respondents into categories based off which category had the highest predictive probability.
egen max = rowmax(classpost*)
gen predclass = 1 if classpost1 == max
replace predclass = 2 if classpost2 == max
replace predclass = 3 if classpost3 ==max

//Ocular examination of the means for each class and variable
foreach i in $vars_clean {
mean `i', over(predclass)
}

tab bible_clean if predclass == 1
tab bible_clean if predclass == 2
tab bible_clean if predclass == 3

//This section depends on the results from the LC analysis. I review the conditional means of the results and label them accordingly.
label define classes 1"Post-secular" 2"Traditional" 3"Modern"
label values predclass classes

//I create a new variable in order to run the t-test with 1 as Post-secularists and 0 as traditionalists.
gen PostsecVsTrad = 1 if predclass == 1
replace PostsecVsTrad = 0 if predclass == 2
label values PostsecVsTrad posttrad

// Result of interest for years in original study:
ttest evolved_clean, by(PostsecVsTrad)


log close


