* Encoding: UTF-8.

COMPUTE Wrote30WordsOrMore = 99.
EXECUTE.

RECODE Wrote30WordsOrMore (99=SYSMIS).
EXECUTE.

IF (WordCount>29) Wrote30WordsOrMore = 1.
IF (WordCount<30) Wrote30WordsOrMore = 0.
EXECUTE.

FORMATS Wrote30WordsOrMore (F9.0).
EXECUTE.

COMPUTE EuropeanAmerican = 0.
EXECUTE.

IF (CulturalBackground = 1) EuropeanAmerican = 1.
EXECUTE.

COMPUTE EastAsian = 0.
EXECUTE.

IF (CulturalBackground = 4) EastAsian = 1.
EXECUTE.

COMPUTE Culture = 99.
EXECUTE.

RECODE Culture (99=SYSMSIS). 
EXECUTE.

IF (EuropeanAmerican = 1) Culture = 0.
IF (EastAsian = 1) Culture = 1.
EXECUTE.

COMPUTE USBorn = 0.
EXECUTE.

IF (BornCountry = 0) USBorn = 1.
EXECUTE.

IF (BornCountry = 187) USBorn = 1.
EXECUTE.

COMPUTE EastAsianBorn = 0.
EXECUTE.

IF (BornCountry = 162) EastAsianBorn = 1.
IF (BornCountry = 156) EastAsianBorn = 1.
IF (BornCountry = 140) EastAsianBorn = 1.
IF (BornCountry = 86) EastAsianBorn = 1.
IF (BornCountry = 75) EastAsianBorn = 1.
IF (BornCountry = 36) EastAsianBorn = 1.
IF (BornCountry = 1358) EastAsianBorn = 1.
EXECUTE.

*Calculating prostitution attitude average scores

RECODE PAS2 (-3 = 3) (-2=2) (-1=1) (0=0)(1=-1)(2=-2)(3=-3) into PAS2Positive.
EXECUTE.

RECODE PAS5 (-3 = 3) (-2=2) (-1=1) (0=0)(1=-1)(2=-2)(3=-3) into PAS5Positive.
EXECUTE.

COMPUTE PASAverage = MEAN(PAS1, PAS2Positive, PAS3, PAS4, PAS5Positive).
EXECUTE.

*Checking for PAS outliers of > 2 SD

DESCRIPTIVES VARIABLES=PASAverage
  /STATISTICS = MEAN STDDEV MIN MAX.

FREQUENCIES VARIABLES=PASAverage
 /ORDER=ANALYSIS.

*Calculating PANAS net scores

COMPUTE PANASNetScore = (PANAS1 + PANAS3 + PANAS5 + PANAS9 + PANAS10 + PANAS12 + PANAS14 + PANAS16 + PANAS17 + PANAS19)-
(PANAS2 + PANAS4 + PANAS6 + PANAS7 + PANAS8 + PANAS11 + PANAS13 + PANAS15 + PANAS18 + PANAS20).
EXECUTE.

*Main effect of mood

UNIANOVA PASAverage BY PANASNetScore
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=PANASNetScore.

*Mood interaction with writing condition

UNIANOVA PASAverage BY PANASNetScore WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=PANASNetScore WritingCondition PANASNetScore*WritingCondition.

*Main effect of mood on bail

UNIANOVA BailAmount BY PANASNetScore
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=PANASNetScore.

*Mood interaction with writing condition on bail

UNIANOVA BailAmount BY PANASNetScore WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=PANASNetScore WritingCondition PANASNetScore*WritingCondition.




*Main replication statistical test

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.


MEANS TABLES=PASAverage BailAmount BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.

*Cultural differences across all conditions
    
UNIANOVA PASAverage BY Culture
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture.



*Breaking down interaction: European American participants only

USE ALL. 
COMPUTE filter_$=(Culture = 0). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=WritingCondition.

*Breaking down interaction: East Asian participants only

USE ALL. 
COMPUTE filter_$=(EastAsian= 1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=WritingCondition.

USE ALL.
EXECUTE.



*Breaking down interaction: cultural differences in control writing condition only

USE ALL. 
COMPUTE filter_$=(WritingCondition= 0). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture.

USE ALL.
EXECUTE.


*Breaking down interaction: cultural differences in experimental writing condition only

USE ALL. 
COMPUTE filter_$=(WritingCondition= 1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture.

USE ALL.
EXECUTE.



*Analyzing bail

UNIANOVA BailAmount BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

*Means for bail
    
MEANS TABLES= BailAmount BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.


*Breaking down interaction: cultural differences in control writing condition only (bail)

USE ALL. 
COMPUTE filter_$=(WritingCondition= 0). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA BailAmount BY Culture
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture.

USE ALL.
EXECUTE.


*Breaking down interaction: cultural differences in experimental writing condition only (bail)

USE ALL. 
COMPUTE filter_$=(WritingCondition= 1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA BailAmount BY Culture
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture.

USE ALL.
EXECUTE.

*Limiting the East Asian sample to those born outside United States

COMPUTE EuropeanAmericanOrEastAsianBorn= 0.
EXECUTE.

IF (EuropeanAmerican = 1) EuropeanAmericanOrEastAsianBorn= 1.
EXECUTE.

IF (EastAsianBorn= 1) EuropeanAmericanOrEastAsianBorn= 1.
EXECUTE.

USE ALL. 
COMPUTE filter_$=(EuropeanAmericanOrEastAsianBorn= 1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.

*Re-running analysis excluding participants who wrote fewer than 30 words.

USE ALL. 
COMPUTE filter_$=(Wrote30WordsOrMore=1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.


*Means for bail
    
MEANS TABLES= PASAverage BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.

*Analyzing bail

UNIANOVA BailAmount BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.


*Means for bail
    
MEANS TABLES= BailAmount BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.

USE ALL.
EXECUTE.


*Re-running analysis excluding participants who wrote less than 1 word. 

USE ALL. 
COMPUTE filter_$=(WordCount>1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.


*Re-running analysis excluding participants who wrote 1 sentence or less. 

USE ALL. 
COMPUTE filter_$=(SentenceCount>1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.

*Testing for skewness and outliers

DESCRIPTIVES VARIABLES=PASAverage BailAmount
  /STATISTICS=MEAN STDDEV MIN MAX.


*Excluding PAS Average outliers
*Mean = 0.9751, SD = 1.377.
*No positive outliers discovered. But negative outliers discovered (< -1.7789)

USE ALL. 
COMPUTE filter_$=(PASAverage>-1.7789). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.

*Bail amount is not normally distrbuted
*Cutoffs for skewness = 2, kurtosis = 7 (Curran, West, & Finch, 1997)
    
DESCRIPTIVES VARIABLES=PASAverage BailAmount
  /STATISTICS=MEAN STDDEV MIN MAX KURTOSIS SKEWNESS.

*Bail mean = 2279.79; SD = 27165.03. Cutoff for +2SD = 56609.85
    
USE ALL. 
COMPUTE filter_$=(BailAmount< 56609.85). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

*I re-ran the interaction between culture and writing condition after excluding outliers. The interaction remained non-significant 

UNIANOVA BailAmount BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.


*Even after removing 3 outliers, skewness and kurtosis still far exceeds limits. 

COMPUTE BailAmountSqrt = SQRT(BailAmount).
EXECUTE.

COMPUTE BailAmountLog = LN(BailAmount+1).
EXECUTE.

DESCRIPTIVES VARIABLES= BailAMount BailAmountSqrt BailAmountLog
  /STATISTICS=MEAN STDDEV MIN MAX KURTOSIS SKEWNESS.

*Lighter square root transformation not sufficient to get skewness and kurtosis under limits, but log transformation is. 

UNIANOVA BailAmountLog BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.

*Testing results without Prolific sample

FREQUENCIES VARIABLES=Prolific
  /ORDER=ANALYSIS.


USE ALL. 
COMPUTE filter_$=(Prolific=0). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

USE ALL.
EXECUTE.


USE ALL. 
COMPUTE filter_$=(Prolific=1). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.


MEANS TABLES=PASAverage BailAmount BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.


USE ALL.
EXECUTE.



COMPUTE Religious = 99.
EXECUTE.

RECODE Religious (99=SYSMIS).
EXECUTE.

IF (Religion <3) Religious=0.
IF (Religion >2) Religious=1.
EXECUTE.

UNIANOVA PASAverage BY Religious
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN= Religious.

*Testing main replication effect among non-religious participants

USE ALL. 
COMPUTE filter_$=(Religious=0). 
VARIABLE LABELS filter_$ 'Exclude=0  (FILTER)'. 
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'. 
FORMATS filter_$ (f1.0). 
FILTER BY filter_$. 
EXECUTE.

UNIANOVA PASAverage BY Culture WritingCondition
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /PRINT=ETASQ
  /CRITERIA=ALPHA(0.05) 
  /DESIGN=Culture WritingCondition Culture*WritingCondition.

MEANS TABLES=PASAverage BY Culture BY WritingCondition
  /CELLS=MEAN COUNT SEMEAN.

USE ALL.
EXECUTE.




