* Encoding: UTF-8.
*****          Need for cognition          *****


RECODE Cog5, Cog6, Cog8, Cog9, Cog10, Cog11, Cog13, Cog14, Cog15, Cog16, Cog17, Cog18, Cog20, Cog21, Cog22, Cog24, Cog25, Cog26, Cog31, Cog32, Cog33
    (-4=4) (-3=3) (-2=2) (-1=1) (0=0) (1=-1) (2=-2) (3=-3) (4=-4) INTO
   Cog5_rec, Cog6_rec, Cog8_rec, Cog9_rec, Cog10_rec, Cog11_rec, Cog13_rec, Cog14_rec, Cog15_rec, Cog16_rec, Cog17_rec, Cog18_rec, Cog20_rec, Cog21_rec, Cog22_rec, Cog24_rec, Cog25_rec, Cog26_rec, Cog31_rec, Cog32_rec, Cog33_rec.
EXECUTE.


RELIABILITY
  /VARIABLES=Cog1 Cog2 Cog3 Cog4 Cog7 Cog12 Cog19 Cog23 Cog27 Cog28 Cog29 Cog30 Cog34 Cog5_rec 
    Cog6_rec Cog8_rec Cog9_rec Cog10_rec Cog11_rec Cog13_rec Cog14_rec Cog15_rec Cog16_rec Cog17_rec 
    Cog18_rec Cog20_rec Cog21_rec Cog22_rec Cog24_rec Cog25_rec Cog26_rec Cog31_rec Cog32_rec Cog33_rec
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE NFCog=mean (Cog1,Cog2,Cog3,Cog4,Cog7,Cog12,Cog19,Cog23,Cog27,Cog28,Cog29,Cog30,Cog34,
    Cog5_rec,Cog6_rec,Cog8_rec,Cog9_rec,Cog10_rec,Cog11_rec,Cog13_rec,Cog14_rec,Cog15_rec,Cog16_rec,
    Cog17_rec,Cog18_rec,Cog20_rec,Cog21_rec,Cog22_rec,Cog24_rec,Cog25_rec,Cog26_rec,Cog31_rec,Cog32_rec,
    Cog33_rec).
EXECUTE.


*****          Interpersonal Reactivity Index          *****

RECODE IRI3, IRI4, IRI7, IRI12, IRI13, IRI14, IRI15, IRI18, IRI19
    (5=1) (4=2) (3=3) (2=4) (1=5) INTO
    IRI3_rec, IRI4_rec, IRI7_rec, IRI12_rec, IRI13_rec, IRI14_rec, IRI15_rec, IRI18_rec, IRI19_rec.
EXECUTE. 

*** All scale ***

RELIABILITY
  /VARIABLES=IRI1 IRI2 IRI5 IRI6 IRI8 IRI9 IRI10 IRI11 IRI16 IRI17 IRI20 IRI21 IRI22 IRI23 IRI24 
    IRI25 IRI26 IRI27 IRI28 IRI3_rec IRI4_rec IRI7_rec  IRI12_rec IRI13_rec IRI14_rec IRI15_rec 
    IRI18_rec IRI19_rec
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE IRI_ALL=mean (IRI1,IRI2,IRI5,IRI6,IRI8,IRI9,IRI10,IRI11,IRI16,IRI17,IRI20,IRI21,IRI22,IRI23,IRI24,
    IRI25,IRI26,IRI27,IRI28,IRI3_rec,IRI4_rec,IRI7_rec,IRI12_rec,IRI13_rec,IRI14_rec,IRI15_rec,
    IRI18_rec,IRI19_rec).
EXECUTE.

***Empatic Concern***

RELIABILITY
  /VARIABLES=IRI2 IRI4_rec IRI9 IRI14_rec IRI18_rec IRI20 IRI22
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE IRI_EC=mean (IRI2,IRI4_rec,IRI9,IRI14_rec,IRI18_rec,IRI20,IRI22).
EXECUTE.

***Fantasy***
    
RELIABILITY
  /VARIABLES=IRI1 IRI5 IRI7_rec IRI12_rec IRI16 IRI23 IRI26
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE IRI_FS=mean (IRI1,IRI5,IRI7_rec,IRI12_rec,IRI16,IRI23,IRI26).
EXECUTE.

***Perspective Taking***
    
RELIABILITY
  /VARIABLES=IRI3_rec IRI8 IRI11 IRI15_rec IRI21 IRI25 IRI28
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE IRI_PT=mean (IRI3_rec,IRI8,IRI11,IRI15_rec,IRI21,IRI25,IRI28).
EXECUTE.

***Personal Distress***

RELIABILITY
  /VARIABLES=IRI6 IRI10 IRI13_rec IRI17 IRI19_rec IRI24 IRI27
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE IRI_PD=mean (IRI6,IRI10,IRI13_rec,IRI17,IRI19_rec,IRI24,IRI27).
EXECUTE.


*****          Interaction and Audience Anxiousness          *****
    
RECODE Fear2, Fear3, Fear6, Fear10, Fear15, Fear17, Fear23
    (5=1) (4=2) (3=3) (2=4) (1=5) INTO
    Fear2_rec, Fear3_rec, Fear6_rec, Fear10_rec, Fear15_rec, Fear17_rec, Fear23_rec.
EXECUTE. 

***All scale***

RELIABILITY
  /VARIABLES=Fear1 Fear4 Fear5 Fear7 Fear8 Fear9 Fear11 Fear12 Fear13 Fear14 Fear16 Fear18 Fear19 
    Fear20 Fear21 Fear22 Fear24 Fear25 Fear26 Fear27 Fear2_rec Fear3_rec Fear6_rec Fear10_rec 
    Fear15_rec Fear17_rec Fear23_rec
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE Fear_ALL=mean (Fear1,Fear4,Fear5,Fear7,Fear8,Fear9,Fear11,Fear12,Fear13,Fear14,Fear16,Fear18,Fear19,
   Fear20,Fear21,Fear22,Fear24,Fear25,Fear26,Fear27,Fear2_rec,Fear3_rec,Fear6_rec,Fear10_rec,
    Fear15_rec,Fear17_rec,Fear23_rec).
EXECUTE.

***Interaction Anxiousness***
    
RELIABILITY
  /VARIABLES=Fear1 Fear4 Fear5 Fear7 Fear8 Fear9 Fear11 Fear12 Fear13 Fear14 Fear2_rec Fear3_rec 
    Fear6_rec Fear10_rec Fear15_rec
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE Fear_IA=mean (Fear1,Fear4,Fear5,Fear7,Fear8,Fear9,Fear11,Fear12,Fear13,Fear14,Fear2_rec,Fear3_rec,
    Fear6_rec,Fear10_rec,Fear15_rec).
EXECUTE.

***Audience Anxiousness***
    
RELIABILITY
  /VARIABLES=Fear16 Fear18 Fear19 Fear20 Fear21 Fear22 Fear24 Fear25 Fear26 Fear27 Fear17_rec 
    Fear23_rec
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE Fear_AA=mean (Fear16,Fear18,Fear19,Fear20,Fear21,Fear22,Fear24,Fear25,Fear26,Fear27,Fear17_rec, 
    Fear23_rec).
EXECUTE.


*****     Quest for Siginificance     *****

RELIABILITY
  /VARIABLES=QfS1 QfS2 QfS3 QfS4 QfS5 QfS6
  /SCALE('ALL VARIABLES') ALL
  /MODEL=ALPHA
  /STATISTICS=DESCRIPTIVE SCALE
  /SUMMARY=TOTAL.

COMPUTE QfS_ALL=mean (QfS1,QfS2,QfS3,QfS4,QfS5,QfS6).
EXECUTE.


*****          Moral Dilemmas Training          *****

RECODE APP1 APP2 APP3 APP4 APP5 APP6 APP7 APP8 APP9 APP10 APP11 APP12 APP13 APP14 APP15 APP16 APP17 
    APP18 APP19 APP20 (1=0) (2=1).
EXECUTE.

***Sum of "Inappropriate" answer in Incongruent dilemmas***

COMPUTE APP_incong=APP1+APP2+APP3+APP4+APP5+APP6+APP7+APP8+APP9+APP10.
EXECUTE.

***Sum of "Inappropriate" answer in Congruent dilemmas***

COMPUTE APP_cong=APP11+APP12+APP13+APP14+APP15+APP16+APP17+APP18+APP19+APP20.
EXECUTE.

***Probability of rejecting harm in Incongruent Dilemmas***

COMPUTE RH_incong=APP_incong/10.
EXECUTE.

***Probability of rejecting harm in Congruent Dilemmas***

COMPUTE RH_cong=APP_cong/10.
EXECUTE.

***Utilitarian factor (Conway&Gawronski, 2013)***

COMPUTE U=RH_cong - RH_incong.
EXECUTE.

***Deontological factor (Conway&Gawronski, 2013)***

COMPUTE D=RH_incong/(1-U).
EXECUTE.


*****          Moral Dilemmas Main Task          *****
    
RECODE APP1_main APP2_main APP3_main APP4_main APP5_main APP6_main APP7_main APP8_main APP9_main 
    APP10_main APP11_main APP12_main APP13_main APP14_main APP15_main APP16_main APP17_main APP18_main 
    APP19_main APP20_main APP21_main APP22_main APP23_main APP24_main APP25_main APP26_main APP27_main 
    APP28_main APP29_main APP30_main APP31_main APP32_main APP33_main APP34_main APP35_main APP36_main 
    APP37_main APP38_main APP39_main APP40_main APP41_main APP42_main APP43_main APP44_main APP45_main 
    APP46_main APP47_main APP48_main APP49_main APP50_main (1=0) (2=1).
EXECUTE.

***Sum of "Inappropriate" answer in Personal Moral Dilemmas***

COMPUTE APP_PMD=APP1_main+APP2_main+APP3_main+APP4_main+APP5_main+APP6_main+APP7_main+APP8_main+
    APP9_main+APP10_main+APP11_main+APP12_main+APP13_main+APP14_main+APP15_main+APP16_main+APP17_main+
    APP18_main+APP19_main+APP20_main.
EXECUTE.

***Sum of "Inappropriate" answer in Non-Personal Moral Dilemmas***

COMPUTE APP_NMD=APP21_main+APP22_main+APP23_main+APP24_main+APP25_main+APP26_main+APP27_main+
    APP28_main+APP29_main+APP30_main+APP31_main+APP32_main+APP33_main+APP34_main+APP35_main.
EXECUTE.

***Sum of "Inappropriate" answer in Control Moral Dilemmas***

COMPUTE APP_CMD=APP36_main+APP37_main+APP38_main+APP39_main+APP40_main+APP41_main+APP42_main+
    APP43_main+APP44_main+APP45_main+APP46_main+APP47_main+APP48_main+APP49_main+APP50_main.
EXECUTE.

***Sum of "Inappropriate" answer in all Moral Dilemmas***

COMPUTE APP_ALL=APP_PMD+APP_NMD+APP_CMD.
EXECUTE.

***Probability of rejecting harm in three types of Moral Dilemmas***

COMPUTE RH_PMD=APP_PMD/20.
EXECUTE.

COMPUTE RH_NMD=APP_NMD/15.
EXECUTE.

COMPUTE RH_CMD=APP_CMD/15.
EXECUTE.

***Probability of rejecting harm in all Moral Dilemmas***
    
COMPUTE RH_ALL=APP_ALL/50.
EXECUTE.

***Confidence Scores***

RECODE CONF1_main CONF2_main CONF3_main CONF4_main CONF5_main CONF6_main CONF7_main CONF8_main 
    CONF9_main CONF10_main CONF11_main CONF12_main CONF13_main CONF14_main CONF15_main CONF16_main 
    CONF17_main CONF18_main CONF19_main CONF20_main CONF21_main CONF22_main CONF23_main CONF24_main 
    CONF25_main CONF26_main CONF27_main CONF28_main CONF29_main CONF30_main CONF31_main CONF32_main 
    CONF33_main CONF34_main CONF35_main CONF36_main CONF37_main CONF38_main CONF39_main CONF40_main 
    CONF41_main CONF42_main CONF43_main CONF44_main CONF45_main CONF46_main CONF47_main CONF48_main 
    CONF49_main CONF50_main (1=2) (4=2) (2=1) (3=1) INTO CONF1_REC_main CONF2_REC_main CONF3_REC_main CONF4_REC_main 
    CONF5_REC_main CONF6_REC_main CONF7_REC_main CONF8_REC_main CONF9_REC_main CONF10_REC_main 
    CONF11_REC_main CONF12_REC_main CONF13_REC_main CONF14_REC_main CONF15_REC_main CONF16_REC_main 
    CONF17_REC_main CONF18_REC_main CONF19_REC_main CONF20_REC_main CONF21_REC_main CONF22_REC_main 
    CONF23_REC_main CONF24_REC_main CONF25_REC_main CONF26_REC_main CONF27_REC_main CONF28_REC_main 
    CONF29_REC_main CONF30_REC_main CONF31_REC_main CONF32_REC_main CONF33_REC_main CONF34_REC_main 
    CONF35_REC_main CONF36_REC_main CONF37_REC_main CONF38_REC_main CONF39_REC_main CONF40_REC_main 
    CONF41_REC_main CONF42_REC_main CONF43_REC_main CONF44_REC_main CONF45_REC_main CONF46_REC_main 
    CONF47_REC_main CONF48_REC_main CONF49_REC_main CONF50_REC_main.
EXECUTE.

COMPUTE CONF_PMD=mean (CONF1_REC_main,CONF2_REC_main,CONF3_REC_main,CONF4_REC_main,
    CONF5_REC_main,CONF6_REC_main,CONF7_REC_main,CONF8_REC_main,CONF9_REC_main,CONF10_REC_main,
    CONF11_REC_main,CONF12_REC_main,CONF13_REC_main,CONF14_REC_main,CONF15_REC_main,CONF16_REC_main,
    CONF17_REC_main,CONF18_REC_main,CONF19_REC_main,CONF20_REC_main).
EXECUTE.

COMPUTE CONF_NMD=mean (CONF21_REC_main,CONF22_REC_main,
    CONF23_REC_main,CONF24_REC_main,CONF25_REC_main,CONF26_REC_main,CONF27_REC_main,CONF28_REC_main,
    CONF29_REC_main,CONF30_REC_main,CONF31_REC_main,CONF32_REC_main,CONF33_REC_main,CONF34_REC_main,
    CONF35_REC_main).
EXECUTE.

COMPUTE CONF_CMD=mean (CONF36_REC_main,CONF37_REC_main,CONF38_REC_main,CONF39_REC_main,CONF40_REC_main,
    CONF41_REC_main,CONF42_REC_main,CONF43_REC_main,CONF44_REC_main,CONF45_REC_main,CONF46_REC_main,
    CONF47_REC_main,CONF48_REC_main,CONF49_REC_main,CONF50_REC_main).
EXECUTE.

COMPUTE CONF_ALL=mean(CONF1_REC_main,CONF2_REC_main,CONF3_REC_main,CONF4_REC_main,
    CONF5_REC_main,CONF6_REC_main,CONF7_REC_main,CONF8_REC_main,CONF9_REC_main,CONF10_REC_main,
    CONF11_REC_main,CONF12_REC_main,CONF13_REC_main,CONF14_REC_main,CONF15_REC_main,CONF16_REC_main,
    CONF17_REC_main,CONF18_REC_main,CONF19_REC_main,CONF20_REC_main,CONF21_REC_main,CONF22_REC_main,
    CONF23_REC_main,CONF24_REC_main,CONF25_REC_main,CONF26_REC_main,CONF27_REC_main,CONF28_REC_main,
    CONF29_REC_main,CONF30_REC_main,CONF31_REC_main,CONF32_REC_main,CONF33_REC_main,CONF34_REC_main,
    CONF35_REC_main,CONF36_REC_main,CONF37_REC_main,CONF38_REC_main,CONF39_REC_main,CONF40_REC_main,
    CONF41_REC_main,CONF42_REC_main,CONF43_REC_main,CONF44_REC_main,CONF45_REC_main,CONF46_REC_main,
    CONF47_REC_main,CONF48_REC_main,CONF49_REC_main,CONF50_REC_main).
EXECUTE.

*Confidence ratings divided by response type were calculated in Excel, using formula Mean.if (Variables: CONF_PMD_Yes, CONF_PMD_No, CONF_NMD_Yes, CONF_NMD_No, CONF_CMD_Yes, CONF_CMD_No).

***Reaction time in three types of Moral Dilemmas***

COMPUTE RT_PMD=mean (RT1_corr,RT2_corr,RT3_corr,RT4_corr,RT5_corr,RT6_corr,RT7_corr,RT8_corr,
    RT9_corr,RT10_corr,RT11_corr,RT12_corr,RT13_corr,RT14_corr,RT15_corr,RT16_corr,RT17_corr,RT18_corr,
    RT19_corr,RT20_corr).
EXECUTE.

COMPUTE RT_NMD=mean (RT21_corr, RT22_corr, RT23_corr,RT24_corr,RT25_corr,RT26_corr,RT27_corr,RT28_corr,
    RT29_corr, RT30_corr,RT31_corr,RT32_corr,RT33_corr,RT34_corr,RT35_corr).
EXECUTE.

COMPUTE RT_CMD=mean (RT36_corr, RT37_corr, RT38_corr,RT39_corr,RT40_corr,RT41_corr,RT42_corr,RT43_corr,
    RT44_corr, RT45_corr,RT46_corr,RT47_corr,RT48_corr,RT49_corr,RT50_corr).
EXECUTE.

*Reaction times divided by response type were calculated in Excel, using formula Mean.if (Variables: RT_PMD_Yes, RT_PMD_No, RT_NMD_Yes, RT_NMD_No, RT_CMD_Yes, RT_CMD_No).


***Reaction time in all Moral Dilemmas***
    
*All reaction times above or below 3SD were excluded (variables RTs are raw times, variables RT_corr are corrected).
   
COMPUTE RT_ALL=mean (RT1_corr,RT2_corr,RT3_corr,RT4_corr,RT5_corr,RT6_corr,RT7_corr,RT8_corr,
    RT9_corr,RT10_corr,RT11_corr,RT12_corr,RT13_corr,RT14_corr,RT15_corr,RT16_corr,RT17_corr,RT18_corr,
    RT19_corr,RT20_corr,RT21_corr, RT22_corr, RT23_corr,RT24_corr,RT25_corr,RT26_corr,RT27_corr,RT28_corr,
    RT29_corr, RT30_corr,RT31_corr,RT32_corr,RT33_corr,RT34_corr,RT35_corr,RT36_corr, RT37_corr, RT38_corr,RT39_corr,RT40_corr,RT41_corr,RT42_corr,RT43_corr,
    RT44_corr, RT45_corr,RT46_corr,RT47_corr,RT48_corr,RT49_corr,RT50_corr).
EXECUTE.


*****          Hypotheses testing          *****

MORAL DILEMMAS - main task

***     2x3 mixed ANOVA (condition x dilemma type)     ***

GLM RH_PMD RH_NMD RH_CMD BY Condition
  /WSFACTOR=dilemma_type 3 Polynomial 
  /METHOD=SSTYPE(3)
  /POSTHOC=Condition(BONFERRONI)
  /PLOT=PROFILE(dilemma_type*Condition) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=dilemma_type 
  /DESIGN=Condition.


***     2x3x2 mixed ANOVA for confidence ratings  (condition x dilemma type x response type)     ***

GLM CONF_PMD_Yes CONF_PMD_No CONF_NMD_Yes CONF_NMD_No CONF_CMD_Yes CONF_CMD_No BY Condition
  /WSFACTOR=dilemma_type 3 Polynomial Answer 2 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(dilemma_type*Condition*Answer) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Answer) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Answer) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type*Answer) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=dilemma_type Answer dilemma_type*Answer
  /DESIGN=Condition.

***     2x3x2 mixed ANOVA for reaction times  (condition x dilemma type x response type)     ***

GLM RT_PMD_Yes RT_PMD_No RT_NMD_Yes RT_NMD_No RT_CMD_Yes RT_CMD_No BY Condition
  /WSFACTOR=dilemma_type 3 Polynomial Answer 2 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(dilemma_type*Condition*Answer) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Answer) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Answer) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type*Answer) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(dilemma_type*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(dilemma_type) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*dilemma_type*Answer) COMPARE(Answer) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=dilemma_type Answer dilemma_type*Answer
  /DESIGN=Condition.

*****          LEXICAL DECISION TASK          *****
 
 All indices for LDT were calculated by the Inquisit software and incorporated into SPSS

***     T-test for overall correctness between conditions     ***   

T-TEST GROUPS=Condition(1 2)
  /MISSING=ANALYSIS
  /VARIABLES=LDT_Corr_ALL
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

***      2(condition) x 4 (word category)  mixed ANOVA for correctness in LDT     ***

GLM LDT_Corr_REP LDT_Corr_Comp LDT_Corr_Warm LDT_Corr_Neut BY Condition
  /WSFACTOR=Category 4 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(Category*Condition) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Category) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Category)
  /EMMEANS=TABLES(Condition*Category) COMPARE (Condition) ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Category) COMPARE (Category) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Category 
  /DESIGN=Condition.

***     T-test for overall reaction time between conditions     ***   

T-TEST GROUPS=Condition(1 2)
  /MISSING=ANALYSIS
  /VARIABLES=LDT_RT_ALL
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

***      2(condition) x 4 (word category)  mixed ANOVA for reaction time in LDT     ***
   
GLM LDT_RT_REP LDT_RT_Comp LDT_RT_Warm LDT_RT_Neut BY Condition
  /WSFACTOR=Category 4 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(Category*Condition) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Category) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Category) 
   /EMMEANS=TABLES(Condition*Category) COMPARE (Condition) ADJ(BONFERRONI)
   /EMMEANS=TABLES(Condition*Category) COMPARE (Category) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Category 
  /DESIGN=Condition.

***      2(condition) x 2 (mode of reasoning)  mixed ANOVA on moral reasoning     ***

GLM U D BY Condition
  /WSFACTOR=Reasoning 2 Polynomial 
  /METHOD=SSTYPE(3)
  /PLOT=PROFILE(Reasoning*Condition) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Reasoning) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Reasoning) 
  /EMMEANS=TABLES(Condition*Reasoning) COMPARE (Condition) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Reasoning 
  /DESIGN=Condition.

***      2(condition) x 2 (measurement)  mixed ANOVA for Mood     ***

GLM MOOD1 MOOD2 BY Condition
  /WSFACTOR=Measurement 2 Polynomial 
  /METHOD=SSTYPE(3)
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Measurement) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Measurement) 
  /EMMEANS=TABLES(Condition*Measurement) COMPARE (Condition) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Measurement 
  /DESIGN=Condition.

***      2(condition) x 2 (measurement)  mixed ANOVA for Arousal     ***

GLM AROUSAL1 AROUSAL2 BY Condition
  /WSFACTOR=Measurement 2 Polynomial 
  /METHOD=SSTYPE(3)
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Condition) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Measurement) COMPARE ADJ(BONFERRONI)
  /EMMEANS=TABLES(Condition*Measurement) 
  /EMMEANS=TABLES(Condition*Measurement) COMPARE (Condition) ADJ(BONFERRONI)
  /PRINT=DESCRIPTIVE ETASQ 
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Measurement 
  /DESIGN=Condition.

***     Comparison of personality indices between control and experimental group     ***

T-TEST GROUPS=Condition(1 2)
  /MISSING=ANALYSIS
  /VARIABLES=IRI_ALL NFCog Fear_ALL
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).
