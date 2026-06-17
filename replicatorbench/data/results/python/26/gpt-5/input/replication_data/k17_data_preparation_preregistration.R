#Removing unneeded variables

T1_1.0 = T1[-c(1:6,8:12,14:17,84,85)]

T2_1.0 = T2[-c(1:6,8:12,14:17,140,141)]

T3_1.0 = T3[-c(1:6,8:12,14:17,78,79)]

#Renaming

names(T1_1.0)[1]="Finished_T1"
names(T1_1.0)[3]="gender"
names(T1_1.0)[4]="birthyear"
names(T1_1.0)[5]="education"
names(T1_1.0)[6]="children"
names(T1_1.0)[7]="work_hours"
names(T1_1.0)[8]="work_days"

names(T1_1.0)[9]="T1_panas_jov_1"
names(T1_1.0)[10]="T1_panas_hos_1"
names(T1_1.0)[11]="T1_panas_att_1"
names(T1_1.0)[12]="T1_panas_shy_1"
names(T1_1.0)[13]="T1_panas_fat_1"
names(T1_1.0)[14]="T1_panas_self_ass_1"
names(T1_1.0)[15]="T1_panas_sur_1"
names(T1_1.0)[16]="T1_panas_self_ass_2"
names(T1_1.0)[17]="T1_panas_hos_2"
names(T1_1.0)[18]="T1_panas_ser_1"
names(T1_1.0)[19]="T1_panas_hos_3"
names(T1_1.0)[20]="T1_panas_jov_2"
names(T1_1.0)[21]="T1_panas_gen_pos_1"
names(T1_1.0)[22]="T1_panas_self_ass_3"
names(T1_1.0)[23]="T1_panas_guilt_1"

names(T1_1.0)[24]="T1_panas_sad_1"
names(T1_1.0)[25]="T1_panas_ser_2"
names(T1_1.0)[26]="T1_panas_fear_1"
names(T1_1.0)[27]="T1_panas_fat_2"
names(T1_1.0)[28]="T1_panas_sur_2"
names(T1_1.0)[29]="T1_panas_fear_2"
names(T1_1.0)[30]="T1_panas_jov_3"
names(T1_1.0)[31]="T1_panas_shy_2"
names(T1_1.0)[32]="T1_panas_sad_2"
names(T1_1.0)[33]="T1_panas_att_2"
names(T1_1.0)[34]="T1_panas_gen_neg_1"
names(T1_1.0)[35]="T1_panas_hos_4"
names(T1_1.0)[36]="T1_panas_self_ass_4"
names(T1_1.0)[37]="T1_panas_sad_3"
names(T1_1.0)[38]="T1_panas_shy_3"

names(T1_1.0)[39]="T1_panas_gen_pos_2"
names(T1_1.0)[40]="T1_panas_guilt_2"
names(T1_1.0)[41]="T1_panas_jov_4"
names(T1_1.0)[42]="T1_panas_fear_3"
names(T1_1.0)[43]="T1_panas_sad_4"
names(T1_1.0)[44]="T1_panas_fat_3"
names(T1_1.0)[45]="T1_panas_jov_5"
names(T1_1.0)[46]="T1_panas_hos_5"
names(T1_1.0)[47]="T1_panas_self_ass_5"
names(T1_1.0)[48]="T1_panas_fear_4"
names(T1_1.0)[49]="T1_panas_jov_6"
names(T1_1.0)[50]="T1_panas_guilt_3"
names(T1_1.0)[51]="T1_panas_ser_3"
names(T1_1.0)[52]="T1_panas_fear_5"
names(T1_1.0)[53]="T1_panas_fat_4"

names(T1_1.0)[54]="T1_panas_guilt_4"
names(T1_1.0)[55]="T1_panas_jov_7"
names(T1_1.0)[56]="T1_panas_sad_5"
names(T1_1.0)[57]="T1_panas_shy_4"
names(T1_1.0)[58]="T1_panas_gen_neg_2"
names(T1_1.0)[59]="T1_panas_guilt_5"
names(T1_1.0)[60]="T1_panas_att_3"
names(T1_1.0)[61]="T1_panas_fear_6"
names(T1_1.0)[62]="T1_panas_sur_3"
names(T1_1.0)[63]="T1_panas_gen_pos_3"
names(T1_1.0)[64]="T1_panas_hos_6"
names(T1_1.0)[65]="T1_panas_self_ass_6"
names(T1_1.0)[66]="T1_panas_jov_8"
names(T1_1.0)[67]="T1_panas_att_4"
names(T1_1.0)[68]="T1_panas_guilt_6"


names(T2_1.0)[1]="Finished_T2"
names(T2_1.0)[3]="req_detach_1"
names(T2_1.0)[4]="req_detach_2"
names(T2_1.0)[5]="req_detach_3"
names(T2_1.0)[6]="req_detach_4"
names(T2_1.0)[7]="req_relax_1"
names(T2_1.0)[8]="req_relax_2"
names(T2_1.0)[9]="req_relax_3"
names(T2_1.0)[10]="req_relax_4"
names(T2_1.0)[11]="req_mastery_1"
names(T2_1.0)[12]="req_mastery_2"
names(T2_1.0)[13]="req_mastery_3"
names(T2_1.0)[14]="req_mastery_4"
names(T2_1.0)[15]="req_control_1"
names(T2_1.0)[16]="req_control_2"
names(T2_1.0)[17]="req_control_3"
names(T2_1.0)[18]="req_control_4"

names(T2_1.0)[19]="has_1"
names(T2_1.0)[20]="has_2"
names(T2_1.0)[21]="has_3"
names(T2_1.0)[22]="has_4"
names(T2_1.0)[23]="has_5"
names(T2_1.0)[24]="has_6"
names(T2_1.0)[25]="has_7"
names(T2_1.0)[26]="has_8"
names(T2_1.0)[27]="has_9"
names(T2_1.0)[28]="has_10"
names(T2_1.0)[29]="has_11"
names(T2_1.0)[30]="has_12"
names(T2_1.0)[31]="has_13"
names(T2_1.0)[32]="has_14"
names(T2_1.0)[33]="has_15"
names(T2_1.0)[34]="has_16"

names(T2_1.0)[35]="has_17"
names(T2_1.0)[36]="has_18"
names(T2_1.0)[37]="has_19"
names(T2_1.0)[38]="has_20"
names(T2_1.0)[39]="has_21"
names(T2_1.0)[40]="has_22"
names(T2_1.0)[41]="has_23"
names(T2_1.0)[42]="has_24"
names(T2_1.0)[43]="has_25"
names(T2_1.0)[44]="has_26"
names(T2_1.0)[45]="has_27"
names(T2_1.0)[46]="has_28"
names(T2_1.0)[47]="has_29"
names(T2_1.0)[48]="has_30"
names(T2_1.0)[49]="has_31"
names(T2_1.0)[50]="has_32"

names(T2_1.0)[51]="has_33"
names(T2_1.0)[52]="has_34"
names(T2_1.0)[53]="has_35"
names(T2_1.0)[54]="has_36"
names(T2_1.0)[55]="has_37"
names(T2_1.0)[56]="has_38"
names(T2_1.0)[57]="has_39"
names(T2_1.0)[58]="has_40"
names(T2_1.0)[59]="has_41"
names(T2_1.0)[60]="has_42"
names(T2_1.0)[61]="has_43"
names(T2_1.0)[62]="has_44"
names(T2_1.0)[63]="has_45"
names(T2_1.0)[64]="has_46"
names(T2_1.0)[65]="has_47"
names(T2_1.0)[66]="has_48"

names(T2_1.0)[67]="has_49"
names(T2_1.0)[68]="has_50"
names(T2_1.0)[69]="has_51"
names(T2_1.0)[70]="has_52"
names(T2_1.0)[71]="has_53"

names(T2_1.0)[72]="upl_1"
names(T2_1.0)[73]="upl_2"
names(T2_1.0)[74]="upl_3"
names(T2_1.0)[75]="upl_4"
names(T2_1.0)[76]="upl_5"
names(T2_1.0)[77]="upl_6"
names(T2_1.0)[78]="upl_7"
names(T2_1.0)[79]="upl_8"
names(T2_1.0)[80]="upl_9"
names(T2_1.0)[81]="upl_10"
names(T2_1.0)[82]="upl_11"
names(T2_1.0)[83]="upl_12"
names(T2_1.0)[84]="upl_13"
names(T2_1.0)[85]="upl_14"
names(T2_1.0)[86]="upl_15"
names(T2_1.0)[87]="upl_16"

names(T2_1.0)[88]="upl_17"
names(T2_1.0)[89]="upl_18"
names(T2_1.0)[90]="upl_19"
names(T2_1.0)[91]="upl_20"
names(T2_1.0)[92]="upl_21"
names(T2_1.0)[93]="upl_22"
names(T2_1.0)[94]="upl_23"
names(T2_1.0)[95]="upl_24"
names(T2_1.0)[96]="upl_25"
names(T2_1.0)[97]="upl_26"
names(T2_1.0)[98]="upl_27"
names(T2_1.0)[99]="upl_28"
names(T2_1.0)[100]="upl_29"
names(T2_1.0)[101]="upl_30"
names(T2_1.0)[102]="upl_31"
names(T2_1.0)[103]="upl_32"

names(T2_1.0)[104]="upl_33"
names(T2_1.0)[105]="upl_34"
names(T2_1.0)[106]="upl_35"
names(T2_1.0)[107]="upl_36"
names(T2_1.0)[108]="upl_37"
names(T2_1.0)[109]="upl_38"
names(T2_1.0)[110]="upl_39"
names(T2_1.0)[111]="upl_40"
names(T2_1.0)[112]="upl_41"
names(T2_1.0)[113]="upl_42"
names(T2_1.0)[114]="upl_43"
names(T2_1.0)[115]="upl_44"
names(T2_1.0)[116]="upl_45"
names(T2_1.0)[117]="upl_46"
names(T2_1.0)[118]="upl_47"
names(T2_1.0)[119]="upl_48"

names(T2_1.0)[120]="upl_49"
names(T2_1.0)[121]="upl_50"
names(T2_1.0)[122]="upl_51"
names(T2_1.0)[123]="upl_52"
names(T2_1.0)[124]="upl_53"

names(T3_1.0)[1]="Finished_T3"
names(T3_1.0)[3]="T3_panas_jov_1"
names(T3_1.0)[4]="T3_panas_hos_1"
names(T3_1.0)[5]="T3_panas_att_1"
names(T3_1.0)[6]="T3_panas_shy_1"
names(T3_1.0)[7]="T3_panas_fat_1"
names(T3_1.0)[8]="T3_panas_self_ass_1"
names(T3_1.0)[9]="T3_panas_sur_1"
names(T3_1.0)[10]="T3_panas_self_ass_2"
names(T3_1.0)[11]="T3_panas_hos_2"
names(T3_1.0)[12]="T3_panas_ser_1"
names(T3_1.0)[13]="T3_panas_hos_3"
names(T3_1.0)[14]="T3_panas_jov_2"
names(T3_1.0)[15]="T3_panas_gen_pos_1"
names(T3_1.0)[16]="T3_panas_self_ass_3"
names(T3_1.0)[17]="T3_panas_guilt_1"

names(T3_1.0)[18]="T3_panas_sad_1"
names(T3_1.0)[19]="T3_panas_ser_2"
names(T3_1.0)[20]="T3_panas_fear_1"
names(T3_1.0)[21]="T3_panas_fat_2"
names(T3_1.0)[22]="T3_panas_sur_2"
names(T3_1.0)[23]="T3_panas_fear_2"
names(T3_1.0)[24]="T3_panas_jov_3"
names(T3_1.0)[25]="T3_panas_shy_2"
names(T3_1.0)[26]="T3_panas_sad_2"
names(T3_1.0)[27]="T3_panas_att_2"
names(T3_1.0)[28]="T3_panas_gen_neg_1"
names(T3_1.0)[29]="T3_panas_hos_4"
names(T3_1.0)[30]="T3_panas_self_ass_4"
names(T3_1.0)[31]="T3_panas_sad_3"
names(T3_1.0)[32]="T3_panas_shy_3"

names(T3_1.0)[33]="T3_panas_gen_pos_2"
names(T3_1.0)[34]="T3_panas_guilt_2"
names(T3_1.0)[35]="T3_panas_jov_4"
names(T3_1.0)[36]="T3_panas_fear_3"
names(T3_1.0)[37]="T3_panas_sad_4"
names(T3_1.0)[38]="T3_panas_fat_3"
names(T3_1.0)[39]="T3_panas_jov_5"
names(T3_1.0)[40]="T3_panas_hos_5"
names(T3_1.0)[41]="T3_panas_self_ass_5"
names(T3_1.0)[42]="T3_panas_fear_4"
names(T3_1.0)[43]="T3_panas_jov_6"
names(T3_1.0)[44]="T3_panas_guilt_3"
names(T3_1.0)[45]="T3_panas_ser_3"
names(T3_1.0)[46]="T3_panas_fear_5"
names(T3_1.0)[47]="T3_panas_fat_4"

names(T3_1.0)[48]="T3_panas_guilt_4"
names(T3_1.0)[49]="T3_panas_jov_7"
names(T3_1.0)[50]="T3_panas_sad_5"
names(T3_1.0)[51]="T3_panas_shy_4"
names(T3_1.0)[52]="T3_panas_gen_neg_2"
names(T3_1.0)[53]="T3_panas_guilt_5"
names(T3_1.0)[54]="T3_panas_att_3"
names(T3_1.0)[55]="T3_panas_fear_6"
names(T3_1.0)[56]="T3_panas_sur_3"
names(T3_1.0)[57]="T3_panas_gen_pos_3"
names(T3_1.0)[58]="T3_panas_hos_6"
names(T3_1.0)[59]="T3_panas_self_ass_6"
names(T3_1.0)[60]="T3_panas_jov_8"
names(T3_1.0)[61]="T3_panas_att_4"
names(T3_1.0)[62]="T3_panas_guilt_6"


#Combining into one dataset

score_all <- merge(T1_1.0,T2_1.0, by="External Data Reference")
score_all <- merge(score_all,T3_1.0, by="External Data Reference")

#Removing participants who did not complete all questionnaires

score_all <- score_all[score_all$Finished_T1 == "True" & score_all$Finished_T2 == "True" & score_all$Finished_T3 == "True", ] 

#Dummy coding

score_all$gender = as.factor(score_all$gender)

score_all$gender = relevel(score_all$gender, "mannlich")

score_all$children[score_all$children >= 1] = 1

score_all$children[score_all$children < 1] = 0

score_all$children = as.factor(score_all$children)


#Indices

#Recovery Experiences Questionnaire

score_all$req_control = rowMeans(data.frame(score_all$req_control_1, score_all$req_control_2, score_all$req_control_3, score_all$req_control_4) , na.rm = T)

score_all$req_detach = rowMeans(data.frame(score_all$req_detach_1, score_all$req_detach_2, score_all$req_detach_3, score_all$req_detach_4) , na.rm = T)

score_all$req_relax = rowMeans(data.frame(score_all$req_relax_1, score_all$req_relax_2, score_all$req_relax_3, score_all$req_relax_4) , na.rm = T)

score_all$req_mastery = rowMeans(data.frame(score_all$req_mastery_1, score_all$req_mastery_2, score_all$req_mastery_3, score_all$req_mastery_4) , na.rm = T)

#Hassles

score_all$hassles = rowSums(data.frame(score_all$has_1, score_all$has_2, score_all$has_3, score_all$has_4, score_all$has_5,
                                       score_all$has_6, score_all$has_7, score_all$has_8, score_all$has_9, score_all$has_10,
                                       score_all$has_11, score_all$has_12, score_all$has_13, score_all$has_14,score_all$has_15,
                                       score_all$has_16,score_all$has_17,score_all$has_18,score_all$has_19,score_all$has_20,
                                       score_all$has_21,score_all$has_22,score_all$has_23,score_all$has_24,score_all$has_25,
                                       score_all$has_26,score_all$has_27,score_all$has_28,score_all$has_29,score_all$has_30,
                                       score_all$has_31,score_all$has_32,score_all$has_33,score_all$has_34,score_all$has_35,
                                       score_all$has_36,score_all$has_37,score_all$has_38,score_all$has_39,score_all$has_40,
                                       score_all$has_41,score_all$has_42,score_all$has_43,score_all$has_44,score_all$has_45,
                                       score_all$has_46,score_all$has_47,score_all$has_48,score_all$has_49,score_all$has_50,
                                       score_all$has_51,score_all$has_52,score_all$has_53) , na.rm = T)

#PANAS T1

score_all$T1_panas_negative = rowMeans(data.frame(score_all$T1_panas_fear_1, score_all$T1_panas_fear_5, score_all$T1_panas_fear_3, score_all$T1_panas_fear_4, score_all$T1_panas_guilt_2,
                                                  score_all$T1_panas_guilt_3, score_all$T1_panas_hos_3,score_all$T1_panas_hos_5,score_all$T1_panas_gen_neg_1,score_all$T1_panas_gen_neg_2) , na.rm = T)

score_all$T1_panas_fear = rowMeans(data.frame(score_all$T1_panas_fear_1, score_all$T1_panas_fear_2, score_all$T1_panas_fear_3, score_all$T1_panas_fear_4, score_all$T1_panas_fear_5,
                                              score_all$T1_panas_fear_6) , na.rm = T)

score_all$T1_panas_sadness = rowMeans(data.frame(score_all$T1_panas_sad_1, score_all$T1_panas_sad_2, score_all$T1_panas_sad_3, score_all$T1_panas_sad_4, score_all$T1_panas_sad_5) , na.rm = T)

score_all$T1_panas_guilt = rowMeans(data.frame(score_all$T1_panas_guilt_1, score_all$T1_panas_guilt_2, score_all$T1_panas_guilt_3, score_all$T1_panas_guilt_4, score_all$T1_panas_guilt_5, score_all$T1_panas_guilt_6) , na.rm = T)

score_all$T1_panas_hostility = rowMeans(data.frame(score_all$T1_panas_hos_1,score_all$T1_panas_hos_2,score_all$T1_panas_hos_3,score_all$T1_panas_hos_4,score_all$T1_panas_hos_5,score_all$T1_panas_hos_6), na.rm = T)

score_all$T1_panas_shyness = rowMeans(data.frame(score_all$T1_panas_shy_1,score_all$T1_panas_shy_2,score_all$T1_panas_shy_3,score_all$T1_panas_shy_4), na.rm = T)

score_all$T1_panas_fatigue = rowMeans(data.frame(score_all$T1_panas_fat_1,score_all$T1_panas_fat_2,score_all$T1_panas_fat_3,score_all$T1_panas_fat_4), na.rm = T)

score_all$T1_panas_positive = rowMeans(data.frame(score_all$T1_panas_gen_pos_1,score_all$T1_panas_gen_pos_2,score_all$T1_panas_gen_pos_3,score_all$T1_panas_att_2, score_all$T1_panas_att_1,
                                                  score_all$T1_panas_jov_7, score_all$T1_panas_jov_5, score_all$T1_panas_self_ass_5,score_all$T1_panas_self_ass_2, score_all$T1_panas_att_3), na.rm = T)

score_all$T1_panas_joviality = rowMeans(data.frame(score_all$T1_panas_jov_1,score_all$T1_panas_jov_2,score_all$T1_panas_jov_3,score_all$T1_panas_jov_4,score_all$T1_panas_jov_5,
                                                   score_all$T1_panas_jov_6,score_all$T1_panas_jov_7,score_all$T1_panas_jov_8), na.rm = T)

score_all$T1_panas_self_assurance = rowMeans(data.frame(score_all$T1_panas_self_ass_1, score_all$T1_panas_self_ass_2,score_all$T1_panas_self_ass_3,score_all$T1_panas_self_ass_4,score_all$T1_panas_self_ass_5,score_all$T1_panas_self_ass_6), na.rm = T)

score_all$T1_panas_attentiveness = rowMeans(data.frame(score_all$T1_panas_att_1, score_all$T1_panas_att_2,score_all$T1_panas_att_3,score_all$T1_panas_att_4), na.rm = T)

score_all$T1_panas_serenity = rowMeans(data.frame(score_all$T1_panas_ser_1, score_all$T1_panas_ser_2,score_all$T1_panas_ser_3), na.rm = T)

score_all$T1_panas_surprise = rowMeans(data.frame(score_all$T1_panas_sur_1, score_all$T1_panas_sur_2,score_all$T1_panas_sur_3), na.rm = T)

#PANAS T3

score_all$T3_panas_negative = rowMeans(data.frame(score_all$T3_panas_fear_1, score_all$T3_panas_fear_5, score_all$T3_panas_fear_3, score_all$T3_panas_fear_4, score_all$T3_panas_guilt_2,
                                                  score_all$T3_panas_guilt_3, score_all$T3_panas_hos_3,score_all$T3_panas_hos_5,score_all$T3_panas_gen_neg_1,score_all$T3_panas_gen_neg_2) , na.rm = T)

score_all$T3_panas_fear = rowMeans(data.frame(score_all$T3_panas_fear_1, score_all$T3_panas_fear_2, score_all$T3_panas_fear_3, score_all$T3_panas_fear_4, score_all$T3_panas_fear_5,
                                              score_all$T3_panas_fear_6) , na.rm = T)

score_all$T3_panas_sadness = rowMeans(data.frame(score_all$T3_panas_sad_1, score_all$T3_panas_sad_2, score_all$T3_panas_sad_3, score_all$T3_panas_sad_4, score_all$T3_panas_sad_5) , na.rm = T)

score_all$T3_panas_guilt = rowMeans(data.frame(score_all$T3_panas_guilt_1, score_all$T3_panas_guilt_2, score_all$T3_panas_guilt_3, score_all$T3_panas_guilt_4, score_all$T3_panas_guilt_5, score_all$T3_panas_guilt_6) , na.rm = T)

score_all$T3_panas_hostility = rowMeans(data.frame(score_all$T3_panas_hos_1,score_all$T3_panas_hos_2,score_all$T3_panas_hos_3,score_all$T3_panas_hos_4,score_all$T3_panas_hos_5,score_all$T3_panas_hos_6), na.rm = T)

score_all$T3_panas_shyness = rowMeans(data.frame(score_all$T3_panas_shy_1,score_all$T3_panas_shy_2,score_all$T3_panas_shy_3,score_all$T3_panas_shy_4), na.rm = T)

score_all$T3_panas_fatigue = rowMeans(data.frame(score_all$T3_panas_fat_1,score_all$T3_panas_fat_2,score_all$T3_panas_fat_3,score_all$T3_panas_fat_4), na.rm = T)

score_all$T3_panas_positive = rowMeans(data.frame(score_all$T3_panas_gen_pos_1,score_all$T3_panas_gen_pos_2,score_all$T3_panas_gen_pos_3,score_all$T3_panas_att_2, score_all$T3_panas_att_1,
                                                  score_all$T3_panas_jov_7, score_all$T3_panas_jov_5, score_all$T3_panas_self_ass_5,score_all$T3_panas_self_ass_2, score_all$T3_panas_att_3), na.rm = T)

score_all$T3_panas_joviality = rowMeans(data.frame(score_all$T3_panas_jov_1,score_all$T3_panas_jov_2,score_all$T3_panas_jov_3,score_all$T3_panas_jov_4,score_all$T3_panas_jov_5,
                                                   score_all$T3_panas_jov_6,score_all$T3_panas_jov_7,score_all$T3_panas_jov_8), na.rm = T)

score_all$T3_panas_self_assurance = rowMeans(data.frame(score_all$T3_panas_self_ass_1, score_all$T3_panas_self_ass_2,score_all$T3_panas_self_ass_3,score_all$T3_panas_self_ass_4,score_all$T3_panas_self_ass_5,score_all$T3_panas_self_ass_6), na.rm = T)

score_all$T3_panas_attentiveness = rowMeans(data.frame(score_all$T3_panas_att_1, score_all$T3_panas_att_2,score_all$T3_panas_att_3,score_all$T3_panas_att_4), na.rm = T)

score_all$T3_panas_serenity = rowMeans(data.frame(score_all$T3_panas_ser_1, score_all$T3_panas_ser_2,score_all$T3_panas_ser_3), na.rm = T)

score_all$T3_panas_surprise = rowMeans(data.frame(score_all$T3_panas_sur_1, score_all$T3_panas_sur_2,score_all$T3_panas_sur_3), na.rm = T)

#Variable standardization

library(dplyr)

score_all%>%mutate_if(is.numeric,scale)
