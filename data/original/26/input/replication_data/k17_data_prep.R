rm(list = ls())

t1<-read.csv("k17_t1_data.csv", stringsAsFactors = FALSE)
t2<-read.csv("k17_t2_data.csv", stringsAsFactors = FALSE)
t3<-read.csv("k17_t3_data.csv", stringsAsFactors = FALSE)

#identifying participants present in all waves

intersection <- Reduce(intersect, list(t1$RecipientEmail,t2$RecipientEmail,t3$RecipientEmail))

t1i<-t1[which(t1$RecipientEmail %in% intersection),]
t2i<-t2[which(t2$RecipientEmail %in% intersection),]
t3i<-t3[which(t3$RecipientEmail %in% intersection),]
rm(t1, t2, t3)

#selecting participants who completed the questionnaire
t1i<-t1i[which(t1i$Finished == 1),]
t2i<-t2i[which(t2i$Finished == 1),]
t3i<-t3i[which(t3i$Finished == 1),]

#adding the number of the data collection wave to data files
t1i$Wave <- round(as.numeric(difftime(t1i$StartDate, t1i$StartDate[1], unit = "days"))/7)+1
t2i$Wave <- round(as.numeric(difftime(t2i$StartDate, (t1i$StartDate[1]), unit = "days"))-2)/7+1
t3i$Wave <- round(as.numeric(difftime(t3i$StartDate, (t1i$StartDate[1]), unit = "days"))-7)/7+1

#updating column names
T1 = t1i[-c(1:6,8:11,13:17,84,85)]

names(T1)[1]="Finished_T1"
names(T1)[3]="gender"
names(T1)[4]="birthyear"
names(T1)[5]="education"
names(T1)[6]="children"
names(T1)[7]="work_hours"
names(T1)[8]="work_days"

names(T1)[9]="T1_panas_jov_1"
names(T1)[10]="T1_panas_sad_1"
names(T1)[11]="T1_panas_gen_pos_2"
names(T1)[12]="T1_panas_guilt_4"
names(T1)[13]="T1_panas_hos_1"
names(T1)[14]="T1_panas_ser_2"
names(T1)[15]="T1_panas_guilt_2"
names(T1)[16]="T1_panas_jov_7"
names(T1)[17]="T1_panas_att_1"
names(T1)[18]="T1_panas_fear_1"
names(T1)[19]="T1_panas_jov_4"
names(T1)[20]="T1_panas_sad_5"
names(T1)[21]="T1_panas_shy_1"
names(T1)[22]="T1_panas_fat_2"
names(T1)[23]="T1_panas_fear_3"

names(T1)[24]="T1_panas_shy_4"
names(T1)[25]="T1_panas_fat_1"
names(T1)[26]="T1_panas_sur_2"
names(T1)[27]="T1_panas_sad_4"
names(T1)[28]="T1_panas_gen_neg_2"
names(T1)[29]="T1_panas_self_ass_1"
names(T1)[30]="T1_panas_fear_2"
names(T1)[31]="T1_panas_fat_3"
names(T1)[32]="T1_panas_guilt_5"
names(T1)[33]="T1_panas_sur_1"
names(T1)[34]="T1_panas_jov_3"
names(T1)[35]="T1_panas_jov_5"
names(T1)[36]="T1_panas_att_3"
names(T1)[37]="T1_panas_self_ass_2"
names(T1)[38]="T1_panas_shy_2"

names(T1)[39]="T1_panas_hos_5"
names(T1)[40]="T1_panas_fear_6"
names(T1)[41]="T1_panas_hos_2"
names(T1)[42]="T1_panas_sad_2"
names(T1)[43]="T1_panas_self_ass_5"
names(T1)[44]="T1_panas_sur_3"
names(T1)[45]="T1_panas_ser_1"
names(T1)[46]="T1_panas_att_2"
names(T1)[47]="T1_panas_fear_4"
names(T1)[48]="T1_panas_gen_pos_3"
names(T1)[49]="T1_panas_hos_3"
names(T1)[50]="T1_panas_gen_neg_1"
names(T1)[51]="T1_panas_jov_6"
names(T1)[52]="T1_panas_hos_6"
names(T1)[53]="T1_panas_jov_2"

names(T1)[54]="T1_panas_hos_4"
names(T1)[55]="T1_panas_guilt_3"
names(T1)[56]="T1_panas_self_ass_6"
names(T1)[57]="T1_panas_gen_pos_1"
names(T1)[58]="T1_panas_self_ass_4"
names(T1)[59]="T1_panas_ser_3"
names(T1)[60]="T1_panas_jov_8"
names(T1)[61]="T1_panas_self_ass_3"
names(T1)[62]="T1_panas_sad_3"
names(T1)[63]="T1_panas_fear_5"
names(T1)[64]="T1_panas_att_4"
names(T1)[65]="T1_panas_guilt_1"
names(T1)[66]="T1_panas_shy_3"
names(T1)[67]="T1_panas_fat_4"
names(T1)[68]="T1_panas_guilt_6"

T2 <- t2i[,-c(18:77,201)]
T2 <- T2[-c(1:6,8:11,13:17,140)]

names(T2)[1]="Finished_T2"
names(T2)[3]="req_detach_1"
names(T2)[4]="req_detach_2"
names(T2)[5]="req_detach_3"
names(T2)[6]="req_detach_4"
names(T2)[7]="req_relax_1"
names(T2)[8]="req_relax_2"
names(T2)[9]="req_relax_3"
names(T2)[10]="req_relax_4"
names(T2)[11]="req_mastery_1"
names(T2)[12]="req_mastery_2"
names(T2)[13]="req_mastery_3"
names(T2)[14]="req_mastery_4"
names(T2)[15]="req_control_1"
names(T2)[16]="req_control_2"
names(T2)[17]="req_control_3"
names(T2)[18]="req_control_4"

names(T2)[19]="has_1"
names(T2)[20]="has_2"
names(T2)[21]="has_3"
names(T2)[22]="has_4"
names(T2)[23]="has_5"
names(T2)[24]="has_6"
names(T2)[25]="has_7"
names(T2)[26]="has_8"
names(T2)[27]="has_9"
names(T2)[28]="has_10"
names(T2)[29]="has_11"
names(T2)[30]="has_12"
names(T2)[31]="has_13"
names(T2)[32]="has_14"
names(T2)[33]="has_15"
names(T2)[34]="has_16"

names(T2)[35]="has_17"
names(T2)[36]="has_18"
names(T2)[37]="has_19"
names(T2)[38]="has_20"
names(T2)[39]="has_21"
names(T2)[40]="has_22"
names(T2)[41]="has_23"
names(T2)[42]="has_24"
names(T2)[43]="has_25"
names(T2)[44]="has_26"
names(T2)[45]="has_27"
names(T2)[46]="has_28"
names(T2)[47]="has_29"
names(T2)[48]="has_30"
names(T2)[49]="has_31"
names(T2)[50]="has_32"

names(T2)[51]="has_33"
names(T2)[52]="has_34"
names(T2)[53]="has_35"
names(T2)[54]="has_36"
names(T2)[55]="has_37"
names(T2)[56]="has_38"
names(T2)[57]="has_39"
names(T2)[58]="has_40"
names(T2)[59]="has_41"
names(T2)[60]="has_42"
names(T2)[61]="has_43"
names(T2)[62]="has_44"
names(T2)[63]="has_45"
names(T2)[64]="has_46"
names(T2)[65]="has_47"
names(T2)[66]="has_48"

names(T2)[67]="has_49"
names(T2)[68]="has_50"
names(T2)[69]="has_51"
names(T2)[70]="has_52"
names(T2)[71]="has_53"

names(T2)[72]="upl_1"
names(T2)[73]="upl_2"
names(T2)[74]="upl_3"
names(T2)[75]="upl_4"
names(T2)[76]="upl_5"
names(T2)[77]="upl_6"
names(T2)[78]="upl_7"
names(T2)[79]="upl_8"
names(T2)[80]="upl_9"
names(T2)[81]="upl_10"
names(T2)[82]="upl_11"
names(T2)[83]="upl_12"
names(T2)[84]="upl_13"
names(T2)[85]="upl_14"
names(T2)[86]="upl_15"
names(T2)[87]="upl_16"

names(T2)[88]="upl_17"
names(T2)[89]="upl_18"
names(T2)[90]="upl_19"
names(T2)[91]="upl_20"
names(T2)[92]="upl_21"
names(T2)[93]="upl_22"
names(T2)[94]="upl_23"
names(T2)[95]="upl_24"
names(T2)[96]="upl_25"
names(T2)[97]="upl_26"
names(T2)[98]="upl_27"
names(T2)[99]="upl_28"
names(T2)[100]="upl_29"
names(T2)[101]="upl_30"
names(T2)[102]="upl_31"
names(T2)[103]="upl_32"

names(T2)[104]="upl_33"
names(T2)[105]="upl_34"
names(T2)[106]="upl_35"
names(T2)[107]="upl_36"
names(T2)[108]="upl_37"
names(T2)[109]="upl_38"
names(T2)[110]="upl_39"
names(T2)[111]="upl_40"
names(T2)[112]="upl_41"
names(T2)[113]="upl_42"
names(T2)[114]="upl_43"
names(T2)[115]="upl_44"
names(T2)[116]="upl_45"
names(T2)[117]="upl_46"
names(T2)[118]="upl_47"
names(T2)[119]="upl_48"

names(T2)[120]="upl_49"
names(T2)[121]="upl_50"
names(T2)[122]="upl_51"
names(T2)[123]="upl_52"
names(T2)[124]="upl_53"

T3 = t3i[-c(1:6,8:11,13:17,78:79)]

names(T3)[1]="Finished_T3"
names(T3)[3]="T3_panas_jov_1"
names(T3)[4]="T3_panas_sad_1"
names(T3)[5]="T3_panas_gen_pos_2"
names(T3)[6]="T3_panas_guilt_4"
names(T3)[7]="T3_panas_hos_1"
names(T3)[8]="T3_panas_ser_2"
names(T3)[9]="T3_panas_guilt_2"
names(T3)[10]="T3_panas_jov_7"
names(T3)[11]="T3_panas_att_1"
names(T3)[12]="T3_panas_fear_1"
names(T3)[13]="T3_panas_jov_4"
names(T3)[14]="T3_panas_sad_5"
names(T3)[15]="T3_panas_shy_1"
names(T3)[16]="T3_panas_fat_2"
names(T3)[17]="T3_panas_fear_3"

names(T3)[18]="T3_panas_shy_4"
names(T3)[19]="T3_panas_fat_1"
names(T3)[20]="T3_panas_sur_2"
names(T3)[21]="T3_panas_sad_4"
names(T3)[22]="T3_panas_gen_neg_2"
names(T3)[23]="T3_panas_self_ass_1"
names(T3)[24]="T3_panas_fear_2"
names(T3)[25]="T3_panas_fat_3"
names(T3)[26]="T3_panas_guilt_5"
names(T3)[27]="T3_panas_sur_1"
names(T3)[28]="T3_panas_jov_3"
names(T3)[29]="T3_panas_jov_5"
names(T3)[30]="T3_panas_att_3"
names(T3)[31]="T3_panas_self_ass_2"
names(T3)[32]="T3_panas_shy_2"

names(T3)[33]="T3_panas_hos_5"
names(T3)[34]="T3_panas_fear_6"
names(T3)[35]="T3_panas_hos_2"
names(T3)[36]="T3_panas_sad_2"
names(T3)[37]="T3_panas_self_ass_5"
names(T3)[38]="T3_panas_sur_3"
names(T3)[39]="T3_panas_ser_1"
names(T3)[40]="T3_panas_att_2"
names(T3)[41]="T3_panas_fear_4"
names(T3)[42]="T3_panas_gen_pos_3"
names(T3)[43]="T3_panas_hos_3"
names(T3)[44]="T3_panas_gen_neg_1"
names(T3)[45]="T3_panas_jov_6"
names(T3)[46]="T3_panas_hos_6"
names(T3)[47]="T3_panas_jov_2"

names(T3)[48]="T3_panas_hos_4"
names(T3)[49]="T3_panas_guilt_3"
names(T3)[50]="T3_panas_self_ass_6"
names(T3)[51]="T3_panas_gen_pos_1"
names(T3)[52]="T3_panas_self_ass_4"
names(T3)[53]="T3_panas_ser_3"
names(T3)[54]="T3_panas_jov_8"
names(T3)[55]="T3_panas_self_ass_3"
names(T3)[56]="T3_panas_sad_3"
names(T3)[57]="T3_panas_fear_5"
names(T3)[58]="T3_panas_att_4"
names(T3)[59]="T3_panas_guilt_1"
names(T3)[60]="T3_panas_shy_3"
names(T3)[61]="T3_panas_fat_4"
names(T3)[62]="T3_panas_guilt_6"


#Combining into one dataset

score_all <- merge(T1,T2, by = c("RecipientEmail", "Wave"))
score_all <- merge(score_all,T3, by = c("RecipientEmail", "Wave"))

#Removing participants who completed the study more than once, keeping first instance
score_all <- score_all[!duplicated(score_all$RecipientEmail),]


#dealing with factors

score_all$gender <- as.numeric(as.character(score_all$gender))
score_all$birthyear <- as.numeric(as.character(score_all$birthyear))

score_all$children = as.numeric(as.character(score_all$children))
score_all$children[score_all$children >= 1] = 1
score_all$children[score_all$children < 1] = 0
score_all$children = as.factor(score_all$children)

score_all[,3:252]<-sapply(score_all[,3:252],as.character)
score_all[,3:252]<-sapply(score_all[,3:252],as.numeric)

#Indices

#Recode age

score_all$age <- 2021-score_all$birthyear

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

write.csv(score_all, file = "k17_processed_data.csv", row.names = FALSE)
