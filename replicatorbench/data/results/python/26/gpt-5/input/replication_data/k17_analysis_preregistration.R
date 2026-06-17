#Hierarchical regressions
#Inference criterion: Criteria for a successful replication for the focal test would be a statistically 
#significant positive effect (alpha = .05, one-tailed) of relaxation at Time 2 on joviality at Time 3 in 
#the regression model.

Model_1 <- lm(T3_panas_joviality ~ birthyear + gender + children + work_hours + work_days, data = score_all)

Model_2 <- lm(T3_panas_joviality ~ birthyear + gender + children + work_hours + work_days + T1_panas_joviality, data = score_all)

Model_3 <- lm(T3_panas_joviality ~ birthyear + gender + children + work_hours + work_days + T1_panas_joviality + req_control + req_mastery + req_relax + req_detach + hassles, data = score_all)

summary(Model_1)$r.squared

summary(Model_2)$r.squared

summary(Model_2)$r.squared - summary(Model_1)$r.squared

anova(Model_1, Model_2)

summary(Model_3)$r.squared

summary(Model_3)$r.squared - summary(Model_2)$r.squared

anova(Model_2, Model_3)

#Exploratory: PANAS T3 ~ REQ

Model_4 <- lm(T3_panas_self_assurance ~ birthyear + gender + children + work_hours + work_days + T1_panas_self_assurance + req_control + req_mastery + req_relax + req_detach + hassles, data = score_all)

summary(Model_4)

Model_5 <- lm(T3_panas_serenity ~ birthyear + gender + children + work_hours + work_days + T1_panas_serenity + req_control + req_mastery + req_relax + req_detach + hassles, data = score_all)

summary(Model_5)

Model_6 <- lm(T3_panas_fear ~ birthyear + gender + children + work_hours + work_days + T1_panas_fear + req_control + req_mastery + req_relax + req_detach + hassles, data = score_all)

summary(Model_6)

Model_7 <- lm(T3_panas_sadness ~ birthyear + gender + children + work_hours + work_days + T1_panas_sadness + req_control + req_mastery + req_relax + req_detach + hassles, data = score_all)

summary(Model_7)