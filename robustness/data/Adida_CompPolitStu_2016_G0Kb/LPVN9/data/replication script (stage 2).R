library(dplyr)
library(summarytools)
library(emmeans)
library(MASS)

d <- read.csv("Benin2012survey.csv")

#choosing respondents included in control and experimental groups
dexp <- d %>% filter(passage %in% c("Control", "Femme", "FonFemme"))
dexp$Group <- relevel(as.factor(dexp$passage), ref="Control")
freq(dexp$Group)

#ethnicity (based on the "ethnie" variable)
dexp$et <- NA
dexp$et[dexp$ethnie=="Fon"] <- "Wife Coethnics"
dexp$et[dexp$ethnie=="Yorouba"|dexp$ethnie=="Bariba"] <- "President Coethnics"
dexp$et[dexp$ethnie!="Fon" & dexp$ethnie!="Yorouba" & dexp$ethnie!="Bariba"] <- "Non Ethnics"
freq(dexp$et)

ctable(dexp$Group, dexp$et)

#DV: If there were no term limits and the election were held today, would you vote for Yayi Boni for President? (BoniVote; 0-1)
freq(dexp$BoniVote)

#checking group means
dexp %>%
  group_by(et, Group) %>%
  summarise_at(vars(BoniVote), list(name = mean))


#testing the effect of the manipulation
vote.glm <- glm(BoniVote ~ Group*et,family=binomial(link='logit'),data=dexp)
summary(vote.glm)

#testing the effect of the manipulation by ethnic group
vote.emm <- emmeans(vote.glm, ~ Group | et)
c <- contrast(vote.emm, "trt.vs.ctrl", ref = "Control")
summary(c)

#95% CI
confint(c, level = .95)
#95% CI for OR
confint(c, level = .95, type = "response")


