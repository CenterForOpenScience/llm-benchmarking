#'-------------------------------------------------------------------------
#' Multi100 Project
#' Replication of "The Spousal Bump: Do Cross-Ethnic Marriages Increase 
#' Political Support in Multiethnic Democracies?"
#'
#' Claim: "priming the first lady’s ethnicity increases support for 
#' President Yayi among her coethnics"
#'
#' @author Youri Mora (Univ. libre de Bruxelles)
#' @version	2022/05/16
#'-------------------------------------------------------------------------



# Packages loading --------------------------------------------------------

# Used packages
packages <- c("here", # to use relative directories
              "readr", # to import dataset
              "dplyr", # to manipulate and clean data
              "emmeans", # for predicted marginal means of the model
              "ggplot2" # for pretty plots
              )

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))



# Data importation, cleaning and recoding ---------------------------------
dt <- as.data.frame(read_csv(here("Benin2012survey.csv")))

# codebook is missing → let's figure out which variables are appropriate
#
# /!\ the Independent Variable includes 5 conditions  in the dataset vs. 3 modalities reported in the paper.
# This discrepancy is of importance
# Since there is no justification to remove the two conditions, they should be included in the data analysis
# I will conduct both analyses, with and without (as in the paper) these conditions
# I contacted the corresponding author regarding this matter and received no answer.
# This is a case of QRP in my opinion

dt <- filter(dt, !is.na(BoniVote)) # remove participants with no data on the DV


dt$bonivote # dependent variable
dt$BoniVote # dependent variable
table(dt$bonivote, dt$BoniVote, useNA=c("always"))
# → these two variables are strictly equivalent

table(dt$passage)
# The distribution across conditions strongly suggests that it was intended to 
# compare all 5 modalities 


dt$FonGroup # this variable indicates whether participant is Fon (1) or not (0)
dt$FonGroup <- factor(dt$FonGroup)
contrasts(dt$FonGroup) <- -contr.sum(2)/2 # create contrasts

dt$NonCoethnics # this variables indicates whether participant are non coethnic of Yayi or his Wife (1) or not (0)


dt$FonCoEthnics
table(dt$FonGroup, dt$FonCoEthnics, useNA=c("always"))


table(dt$passage)
dt$passage <- factor(dt$passage, levels = c("Control","Femme","Nago","Bariba","FonFemme"))
contrasts(dt$passage) <- contr.helmert(5)


# Let's create dt1 to replicate the conditions mentioned in the paper
dt1 <- filter(dt, passage=="Control" | passage=="Femme" | passage=="FonFemme") # select three conditions as described in the paper

dt1$passage <- factor(dt1$passage)
contrasts(dt1$passage) <- contr.helmert(3)





# Inferential analysis ----------------------------------------------------


# > Following paper's method regarding inclusion of conditions ------------


# The DV is binary, let's use logistic regression
# The paper describes a 3 (passage: Control, Femme, FemmeFon) X 
# 2 (FonGroup - coethnicity with first lady)
# /!\ In my opinion, the "passage" variable should include all experimental conditions.See second analysis.
# The claim is an interaction : priming Fon ethnicity increases support for Yayi
# only among her co-ethnics
# The predictions calls for the use of Helmert contrasts for the experimental 
# condition
#      Control Femme FonFemme
# C1:    -1     1      0
# C2:    -1     -1      2
#
# We should observe an interaction between the second contrast and FonGroup


m1 <- glm(BoniVote ~ passage*FonGroup, data=dt1, family = binomial(link ='logit'))
summary(m1)

# The hypothesis is confirmed, priming co-ethnicity of first lady
# increases among her co-ethnics
# z=-2.253, p = .0242

m1_cov <- glm(BoniVote ~ passage*FonGroup+age+sexe, data=dt1, family = binomial(link ='logit'))
summary(_cov)
# This result seems robust with inclusion of "basic" covariables

em1 <-  emmeans(m1, ~passage * FonGroup,transform=TRUE)
em1 <- as.data.frame(em1)
em1 

plot1 = ggplot(em1, aes(x= passage,y=prob,fill=FonGroup)) +
  geom_bar(stat="identity",position="dodge") +
  scale_y_continuous(breaks=c(0,0.25,0.50,0.75,1),limits=c(0,1)) +
  ggtitle('Voting ~ priming and co-ethnicity with first lady ') + xlab('priming') + ylab('Voting intention') + 
  labs(fill = "Fon ethnicity") +
  geom_errorbar(aes(ymin=asymp.LCL, ymax=asymp.UCL), width=0.3, position=position_dodge(0.9)) +
  theme_bw() +
  theme(axis.text.x=element_text(size=11)) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=.1))
plot1



# > Including all modalities reported in the Independent Variable ---------
# The DV is binary, let's use logistic regression
# The dataset shows 5 modalities for the experiment scenario 
# (passage: Control, Femme, Nago, Bariba, FemmeFon) 
# Therefore, the design is a 5(passage: Control, Femme, Nago, Bariba, FemmeFon) X 
# 2 (FonGroup - coethnicity with first lady) between-subjects
# The claim is an interaction : priming Fon ethnicity increases support for Yayi
# only among her co-ethnics
# The predictions calls for the use of Helmert contrasts for the experimental 
# condition
#      Control, Femme, Nago, Bariba, FemmeFon
# C1:    -1       1      0     0        0      
# C2:    -1      -1      2     0        0
# C3:    -1      -1     -1     3        0
# C4:    -1      -1     -1     -1       4
# We should observe an interaction between the fourth contrast and FonGroup

m2 <- glm(BoniVote ~ passage*FonGroup, data=dt, family = binomial(link ='logit'))
summary(m2)

# The null hypothesis cannot be rejected. 
# Fourth Helmert Contrast's interaction with FonGroup is not significant on the 
# 0.05 alpha level.
# Priming first lady's ethnicity vs other conditions does not significantly
# increase voting intention among co-ethnics of first lady

m2_cov <- glm(BoniVote ~ passage*FonGroup+age+sexe, data=dt, family = binomial(link ='logit'))
summary(m2_cov)
# Including these two covariates does not influence conclusions


em2 <-  emmeans(m2, ~passage * FonGroup,transform=TRUE)
em2 <- as.data.frame(em2)
em2

plot2 = ggplot(em2,aes(x= passage,y=prob,fill=FonGroup)) +
  geom_bar(stat="identity",position="dodge") +
  scale_y_continuous(breaks=c(0,0.25,0.50,0.75,1),limits=c(0,1)) +
  ggtitle('Voting ~ priming and co-ethnicity with first lady ') + xlab('priming') + ylab('Voting intention') + 
  labs(fill = "Fon ethnicity") +
  geom_errorbar(aes(ymin=asymp.LCL, ymax=asymp.UCL), width=0.3, position=position_dodge(0.9)) +
  theme_bw() +
  theme(axis.text.x=element_text(size=11)) +
  theme(panel.border = element_rect(colour = "black", fill=NA, size=.1))
plot2
