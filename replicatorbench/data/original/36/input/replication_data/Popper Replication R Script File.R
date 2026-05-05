suppressPackageStartupMessages(library(lavaan))
has_semPlot <- requireNamespace("semPlot", quietly = TRUE)
has_lavaanPlot <- requireNamespace("lavaanPlot", quietly = TRUE)
suppressPackageStartupMessages(library(psych))
has_apaTables <- requireNamespace("apaTables", quietly = TRUE)
if (!has_semPlot) message("semPlot not installed; skipping path diagrams.")
if (!has_lavaanPlot) message("lavaanPlot not installed; skipping measurement model diagrams.")
if (!has_apaTables) message("apaTables not installed; falling back to CSV for correlations.")

# Read datasets from mounted volume
Popper_Data.for.CFA <- read.csv('replication_data/Popper Replication Data Files/Popper_Data for CFA and SEM.csv')
Popper.Data.for.Correlations <- read.csv('replication_data/Popper Replication Data Files/Popper Data for Correlations.csv')
# Drop NA-only columns for correlations
Popper.Data.for.Correlations <- Popper.Data.for.Correlations[, colSums(!is.na(Popper.Data.for.Correlations)) > 0]

Cordata<-Popper.Data.for.Correlations


if (has_apaTables) {
  apaTables::apa.cor.table(
    Cordata,
    filename = "/app/data/PopperAmitCorr1.doc",
    table.number = 1,
    show.sig.stars = TRUE,
    landscape = TRUE
  )
} else {
  # Fallback: save Pearson correlation matrix as CSV
  cmat <- suppressWarnings(cor(Cordata, use = "pairwise.complete.obs"))
  write.csv(cmat, "/app/data/PopperAmitCorr1.csv", row.names = TRUE)
}

##Scale Reliabilities##Scale Reliabilities
# Compute reliability (Cronbach's alpha) using available parcel indicators per construct.
reliability_alpha <- function(df, cols, label) {
  missing <- setdiff(cols, names(df))
  if (length(missing) == 0) {
    cat(sprintf("Computing alpha for %s on %d indicators...\n", label, length(cols)))
    print(psych::alpha(df[, cols], warnings = FALSE))
  } else {
    cat(sprintf("Skipping alpha for %s; missing columns: %s\n", label, paste(missing, collapse = ", ")))
  }
}

reliability_alpha(Popper_Data.for.CFA, c("AvoidC_Par1","AvoidD_Par2","AttachX_Par3"), "ATTACH")
reliability_alpha(Popper_Data.for.CFA, c("STAI_Par1","STAI_Par2","STAI_Par3"), "ANXIETY")
reliability_alpha(Popper_Data.for.CFA, c("Open_Par1","Open_Par2","Open_Par3"), "OPEN")
reliability_alpha(Popper_Data.for.CFA, c("Lead_Par1","Lead_Par2","Lead_Par3"), "LEAD")


##Confirmatory Factor Analysis for study variables
##Confirmatory Factor Analysis for study variables

PopperModel<-'ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3
'

fitPopper <- cfa(PopperModel, data = Popper_Data.for.CFA, std.ov = TRUE) 
summary(fitPopper, fit.measures = TRUE)

standardizedsolution(fitPopper)
modindices(fitPopper)

##Visualize measurement model
if (has_lavaanPlot) {
  lavaanPlot::lavaanPlot(model = fitPopper, node_options = 
               list(shape = "box", fontname =  "Helvetica"), 
             edge_options = list(color = "grey"), coefs = TRUE, covs=TRUE)
}


###One-Factor CFA
OneFactorModel<-'OneFactor=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3 +
STAI_Par1 + STAI_Par2 + STAI_Par3 + Open_Par1 + Open_Par2 + Open_Par3 + 
Lead_Par1 + Lead_Par2 + Lead_Par3
'

fitOneFactor <- cfa(OneFactorModel, data = Popper_Data.for.CFA, std.ov = TRUE) 
summary(fitOneFactor, fit.measures = TRUE)

standardizedsolution(fitOneFactor)
modindices(fitOneFactor)

##Visualize measurement model
if (has_lavaanPlot) {
  lavaanPlot::lavaanPlot(model = fitOneFactor, node_options = 
               list(shape = "box", fontname =  "Helvetica"), 
             edge_options = list(color = "grey"), coefs = TRUE, covs=TRUE)
}

##Chi square difference test to compare 4 factor and 1 factor models
round(cbind(Model=inspect(fitPopper, 'fit.measures'),
            Model_1=inspect(fitOneFactor, 'fit.measures')), 3)
anova(fitPopper, fitOneFactor)


####################SEM Models ######################
##Define constructs; "Par" indicates parcel - Poppers Model
Model1<- 'ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

ANXIETY~a*ATTACH
OPEN~c*ATTACH
LEAD~b*ANXIETY + d*OPEN

##indirect effect anxiety path
ab := a*b
##indirect effect openness path
cd := c*d'

##Fit model  
fitModel1 <- sem(Model1, data = Popper_Data.for.CFA)
summary(fitModel1, standardized=TRUE, fit.measures=TRUE)


if (has_semPlot) semPlot::semPaths(fitModel1, whatLabels = "std", layout = "tree")

##Define constructs; "Par" indicates parcel
Model2<- 'ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

ANXIETY~ a*ATTACH
OPEN~ c*ATTACH
LEAD~ b*ANXIETY+ d*OPEN + e*ATTACH

##indirect effect anxiety path
ab:= a*b

##indirect effect openness path
cd:=c*d'



##Fit model 2
fitModel2 <- sem(Model2, data = Popper_Data.for.CFA)
summary(fitModel2, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel2, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 2
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel2, 'fit.measures')), 3)
anova(fitModel1, fitModel2)


##Model 3 - Only direct paths from all three IVs to Leader Experiences
Model3<-'
##Define constructs; "Par" indicates parcel
ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

LEAD~ATTACH + OPEN + ANXIETY
'

##Fit model 3
fitModel3 <- sem(Model3, data = Popper_Data.for.CFA)
summary(fitModel3, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel3, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 3
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel3, 'fit.measures')), 3)
anova(fitModel1, fitModel3)

##Model 4 - Alternative solution: Partial mediation with direct path from anxiety to lead experiences
Model4<-'
##Define constructs; "Par" indicates parcel
ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

ATTACH~ a*ANXIETY
OPEN~ c*ANXIETY
LEAD~ b*ATTACH + d*OPEN + e*ANXIETY

##indirect effect attachment path
ab:= a*b

##indirect effect openness path
cd:=c*d
'
##Fit model 4
fitModel4 <- sem(Model4, data = Popper_Data.for.CFA)
summary(fitModel4, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel4, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 4
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel4, 'fit.measures')), 3)
anova(fitModel1, fitModel4)

##Model 5 - Alternative solution: Partial mediation with direct path from openness to lead experiences
Model5<-'
##Define constructs; "Par" indicates parcel
ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

ANXIETY~ a*OPEN
ATTACH~ c*OPEN
LEAD~ b*ANXIETY + d*ATTACH + e*OPEN

##indirect effect anxiety path
ab:= a*b

##indirect effect attachment path
cd:=c*d
'
##Fit model 5
fitModel5 <- sem(Model5, data = Popper_Data.for.CFA)
summary(fitModel5, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel5, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 5
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel5, 'fit.measures')), 3)
anova(fitModel1, fitModel5)




##Model 6 - Alternative solution: Partial mediation with direct path from openness to lead experiences and correlation between attachment and anxiety 
Model6<-'
##Define constructs; "Par" indicates parcel
ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

ANXIETY~~ATTACH
OPEN~ b*ATTACH
LEAD~ a*ANXIETY + c*OPEN 

##indirect effect attachment path
bc:= b*c
'
##Fit model 6
fitModel6 <- sem(Model6, data = Popper_Data.for.CFA)
summary(fitModel6, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel6, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 6
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel6, 'fit.measures')), 3)
anova(fitModel1, fitModel6)

##Model 7 - model where leader experiences predict other vars

Model7<-'
##Define constructs; "Par" indicates parcel
ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN=~ Open_Par1 + Open_Par2 + Open_Par3
LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3

##Structural relationships

ATTACH~LEAD
OPEN~LEAD
ANXIETY~LEAD
'

##Fit model 7
fitModel7 <- sem(Model7, data = Popper_Data.for.CFA)
summary(fitModel3, standardized=TRUE, fit.measures=TRUE)

if (has_semPlot) semPlot::semPaths(fitModel7, whatLabels = "std", layout = "tree")

##Chi square difference test to compare Model 1 and Model 3
round(cbind(Model=inspect(fitModel1, 'fit.measures'),
            Model_1=inspect(fitModel7, 'fit.measures')), 3)
anova(fitModel1, fitModel7)
