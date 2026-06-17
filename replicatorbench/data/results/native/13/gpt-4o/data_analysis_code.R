# Replication of:
# "The Cultural Divide in Europe: Migration, Multiculturalism, and Political Trust"
# by Lauren M. McLaren
# World Politics, Volume 64, Issue 2April 2012 , pp. 199-241
# DOI: https://doi.org/10.1017/S0043887112000032
# 
# Data analysis code
# June 26, 2020
#
# Marta Kolczynska, mkolczynska@gmail.com

setwd('/workspace/replication_data')
# 1. SETUP ----------

sessionInfo()

# R version 3.6.3 (2020-02-29)
# Platform: x86_64-w64-mingw32/x64 (64-bit)
# Running under: Windows 10 x64 (build 18362)
# 
# Matrix products: default
# 
# locale:
#   [1] LC_COLLATE=English_United States.1252  LC_CTYPE=English_United States.1252    LC_MONETARY=English_United States.1252
# [4] LC_NUMERIC=C                           LC_TIME=English_United States.1252    
# 
# attached base packages:
#   [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# loaded via a namespace (and not attached):
#   [1] compiler_3.6.3 tools_3.6.3    packrat_0.5.0 


## 1.1. Packages ----------

library(lme4) # for estimating multi-level models
library(mice) # for imputation and analyzing imputed data


# 2. Reading in the data ----------

# complete-case survey data
data_clean_5pct <- readRDS("data_clean_5pct.rds")

# imputed data
data_imp_5pct <- readRDS("data_imp_5pct.rds")


# 3. Analyses -----------

## 3.1 Main analysis (complete cases, weights) -----------

m1 <- lmer(trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc +
             stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +
             vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female + 
             vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp + (1 | cntry),
           weights = pspwght,
           data = data_clean_5pct)

summary(m1)


# Linear mixed model fit by REML ['lmerMod']
# Formula: trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev +  
#   distrust_soc + stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +  
#   vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ +  
#   female + vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc +      unemp + (1 | cntry)
# Data: data_clean_5pct
# Weights: pspwght
# 
# REML criterion at convergence: 3694.5
# 
# Scaled residuals: 
#   Min      1Q  Median      3Q     Max 
# -3.5752 -0.6246 -0.0678  0.5711  3.5389 
# 
# Random effects:
#   Groups   Name        Variance Std.Dev.
# cntry    (Intercept) 0.0292   0.1709  
# Residual             3.3620   1.8336  
# Number of obs: 858, groups:  cntry, 13
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)    3.246e+00  9.812e-01   3.308
# imm_concern    1.797e-01  3.864e-02   4.651
# happy_rev     -8.221e-02  5.449e-02  -1.509
# stflife_rev    1.577e-01  5.239e-02   3.009
# sclmeet_rev    8.809e-02  4.717e-02   1.868
# distrust_soc   6.500e-02  4.605e-02   1.412
# stfeco_rev     2.367e-01  3.805e-02   6.221
# hincfel        3.787e-02  1.001e-01   0.378
# stfhlth_rev    1.117e-02  3.419e-02   0.327
# stfedu_rev     1.281e-01  3.599e-02   3.560
# vote_gov1     -4.757e-01  1.401e-01  -3.394
# vote_frparty1  2.782e-01  3.601e-01   0.773
# lrscale       -3.453e-02  3.403e-02  -1.015
# hhinc_std     -1.384e-01  8.175e-02  -1.693
# agea          -4.726e-03  3.932e-03  -1.202
# educ          -1.021e-01  5.105e-02  -2.000
# female        -2.457e-02  1.321e-01  -0.186
# vote_share_fr -6.641e-02  3.264e-02  -2.034
# socexp        -9.360e-05  9.134e-05  -1.025
# lt_imm_cntry  -9.512e-01  6.520e-01  -1.459
# wgi           -4.637e-01  5.646e-01  -0.821
# gdppc          7.341e-05  2.792e-05   2.630
# unemp         -1.136e-01  5.120e-02  -2.219
# 
# Correlation matrix not shown by default, as p = 23 > 12.
# Use print(x, correlation=TRUE)  or
# vcov(x)        if you need it
# 
# fit warnings:
#   Some predictor variables are on very different scales: consider rescaling


## 3.2 Auxiliary analysis 1 (complete cases, no weights) -----------

m2 <- lmer(trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc +
             stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +
             vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female + 
             vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp + (1 | cntry),
           data = data_clean_5pct)

summary(m2)

# Linear mixed model fit by REML ['lmerMod']
# Formula: trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev +  
#   distrust_soc + stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +  
#   vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ +  
#   female + vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc +      unemp + (1 | cntry)
# Data: data_clean_5pct
# 
# REML criterion at convergence: 3627.7
# 
# Scaled residuals: 
#   Min      1Q  Median      3Q     Max 
# -3.6483 -0.6477 -0.0737  0.6150  2.9536 
# 
# Random effects:
#   Groups   Name        Variance Std.Dev.
# cntry    (Intercept) 0.00     0.000   
# Residual             3.58     1.892   
# Number of obs: 858, groups:  cntry, 13
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)    2.980e+00  8.623e-01   3.455
# imm_concern    2.145e-01  3.925e-02   5.465
# happy_rev     -1.045e-01  5.506e-02  -1.897
# stflife_rev    1.505e-01  5.354e-02   2.811
# sclmeet_rev    8.803e-02  4.746e-02   1.855
# distrust_soc   8.275e-02  4.640e-02   1.784
# stfeco_rev     2.582e-01  3.798e-02   6.796
# hincfel        3.160e-02  1.001e-01   0.316
# stfhlth_rev    5.227e-02  3.535e-02   1.479
# stfedu_rev     9.396e-02  3.623e-02   2.593
# vote_gov1     -3.669e-01  1.404e-01  -2.613
# vote_frparty1  2.402e-01  3.846e-01   0.624
# lrscale       -5.029e-02  3.483e-02  -1.444
# hhinc_std     -1.134e-01  8.351e-02  -1.358
# agea          -5.627e-03  4.071e-03  -1.382
# educ          -1.100e-01  5.108e-02  -2.153
# female        -2.732e-02  1.330e-01  -0.205
# vote_share_fr -7.272e-02  2.740e-02  -2.654
# socexp        -1.531e-04  7.600e-05  -2.015
# lt_imm_cntry  -1.306e+00  5.447e-01  -2.397
# wgi           -1.699e-02  4.757e-01  -0.036
# gdppc          8.837e-05  2.309e-05   3.827
# unemp         -1.448e-01  4.246e-02  -3.410
# 
# Correlation matrix not shown by default, as p = 23 > 12.
# Use print(x, correlation=TRUE)  or
# vcov(x)        if you need it
# 
# fit warnings:
#   Some predictor variables are on very different scales: consider rescaling
# convergence code: 0
# boundary (singular) fit: see ?isSingular


## 3.3 Auxiliary analysis 2 (imputed data, weights) -----------

fitimp1 <- with(data_imp_5pct,
               lmer(trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc +
                      stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +
                      vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female + 
                      vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp + (1 | cntry),
                    weights = pspwght))

summary(pool(fitimp1))

# term      estimate    std.error  statistic         df      p.value
# 1    (Intercept)  2.816689e+00 8.221700e-01  3.4259212  451.61567 6.687248e-04
# 2    imm_concern  1.423741e-01 3.119772e-02  4.5636050 1208.19658 5.541157e-06
# 3      happy_rev -4.983350e-02 4.560246e-02 -1.0927809  805.54346 2.748167e-01
# 4    stflife_rev  7.047185e-02 4.111709e-02  1.7139311  818.63266 8.691993e-02
# 5    sclmeet_rev  3.297687e-02 3.890915e-02  0.8475351  472.48439 3.971261e-01
# 6   distrust_soc  1.626593e-01 3.786510e-02  4.2957577  629.47191 2.016327e-05
# 7     stfeco_rev  2.771359e-01 3.316246e-02  8.3569146  239.35237 5.329071e-15
# 8        hincfel -5.732207e-02 8.306497e-02 -0.6900872 1107.58966 4.902839e-01
# 9    stfhlth_rev  5.563212e-02 2.951422e-02  1.8849260  372.91831 6.021716e-02
# 10    stfedu_rev  1.259361e-01 3.149656e-02  3.9984089  291.20678 8.089455e-05
# 11     vote_gov1 -2.859426e-01 1.236809e-01 -2.3119386  781.13063 2.104033e-02
# 12 vote_frparty1  1.669440e-01 3.400873e-01  0.4908858 1172.40909 6.235991e-01
# 13       lrscale -5.498991e-02 3.573257e-02 -1.5389295   34.40401 1.329694e-01
# 14     hhinc_std -1.738076e-01 7.689969e-02 -2.2601861   64.11253 2.721389e-02
# 15          agea -2.954197e-03 3.197596e-03 -0.9238808  500.03315 3.559940e-01
# 16          educ -8.500129e-02 4.405474e-02 -1.9294472  419.87255 5.434886e-02
# 17        female  6.804313e-02 1.115148e-01  0.6101711  653.26185 5.419607e-01
# 18 vote_share_fr -7.867001e-02 3.095496e-02 -2.5414351 1235.93419 1.116120e-02
# 19        socexp -3.232296e-05 8.388477e-05 -0.3853257 1203.17571 7.000642e-01
# 20  lt_imm_cntry -9.573876e-01 4.922548e-01 -1.9449023 1237.44329 5.201302e-02
# 21           wgi -4.402953e-01 4.968003e-01 -0.8862622 1246.29015 3.756472e-01
# 22         gdppc  6.614430e-05 2.555400e-05  2.5884131 1231.41218 9.755476e-03
# 23         unemp -1.224877e-01 4.030609e-02 -3.0389375 1159.40209 2.427313e-03


## 3.4 Auxiliary analysis 3 (imputed data, no weights) -----------

fitimp2 <- with(data_imp_5pct,
                lmer(trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc +
                       stfeco_rev + hincfel + stfhlth_rev + stfedu_rev +
                       vote_gov + vote_frparty + lrscale + hhinc_std + agea + educ + female + 
                       vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp + (1 | cntry)))

summary(pool(fitimp2))


# term      estimate    std.error    statistic         df      p.value
# 1    (Intercept)  2.670425e+00 6.820727e-01  3.915162454  931.25755 9.693099e-05
# 2    imm_concern  1.733889e-01 3.111443e-02  5.572619399 1244.66947 3.073233e-08
# 3      happy_rev -3.984772e-02 4.427729e-02 -0.899958569 1176.99260 3.683264e-01
# 4    stflife_rev  6.032247e-02 4.131462e-02  1.460075785 1138.43822 1.445451e-01
# 5    sclmeet_rev  6.519698e-02 3.776982e-02  1.726165822  776.34841 8.471544e-02
# 6   distrust_soc  1.442623e-01 3.720122e-02  3.877892535  988.38166 1.123306e-04
# 7     stfeco_rev  2.832967e-01 3.073871e-02  9.216286016 1210.79162 0.000000e+00
# 8        hincfel -1.648051e-02 8.131428e-02 -0.202676669 1070.52447 8.394263e-01
# 9    stfhlth_rev  8.005235e-02 2.902248e-02  2.758287879  948.94275 5.922195e-03
# 10    stfedu_rev  1.087282e-01 3.101416e-02  3.505760697  352.97420 5.140006e-04
# 11     vote_gov1 -2.670967e-01 1.193788e-01 -2.237388784 1161.03361 2.545008e-02
# 12 vote_frparty1  1.000650e-01 3.494422e-01  0.286356421 1237.24931 7.746531e-01
# 13       lrscale -6.169859e-02 3.015253e-02 -2.046216038  501.96215 4.125558e-02
# 14     hhinc_std -1.551995e-01 7.601284e-02 -2.041754293   89.44537 4.412153e-02
# 15          agea -4.243024e-03 3.134274e-03 -1.353750000 1205.77576 1.760698e-01
# 16          educ -9.205973e-02 4.230064e-02 -2.176320097  889.41254 2.979383e-02
# 17        female  8.929983e-04 1.105279e-01  0.008079392  750.23635 9.935558e-01
# 18 vote_share_fr -7.701945e-02 2.459950e-02 -3.130935119 1222.91623 1.783804e-03
# 19        socexp -7.656541e-05 6.729134e-05 -1.137819684 1202.30676 2.554225e-01
# 20  lt_imm_cntry -1.076957e+00 3.907036e-01 -2.756455645 1210.87103 5.930883e-03
# 21           wgi -2.017787e-01 3.978470e-01 -0.507176717 1244.09435 6.121207e-01
# 22         gdppc  7.336745e-05 1.999235e-05  3.669775968 1243.82138 2.530379e-04
# 23         unemp -1.387186e-01 3.162545e-02 -4.386295851 1180.15849 1.255900e-05


