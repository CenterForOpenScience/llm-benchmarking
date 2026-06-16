# first, load/install packages
if(!require(car)) {
  install.packages("car")
}

if(!require(here)) {
  install.packages("here")
  library(here)
}

if(!require(tidyverse)) {
  install.packages("tidyverse")
  library(tidyverse)
}

if(!require(effectsize)) {
  install.packages("effectsize")
  library(patchwork)
}


if(!dir.exists("data")) {
  dir.create("data")
}

osfr::osf_download()
# we need to download final_data.dta from the https://osf.io/46swk/?view_only=277b7d18bb24436eb5cfca69a07eb07a and store it in the data directory
if(!file.exists(here("data/final_data.dta"))) {
  stop("Download the data first")
}

# load data
df <- haven::read_dta(here("data/final_data.dta")) %>% 
  mutate(complaints_2008 = as.numeric(complaints_2008)) %>% # convert number of complaints to numeric
  mutate(first_A = if_else(first_A == 1, "A or number", "rest")) # use better names

# I first started by inspecting the data in the osf repo and reading through the paper. 
# After the reading, I was sure that the data I need is really final_data.dta

# the variable should be first_A
# I was not sure whether it also contains cases, where it starts with number, 
# but checking the other files, namely angieslist show that first_A is actually A and number


# desc stat ---------------------------------------------------------------

# lets start with some dexcriptive statistics

df %>% 
  group_by(first_A) %>% 
  summarize(n = n(), 
            min = min(complaints_2008), 
            max = max(complaints_2008),
            mean = mean(complaints_2008),
            median = median(complaints_2008),
            sd = sd(complaints_2008))

df %>% 
  group_by(first_A,num_names_2008) %>% 
  summarize(n = n(), 
            min = min(complaints_2008), 
            max = max(complaints_2008),
            mean = mean(complaints_2008),
            median = median(complaints_2008),
            sd = sd(complaints_2008))

# Ok, distributions are heavily skewed, but the extremes in both groups do not differ that much and thus t-tests won't be much off



# t-tests -----------------------------------------------------------------
# this could be tested with t-tests/wilcoxon tests

# this actually matches the statistics in the paper (Table 2)
t.test(complaints_2008~first_A,df,var.equal = T)


## assumptions

# not surprisingly, variances differ
car::leveneTest(complaints_2008~first_A, data = df)

## Welsch's t-test

t.test(complaints_2008~first_A,df) 

## non-parametric t-test

wilcox.test(complaints_2008~first_A,df)

## -> given the large differences, multiverse approach yields the same result


## Try some other parametrizations

effectsize::cohens_d(complaints_2008~first_A,data = df,pooled_sd = FALSE)
effectsize::cohens_d(complaints_2008~first_A,data = df,pooled_sd = T)

# note that simple usage of non-equal variances reduces the effect size 

# Causal analysis

# I also drawed causal diagram for possible confounders, however, only ad spending could be think of as possible mediator
# adding these variables into model did not change the results

lm(complaints_2008~first_A+ad_spend_k,df) %>% car::Anova()

lm(complaints_2008~first_A+total_price,df) %>% car::Anova()

lm(complaints_2008~first_A+has_fixed_fee+ad_spend_k+emp_size,df) %>% car::Anova()
lm(complaints_2008~first_A+num_names_2008,df) %>% car::Anova()

# story changes when we think about the number of names. 
# multiple firms have more than one name. Thus it is reasonable to assume that firms with more names has more complaints (as the paper states that number of complaints was computed as sum)

xtabs(~num_names_2008,df)
xtabs(~num_names_2008,df) %>% prop.table(margin =1)
xtabs(~num_names_2008+first_A,df) %>% prop.table(margin =1)

# looking at the table, we can see that proportion of A names firm increases

# we thus tried to run a model, in which we added number of names as factor 
lm(complaints_2008~first_A+num_names_2008,df) %>% car::Anova()
lm(complaints_2008~first_A*num_names_2008,df) %>% car::Anova()

# just for firms with 1 occurence
t.test(complaints_2008~first_A,df %>% filter(num_names_2008==1), var.equal = F)

# normalize names
# we can try to normalize number of complaints
t.test(complaints_2008~first_A,df %>% mutate(complaints_2008=complaints_2008/num_names_2008),var.equal = F)
effectsize::cohens_d(complaints_2008~first_A,data = df %>% mutate(complaints_2008=complaints_2008/num_names_2008),var.equal = F)

# we are at half of the effect size for all data

# we can visualize means + CI for different number of names conditions
df %>% 
  mutate(num_names_2008 = as.character(num_names_2008)) %>% 
  ggplot(aes(x = first_A, y = complaints_2008, col = num_names_2008, group = num_names_2008)) +
  stat_summary(fun.data = "mean_cl_boot") +
  stat_summary(fun = "mean", geom = "line")


# final model for reporting
# lm

lm1 <- lm(scale(complaints_2008)~first_A+scale(num_names_2008),df) 

lmF <- lm1 %>% car::Anova() %>% broom::tidy()
lmF$statistic[1] %>% round(3)
lmF$p.value[1] %>% round(3)
