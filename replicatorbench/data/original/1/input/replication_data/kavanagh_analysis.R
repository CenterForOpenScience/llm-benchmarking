{
  library(tidyverse)
  library(haven)
  library(glue)
  library(jtools)
  library(lubridate)
  library(huxtable)
  library(multcomp)
  library(lfe)
}

# Data merging code copied in from kavanagh_g66z_data_merge.R

# load input files

# 5% Sample
set.seed(2982)
county_variables <- read_csv('replication_data/county_variables.csv') %>%
  sample_frac(.05)
transportation <- read_csv('replication_data/transportation.csv')

# changes in distancing
flat_data <- transportation %>%
  mutate(prop_home = pop_home/(pop_home + pop_not_home),
         # Define the three time periods
         time_period = case_when(
    between(date, ymd('2020-02-16'),ymd('2020-02-29')) ~ 'AAA Reference',
    between(date, ymd('2020-03-19'),ymd('2020-04-01')) ~ 'March',
    between(date, ymd('2020-08-16'),ymd('2020-08-29')) ~ 'August')
  ) %>%
  filter(!is.na(time_period), !is.na(pop_home)) %>%
  group_by(time_period, fips, state) %>%
  # Average over county, time period
  summarize(prop_home = mean(prop_home, na.rm = TRUE)) %>%
  arrange(state, fips, time_period) %>%
  group_by(fips, state) %>%
  # Scale to 100
  mutate(prop_home_change = 100*(prop_home/first(prop_home) - 1)) %>%
  filter(time_period != 'AAA Reference') %>%
  # Reshape to get one variable for March and one for August
  pivot_wider(id_cols = c('fips','state'),
              names_from = 'time_period',
              values_from = c('prop_home','prop_home_change')) %>%
  # Bring in county-level data
  right_join(county_variables, by = 'fips')




# IQR of Trump support
trumpIQR <- county_variables %>%
  dplyr::select(fips, trump_share) %>%
  unique() %>%
  pull(trump_share) %>%
  quantile(c(.25, .75), na.rm = TRUE) %>%
  {.[2] - .[1]} %>%
  unname()

# Variable construction
flat_data <- flat_data %>%
  mutate(state = factor(state)) %>%
  dplyr::select(prop_home_change_March,
         prop_home_change_August,
         income_per_capita,
         trump_share,
         male_percent,
         percent_black,
         percent_hispanic,
         percent_college,
         percent_retail,
         percent_transportation,
         percent_hes,
         prop_rural,
         ten_nineteen,
         twenty_twentynine,
         thirty_thirtynine,
         forty_fortynine,
         fifty_fiftynine,
         sixty_sixtynine,
         seventy_seventynine,
         over_eighty,
         state,
         fips) %>%
  ungroup() %>%
  # These are stored as 0-1 but everything else is 0-100
  mutate(across(starts_with('percent_'),function(x) x*100)) %>%
  mutate(male_percent = male_percent*100,
         percent_college = percent_college/100) %>%
  mutate(income_per_capita = income_per_capita/1000)


# Create regression formulae
formula_maker <- function(depvar, data) {
  vnames <- data %>%
    dplyr::select(-fips, -prop_home_change_March, -prop_home_change_August, -state) %>%
    names()
  
  form <- paste0(depvar,'~',
                 paste(vnames, collapse ='+'),
                 ' | state')

  return(as.formula(form))  
}

# Run fixed effect regressions
m1 <- felm(formula_maker('prop_home_change_March',flat_data), data = flat_data)
m2 <- felm(formula_maker('prop_home_change_August',flat_data), data = flat_data)

# Regression table
results_tab <- export_summs(m1, m2,
             digits = 3,
             model.names = c('March 19-April 1','August 16-29'),
             coefs = c('Income per Capita (Thousands)' = 'income_per_capita',
                       'Share of Trump Voters' = 'trump_share',
                       'Percent Male' = 'male_percent',
                       'Percent Black' = 'percent_black',
                       'Percent Hispanic' = 'percent_hispanic',
                       'Percent with College Degree' = 'percent_college',
                       'Percent in Retail' = 'percent_retail',
                       'Percent in Transportation' = 'percent_transportation',
                       'Percent in Health / Ed / Soc. Svcs' = 'percent_hes',
                       'Percent Rural' = 'prop_rural',
                       'Percent Age 10-19' = 'ten_nineteen',
                       'Percent Age 20-29' = 'twenty_twentynine',
                       'Percent Age 30-39' = 'thirty_thirtynine',
                       'Percent Age 40-49' = 'forty_fortynine',
                       'Percent Age 50-59' = 'fifty_fiftynine',
                       'Percent Age 60-69' = 'sixty_sixtynine',
                       'Percent Age 70-79' = 'seventy_seventynine',
                       'Percent Age 80+' = 'over_eighty'),
             statistics = c(N = 'nobs',
                            R2 = 'r.squared')) %>%
  add_footnote('More-positive numbers indicate more stay-at-home activity. State fixed effects included.')

quick_html(results_tab, file = 'regression_table.html')

# Effect of a one-IQR change in Trump share
summary(glht(m1, paste0(trumpIQR,'*trump_share = 0')))
summary(glht(m2, paste0(trumpIQR,'*trump_share = 0')))

## Additional analysis: spatial autocorrelation
{
  library(tigris)
  library(spdep)
  library(sphet)
  library(spatialreg)
}

# Get information on central county latitude/longitude
counties <- counties()
counties <- as_tibble(counties[,c('STATEFP','COUNTYFP','INTPTLAT','INTPTLON')]) %>%
  mutate(fips = as.numeric(STATEFP)*1000 + as.numeric(COUNTYFP)) %>%
  dplyr::select(-geometry, -STATEFP, -COUNTYFP) %>%
  rename(lat = INTPTLAT, lon = INTPTLON) %>%
  mutate(lat = as.numeric(lat),
         lon = as.numeric(lon))

# Bring in to data
flat_data <- left_join(flat_data, counties)

# K nearest neighbors for spatial spillovers
kn <- knearneigh(as.matrix(flat_data[,c('lon','lat'), with = FALSE]), 5)
nb <- knn2nb(kn)
listw <- nb2listw(nb)

# Create regression formulae
formula_maker <- function(depvar, data) {
  vnames <- data %>%
    dplyr::select(-fips, -prop_home_change_March, -prop_home_change_August) %>%
    names()
  
  form <- paste0(depvar,'~',
                 paste(vnames, collapse ='+'))
  
  return(as.formula(form))  
}

# Run models with spatial autocorrelation term
m3 <- lagsarlm(formula_maker('prop_home_change_March',flat_data), data = flat_data, listw = listw)
m4 <- lagsarlm(formula_maker('prop_home_change_August',flat_data), data = flat_data, listw = listw)

# Regression table
results_tab <- export_summs(m3, m4,
                            digits = 3,
                            model.names = c('March 19-April 1','August 16-29'),
                            coefs = c('Income per Capita (Thousands)' = 'income_per_capita',
                                      'Share of Trump Voters' = 'trump_share',
                                      'Percent Male' = 'male_percent',
                                      'Percent Black' = 'percent_black',
                                      'Percent Hispanic' = 'percent_hispanic',
                                      'Percent with College Degree' = 'percent_college',
                                      'Percent in Retail' = 'percent_retail',
                                      'Percent in Transportation' = 'percent_transportation',
                                      'Percent in Health / Ed / Soc. Svcs' = 'percent_hes',
                                      'Percent Rural' = 'prop_rural',
                                      'Percent Age 10-19' = 'ten_nineteen',
                                      'Percent Age 20-29' = 'twenty_twentynine',
                                      'Percent Age 30-39' = 'thirty_thirtynine',
                                      'Percent Age 40-49' = 'forty_fortynine',
                                      'Percent Age 50-59' = 'fifty_fiftynine',
                                      'Percent Age 60-69' = 'sixty_sixtynine',
                                      'Percent Age 70-79' = 'seventy_seventynine',
                                      'Percent Age 80+' = 'over_eighty',
                                      'rho' = 'rho'),
                            statistics = c(N = 'nobs',
                                           R2 = 'r.squared')) %>%
  add_footnote('More-positive numbers indicate more stay-at-home activity.\nState fixed effects included.\nSpatial autocorrelation included with 5-nearest-neighbor neighbors.')

quick_html(results_tab, file = 'spatial_regression_table.html')
