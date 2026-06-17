import pandas as pd
import numpy as np
from semopy import Model

# Load the data
df = pd.read_csv('/app/data/Kachanoff_Survey_deidentify.csv')

# Data Cleaning and Composite Scores
# Adjust scores as per R script
bai_cols = df.filter(regex='bai|attention_check1').columns
df[bai_cols] = df[bai_cols] - 1

covid_cols = df.filter(regex='covid').columns
ies_cols = df.filter(regex='instrusion|avoid').columns
df[ies_cols] = df[ies_cols] - 1

sds_cols = df.filter(regex='sds').columns
df['sds1'] = 8 - df['sds1']
df['sds2'] = 8 - df['sds2']

# Create composite scores
df['BAI_total'] = df.filter(regex='bai').sum(axis=1)
df['Realistic'] = df.filter(regex='covid_real').mean(axis=1)
df['Symbolic'] = df.filter(regex='covid_symbolic').mean(axis=1)
df['Intrusion'] = df.filter(regex='intrusion').sum(axis=1)
df['Avoidance'] = df.filter(regex='avoid').sum(axis=1)
df['SWLS'] = df.filter(regex='swls').mean(axis=1)
df['Positive'] = df.filter(regex='positive').sum(axis=1)
df['Negative'] = df.filter(regex='negative').sum(axis=1)
df['Social'] = df.filter(regex='social').sum(axis=1)
df['SDS'] = df.filter(regex='sds').mean(axis=1)
df['Norms'] = df.filter(regex='behave_norm').mean(axis=1)
df['American'] = df.filter(regex='behave_american').mean(axis=1)

# Convert long to wide format
df = df.sort_values('created')
variables = ['participant_id', 'created', 'BAI_total', 'Realistic', 'Symbolic', 'Intrusion', 'Avoidance', 'SWLS', 'Positive', 'Negative', 'Social', 'SDS', 'Norms', 'American', 'handwashing']
df_reduced = df[variables]
df_reduced['time2'] = df_reduced.duplicated('participant_id')
df_time1 = df_reduced[df_reduced['time2'] == False]
df_time2 = df_reduced[df_reduced['time2'] == True]
df_wide = pd.merge(df_time1, df_time2, on='participant_id', suffixes=('.x', '.y'))

# SEM Model
model_desc = '''
BAI_total.y ~ Realistic.x + Symbolic.x
Intrusion.y ~ Realistic.x + Symbolic.x
Avoidance.y ~ Realistic.x + Symbolic.x
SWLS.y ~ Realistic.x + Symbolic.x
Positive.y ~ Realistic.x + Symbolic.x
Negative.y ~ Realistic.x + Symbolic.x
Social.y ~ Realistic.x + Symbolic.x
handwashing.y ~ Realistic.x + Symbolic.x
SDS.y ~ Realistic.x + Symbolic.x
American.y ~ Realistic.x + Symbolic.x
Norms.y ~ Realistic.x + Symbolic.x
Realistic.x ~~ Symbolic.x
BAI_total.y ~~ Avoidance.y
BAI_total.y ~~ Intrusion.y
BAI_total.y ~~ SWLS.y
BAI_total.y ~~ Positive.y
BAI_total.y ~~ Negative.y
BAI_total.y ~~ Social.y
BAI_total.y ~~ handwashing.y
BAI_total.y ~~ American.y
BAI_total.y ~~ Norms.y
Avoidance.y ~~ Intrusion.y
Avoidance.y ~~ SWLS.y
Avoidance.y ~~ Positive.y
Avoidance.y ~~ Negative.y
Avoidance.y ~~ Social.y
Avoidance.y ~~ handwashing.y
Avoidance.y ~~ SDS.y
Avoidance.y ~~ American.y
Avoidance.y ~~ Norms.y
Intrusion.y ~~ SWLS.y
Intrusion.y ~~ Positive.y
Intrusion.y ~~ Negative.y
Intrusion.y ~~ Social.y
Intrusion.y ~~ handwashing.y
Intrusion.y ~~ SDS.y
Intrusion.y ~~ American.y
Intrusion.y ~~ Norms.y
SWLS.y ~~ Positive.y
SWLS.y ~~ Negative.y
SWLS.y ~~ Social.y
SWLS.y ~~ handwashing.y
SWLS.y ~~ SDS.y
SWLS.y ~~ American.y
SWLS.y ~~ Norms.y
Positive.y ~~ Negative.y
Positive.y ~~ Social.y
Positive.y ~~ handwashing.y
Positive.y ~~ SDS.y
Positive.y ~~ American.y
Positive.y ~~ Norms.y
Negative.y ~~ Social.y
Negative.y ~~ handwashing.y
Negative.y ~~ SDS.y
Negative.y ~~ American.y
Negative.y ~~ Norms.y
Social.y ~~ SDS.y
Social.y ~~ American.y
Social.y~~ Norms.y
handwashing.y ~~ SDS.y
handwashing.y ~~ American.y
handwashing.y~~ Norms.y
Social.y~~handwashing.y
SDS.y ~~ American.y
SDS.y ~~ Norms.y
American.y ~~ Norms.y
'''

model = Model(model_desc)
model.fit(df_wide)

# Save results
results = model.inspect()
results.to_csv('/app/data/sem_results.csv', index=False)
