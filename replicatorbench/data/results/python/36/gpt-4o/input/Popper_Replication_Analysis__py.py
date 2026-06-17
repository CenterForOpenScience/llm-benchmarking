import pandas as pd
import numpy as np
from semopy import Model
from semopy import Optimizer

# Load datasets
cor_data_path = '/app/data/Popper Replication Data Files/Popper Data for Correlations.csv'
cfa_data_path = '/app/data/Popper Replication Data Files/Popper_Data for CFA and SEM.csv'

cor_data = pd.read_csv(cor_data_path)
cfa_data = pd.read_csv(cfa_data_path)

# Define SEM model
model_desc = '''
ATTACH =~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3
ANXIETY =~ STAI_Par1 + STAI_Par2 + STAI_Par3
OPEN =~ Open_Par1 + Open_Par2 + Open_Par3
LEAD =~ Lead_Par1 + Lead_Par2 + Lead_Par3

ANXIETY ~ a*ATTACH
OPEN ~ c*ATTACH
LEAD ~ b*ANXIETY + d*OPEN

# ab := a*b
cd := c*d
'''

# Fit SEM model
model = Model(model_desc)
opt = Optimizer(model)
opt.optimize(cfa_data)

# Output results
print(model.inspect())

# Save results to a file
with open('/app/data/sem_results.txt', 'w') as f:
    f.write(model.inspect())
