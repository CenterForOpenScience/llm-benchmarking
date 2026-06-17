import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np

# Load the data
try:
    data = pd.read_csv('/app/data/Data_Cleaned_22102020.csv', encoding='ISO-8859-1')
except Exception as e:
    print(f'Failed to load dataset: {e}')

# Create the independent variable
conditions = np.where(data['Lot_check'].notna(), 0, np.where(data['Gift_check'].notna(), 1, np.nan))
data['Cond'] = pd.Categorical(conditions, categories=[0, 1], ordered=True)
data['Cond'] = data['Cond'].cat.rename_categories(['lottery', 'gift'])

# Create the dependent variable
wtp = np.where(data['Cond'] == 'lottery', data['Lot_WTPc'], np.where(data['Cond'] == 'gift', data['Gift_WTPc'], np.nan))
data['WTP'] = pd.to_numeric(wtp, errors='coerce')

data = data.dropna(subset=['WTP'])

# Create the checking variable
check = np.where(data['Cond'] == 'lottery', data['Lot_check'], np.where(data['Cond'] == 'gift', data['Gift_check'], np.nan))
data['check'] = check

# Select those who answered correctly (correct == 3)
dataS = data[data['check'] == 3]
dataS.to_csv('/app/data/dataS.csv', index=False)

# Parametric analysis for those who understood the instructions
lottery_wtp = dataS[dataS['Cond'] == 'lottery']['WTP']
gift_wtp = pd.to_numeric(dataS[dataS['Cond'] == 'gift']['WTP'], errors='coerce')

describe_lottery = DescrStatsW(lottery_wtp)
describe_gift = DescrStatsW(gift_wtp)

print('Lottery WTP Mean:', describe_lottery.mean)
print('Lottery WTP Std:', describe_lottery.std)
print('Lottery WTP Var:', describe_lottery.var)
print('Gift WTP Mean:', describe_gift.mean)
print('Gift WTP Std:', describe_gift.std)
print('Gift WTP Var:', describe_gift.var)

# T-test
ttest_result = ttest_ind(lottery_wtp, gift_wtp)
print('T-test result:', ttest_result)

# Cohen's d
cohen_d = (describe_lottery.mean - describe_gift.mean) / np.sqrt((describe_lottery.var + describe_gift.var) / 2)
print("Cohen's d:", cohen_d)

# Select those who answered 20 or less
dataSC = dataS[dataS['WTP'] <= 20]

lottery_wtp_sc = dataSC[dataSC['Cond'] == 'lottery']['WTP']
gift_wtp_sc = dataSC[dataSC['Cond'] == 'gift']['WTP']

describe_lottery_sc = DescrStatsW(lottery_wtp_sc)
describe_gift_sc = DescrStatsW(gift_wtp_sc)

print('Lottery WTP <= 20 Mean:', describe_lottery_sc.mean)
print('Lottery WTP <= 20 Std:', describe_lottery_sc.std)
print('Lottery WTP <= 20 Var:', describe_lottery_sc.var)
print('Gift WTP <= 20 Mean:', describe_gift_sc.mean)
print('Gift WTP <= 20 Std:', describe_gift_sc.std)
print('Gift WTP <= 20 Var:', describe_gift_sc.var)

# T-test for WTP <= 20
ttest_result_sc = ttest_ind(lottery_wtp_sc, gift_wtp_sc)
print('T-test result for WTP <= 20:', ttest_result_sc)

# Cohen's d for WTP <= 20
cohen_d_sc = (describe_lottery_sc.mean - describe_gift_sc.mean) / np.sqrt((describe_lottery_sc.var + describe_gift_sc.var) / 2)
print("Cohen's d for WTP <= 20:", cohen_d_sc)