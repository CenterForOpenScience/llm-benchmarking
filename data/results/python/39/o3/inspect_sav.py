import pyreadstat, sys, pandas as pd, json, os
path = '/app/data/data/original/39/input/replication_data/SCORE_all data.sav'
try:
    df, meta = pyreadstat.read_sav(path)
    print('shape', df.shape)
    print(df.head())
except Exception as e:
    print('error', e)
