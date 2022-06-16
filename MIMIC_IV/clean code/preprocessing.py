import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import seaborn as sns

def create_batchs(ds):
    batchs = []
    ids = ds.stay_id.unique()
    for i in ids:
        batchs.append(ds.loc[ds['stay_id'] == i])
    return batchs

def remove_missing(df, var, threshold):
#remove from batch the entries where a too large proportion of the variables var are missing 
    res = df
    
    
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                    'percent_missing': percent_missing})
    for vital in var: 
        criterion = missing_value_df.loc[missing_value_df.column_name == vital].percent_missing >= threshold 
        if criterion:
            print('entry removed')
            print(missing_value_df.loc[missing_value_df.column_name == vital].percent_missing)
            df.drop([vital], axis = 1)
        else:
            res.append(batch[i])
    return res

def get_column_name(df):
    listn = [col for col in df.columns]
    return listn

def aggregation(batch, rate):
    'function that takes a batch of patients and returns the aggregated vitals with the correct aggregation rate'
    if rate == 1:
        return batch
    elif rate == 24:
        bch = []
        for df in batch:
            df['hour_slice'] = 0
            df['hour_slice'][range(25,49)] = 1
            df = df.groupby('hour_slice').mean()
            bch.append(df)
        return bch
    elif rate == 48:
        bch = []
        for df in batch:
            df['hour_slice'] = 0
            df = df.groupby('hour_slice').mean()
            bch.append(df)
        return bch

def arrange_ids(df1, df2, df3, df4, df5):
    ids1 = df1.stay_id.unique()
    ids2 = df2.stay_id.unique()
    ids3 = df3.stay_id.unique()
    ids4 = df4.stay_id.unique()
    ids5 = df5.stay_id.unique()

    min_ids = list(set(ids1) & set(ids2) & set(ids3) & set(ids4) & set(ids5))
    return df1.loc[df1['stay_id'].isin(min_ids)], df2.loc[df2['stay_id'].isin(min_ids)], df3.loc[df3['stay_id'].isin(min_ids)], df4.loc[df4['stay_id'].isin(min_ids)], df5.loc[df5['stay_id'].isin(min_ids)]
