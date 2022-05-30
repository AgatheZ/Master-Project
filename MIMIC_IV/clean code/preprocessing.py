
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import seaborn as sns

def trunc_length(ds, nb_hours):
#function that truncates the data to only consider the first nb_hours hours
    df = ds.loc[ds.index <=  nb_hours]
    return df

def pivot_table(ds):
#function to pivot the table for better data readability 
    df_final = ds.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'round')
    return df_final


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