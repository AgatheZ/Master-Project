import numpy as np
import pandas as pd
import warnings
from scipy.stats import uniform
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier, XGBRegressor
from evaluation import Evaluation
from preprocessing import Preprocessing
from hyp_tuning import HypTuning
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt 

df = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\ABP_raw_data.csv', delimiter=',')
pr = Preprocessing()

ds = df.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'vital_name', values = 'vital_reading')
ds = ds.reset_index(level=['stay_id'])
idx_ABPm = np.load(r'C:\Users\USER\OneDrive\Summer_project\Azure\Master-Project\MIMIC_IV\Database analysis\preprocessed_batches\cohort_ABPm.npy', allow_pickle = True)


labels = ds[ds.index == 25]
print(labels)
labels = labels.dropna()
task2_cohort = labels['stay_id']
print(task2_cohort)

import sys
sys.exit()
batchs = []
print(idx_ABPm)
ids = idx_ABPm
for i in ids:
    batchs.append(ds.loc[ds['stay_id'] == i])

np.save('ABP_batches.npy', batchs)