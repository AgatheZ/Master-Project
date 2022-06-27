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
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

##Variables 
nb_hours = 24
random_state = 1
TBI_split = False
tuning = False
SHAP = False
imputation = 'carry_forward'
model_name = 'Stacking'

assert model_name in ['RF', 'XGBoost', 'LightGBM', 'Stacking'], "Please specify a valid model name"
assert imputation in ['carry_forward', 'linear', 'multivariate'], "Please specify a valid imputation method"

##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')
features = pd.read_csv(r'MIMIC_IV\resources\features.csv', header = None)
print('Data Loading - done')

if nb_hours == 24:
   features = features.loc[:415,1] 
else:
   features = features.loc[:,0]


#Preprocessing
pr = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)
final_data, labels = pr.task_2_pr(label = 'ABPm')

print(final_data)
print(labels)
X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

model =  XGBRegressor(random_state = random_state)
eval = Evaluation(model, 'Tuned ' + model_name, final_data, labels, random_state, SHAP, features, nb_hours)
eval.evaluate_regression()

