import numpy as np
import random
import pandas as pd
from hyperopt import fmin, tpe, hp, anneal, Trials
import warnings
from scipy.stats import uniform
import pickle
from sklearn import preprocessing
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
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
pr.time_series_pr()

if TBI_split:
   final_data_mild, final_data_severe, labels_mild, labels_severe = pr.preprocess_data()
else:
   final_data, labels = pr.preprocess_data()
print('Data Preprocessing - done')


#XGBOOST MODEL 
if TBI_split:
   X_train, X_test, y_train, y_test = train_test_split(final_data_severe, labels_severe, test_size=0.2, shuffle = True, random_state=random_state)
   X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
else:
   X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
   X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

# #hyperparameter tuning 

if tuning:
   xgboost_hyp_tuning = HypTuning(5, model_name, X_train, y_train, X_val, y_val, n_iter = 1000, random_state = random_state)
   best_param = xgboost_hyp_tuning.hyperopt()

# final models evaluation using 5-fold cross validation  
if model_name == 'XGBoost':
   if TBI_split:
      best_param_mild = {'alpha': 0.0, 'colsample_bytree': 0.5404929749427543, 'eta': 0.08602706957522405, 'gamma': 1.8786165006154019, 'max_depth': 17.0, 'min_child_weight': 3.0}
      best_param_severe = {'alpha': 13.0, 'colsample_bytree': 0.6268493181974919, 'eta': 0.03821439650403947, 'gamma': 0.4555031107284915, 'max_depth': 15.0, 'min_child_weight': 10.0}
   else:
      if nb_hours == 24:
         best_param = {'alpha': 4.0, 'colsample_bytree': 0.6247402571982401, 'eta': 0.36200417198604184, 'gamma': 1.55002501704347, 'max_depth': 4.0, 'min_child_weight': 1.0}
         best_param_imput = {'alpha': 3.0, 'colsample_bytree': 0.5359922767597586, 'eta': 0.14476892774894637, 'gamma': 2.3216870446364952, 'max_depth': 18.0, 'min_child_weight': 1.0}
         best_param_carry_forw = {}
      else:
         best_param = {'alpha': 0.0, 'colsample_bytree': 0.7336138546940976, 'eta': 0.3108407290269413, 'gamma': 2.3874687080350965, 'max_depth': 10.0, 'min_child_weight': 4.0}
   depth = best_param['max_depth']
   model =  XGBClassifier(random_state = random_state, 
                        max_depth = int(depth), 
                        gamma = best_param['gamma'],
                        eta = best_param['eta'],
                        alpha = (best_param['alpha']),

                        min_child_weight=(best_param['min_child_weight']),
                        colsample_bytree=best_param['colsample_bytree'])

if model_name == 'RF':
   best_param = {'criterion': 'gini', 'max_features': 'sqrt', 'max_depth': 9.0, 'n_estimators': 500}
   model = RandomForestClassifier(max_depth=best_param['max_depth'], random_state=random_state, n_estimators = best_param['n_estimators'], max_features = best_param['max_features'])

if model_name == 'LightGBM':
   best_param = {'boosting_type': 'gbdt', 'colsample_by_tree': 0.9203696945188418, 'learning_rate': 0.04047240211150923, 'max_depth': 14.0, 'min_child_weight': 4.735811783605331, 'num_leaves': 32.0, 'reg_alpha': 0.33592465595298626, 'reg_lambda': 0.21263015922040293}
   model = lgb.LGBMClassifier(boosting_type=best_param['boosting_type'], num_leaves=int(best_param['num_leaves']), 
            max_depth= int(best_param['max_depth']), learning_rate=best_param['learning_rate'], reg_alpha=best_param['reg_alpha'], 
            reg_lambda = best_param['reg_lambda'], colsample_bytree= best_param['colsample_by_tree'], min_child_weight = best_param['min_child_weight'])

if model_name == 'Stacking':
   best_param = {'alpha': 3.0, 'colsample_bytree': 0.5359922767597586, 'eta': 0.14476892774894637, 'gamma': 2.3216870446364952, 'max_depth': 18.0, 'min_child_weight': 1.0}
   depth = best_param['max_depth']
   xgb = XGBClassifier(random_state = random_state, 
                        max_depth = int(depth), 
                        gamma = best_param['gamma'],
                        eta = best_param['eta'],
                        alpha = (best_param['alpha']),
                        min_child_weight=(best_param['min_child_weight']),
                        colsample_bytree=best_param['colsample_bytree'])

   best_param = {'boosting_type': 'gbdt', 'colsample_by_tree': 0.9203696945188418, 'learning_rate': 0.04047240211150923, 'max_depth': 14.0, 'min_child_weight': 4.735811783605331, 'num_leaves': 32.0, 'reg_alpha': 0.33592465595298626, 'reg_lambda': 0.21263015922040293}
   lgbm = lgb.LGBMClassifier()
   best_param = {'criterion': 'gini', 'max_features': 'sqrt', 'max_depth': 9.0, 'n_estimators': 500}
   rf = RandomForestClassifier()
   clf1 = KNeighborsClassifier()
   clf2 = GaussianNB()
   model = StackingCVClassifier(classifiers = (lgbm, rf, clf1, clf2), meta_classifier=lgbm)

#fits and evaluates the model
eval = Evaluation(model, 'Tuned ' + model_name, final_data, labels, random_state, SHAP, features, nb_hours)
eval.evaluate()

# #save the model 
# filename = 'LOS_XGBoost_random_sampling.sav'
# pickle.dump(xgbc, open(filename, 'wb'))





