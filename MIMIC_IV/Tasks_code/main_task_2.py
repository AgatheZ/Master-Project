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

warnings.filterwarnings("ignore")

##Variables 
nb_hours = 24
random_state = 1
TBI_split = False
tuning = False
SHAP = True
imputation = 'carry_forward'
model_name = 'LightGBM'
task = 'ABPd'

assert model_name in ['RF', 'XGBoost', 'LightGBM', 'Stacking'], "Please specify a valid model name"
assert imputation in ['No','carry_forward', 'linear', 'multivariate'], "Please specify a valid imputation method"
assert task in ['ABPm', 'ABPs', 'ABPd'], "Please specify a valid vital sign"

##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')
features = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\Master-Project\MIMIC_IV\resources\features_reg.csv', header = None)
print('Data Loading - done')
if nb_hours == 24:
   features = features.loc[:173,2] 
else:
   features = features.loc[:,0]


#Preprocessing
pr = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)
final_data, labels = pr.task_2_pr(label = task)

X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

if tuning:
   xgboost_hyp_tuning = HypTuning(True, 5,  model_name, X_train, y_train, X_val, y_val, n_iter = 750, random_state = random_state)
   best_param = xgboost_hyp_tuning.hyperopt()

if model_name == 'XGBoost':
   if task == 'ABPm':
      best_param = {'alpha': 0.0, 'colsample_bytree': 0.7484520067034888, 'eta': 0.08835020412267972, 'gamma': 2.9087902143782856, 'max_depth': 7.0, 'min_child_weight': 3.0}
   if task == 'ABPd':
      best_param = {'alpha': 0.0, 'colsample_bytree': 0.7484520067034888, 'eta': 0.08835020412267972, 'gamma': 2.9087902143782856, 'max_depth': 7.0, 'min_child_weight': 3.0}
   if task == 'ABPs':
      best_param = {'alpha': 3.0, 'colsample_bytree': 0.8258389895401705, 'eta': 0.31598562225462623, 'gamma': 4.978976352591667, 'max_depth': 10.0, 'min_child_weight': 2.0}

   depth = best_param['max_depth']
   model =  XGBRegressor(random_state = random_state, 
                        max_depth = int(depth), 
                        gamma = best_param['gamma'],
                        eta = best_param['eta'],
                        alpha = (best_param['alpha']),
                        min_child_weight=(best_param['min_child_weight']),
                        colsample_bytree=best_param['colsample_bytree'])

if model_name == 'RF':
   if task == 'ABPm':
      best_param = {'criterion': "squared_error", 'max_depth': 15.0, 'max_features':'auto', 'n_estimators': 250}
   if task == 'ABPd':
      best_param = {'criterion': "squared_error", 'max_depth': 13.0, 'max_features':'auto', 'n_estimators': 291}
   if task == 'ABPs':
      best_param = {'criterion': "squared_error", 'max_depth': 12.0, 'max_features':'auto', 'n_estimators': 113}
   model = RandomForestRegressor(max_depth=best_param['max_depth'], random_state=random_state, n_estimators = best_param['n_estimators'], max_features = best_param['max_features'])

if model_name == 'LightGBM':
   if task == 'ABPm':
      best_param = {'colsample_by_tree': 0.6079487616414725, 'learning_rate': 0.16337264921768488, 'max_depth': 18.0, 'min_child_weight': 1.0019264444204, 'num_leaves': 32.0, 'reg_alpha': 0.5443469087358566, 'reg_lambda': 0.9756942986738498}
      best_param['boosting_type'] = 'gbdt'
   if task == 'ABPd':
      best_param = {'colsample_by_tree': 0.6239750363045334, 'learning_rate': 0.038513355506933473, 'max_depth': 13.0, 'min_child_weight': 4.01542483578733, 'num_leaves': 39.0, 'reg_alpha': 1.2611979742051511, 'reg_lambda': 0.9534079174578567}
      best_param['boosting_type'] = 'gbdt'
   if task == 'ABPs':
      best_param = {'colsample_by_tree': 0.8670779130689402, 'learning_rate': 0.18539149366816354, 'max_depth': 15.0, 'min_child_weight': 4.0164021855855765, 'num_leaves': 30.0, 'reg_alpha': 1.9501069537710507, 'reg_lambda': 0.7404629054827254}
      best_param['boosting_type'] = 'gbdt'
   model = lgb.LGBMRegressor(boosting_type=best_param['boosting_type'], num_leaves=int(best_param['num_leaves']), 
            max_depth= int(best_param['max_depth']), learning_rate=best_param['learning_rate'], reg_alpha=best_param['reg_alpha'], 
            reg_lambda = best_param['reg_lambda'], colsample_bytree= best_param['colsample_by_tree'], min_child_weight = best_param['min_child_weight'])

if model_name == 'Stacking':
   if task == 'ABPd':
      best_param = {'alpha': 0.0, 'colsample_bytree': 0.7484520067034888, 'eta': 0.08835020412267972, 'gamma': 2.9087902143782856, 'max_depth': 7.0, 'min_child_weight': 3.0}

      depth = best_param['max_depth']
      xgb = XGBRegressor(random_state = random_state, 
                           max_depth = int(depth), 
                           gamma = best_param['gamma'],
                           eta = best_param['eta'],
                           alpha = (best_param['alpha']),
                           min_child_weight=(best_param['min_child_weight']),
                           colsample_bytree=best_param['colsample_bytree'])

      best_param = {'colsample_by_tree': 0.6239750363045334, 'learning_rate': 0.038513355506933473, 'max_depth': 13.0, 'min_child_weight': 4.01542483578733, 'num_leaves': 39.0, 'reg_alpha': 1.2611979742051511, 'reg_lambda': 0.9534079174578567}
      best_param['boosting_type'] = 'gbdt'
      lgbm = lgb.LGBMRegressor(boosting_type=best_param['boosting_type'], num_leaves=int(best_param['num_leaves']), 
            max_depth= int(best_param['max_depth']), learning_rate=best_param['learning_rate'], reg_alpha=best_param['reg_alpha'], 
            reg_lambda = best_param['reg_lambda'], colsample_bytree= best_param['colsample_by_tree'], min_child_weight = best_param['min_child_weight'])
      best_param = {'criterion': "squared_error", 'max_depth': 13.0, 'max_features':'auto', 'n_estimators': 291}

      rf = RandomForestRegressor(max_depth=best_param['max_depth'], random_state=random_state, n_estimators = best_param['n_estimators'], max_features = best_param['max_features'])
      model = StackingCVClassifier(classifiers = (xgb, rf), meta_classifier=lgbm)

   if task == 'ABPs':
      best_param = {'alpha': 3.0, 'colsample_bytree': 0.8258389895401705, 'eta': 0.31598562225462623, 'gamma': 4.978976352591667, 'max_depth': 10.0, 'min_child_weight': 2.0}

      depth = best_param['max_depth']
      xgb = XGBRegressor(random_state = random_state, 
                           max_depth = int(depth), 
                           gamma = best_param['gamma'],
                           eta = best_param['eta'],
                           alpha = (best_param['alpha']),
                           min_child_weight=(best_param['min_child_weight']),
                           colsample_bytree=best_param['colsample_bytree'])

      best_param = {'colsample_by_tree': 0.8670779130689402, 'learning_rate': 0.18539149366816354, 'max_depth': 15.0, 'min_child_weight': 4.0164021855855765, 'num_leaves': 30.0, 'reg_alpha': 1.9501069537710507, 'reg_lambda': 0.7404629054827254}
      best_param['boosting_type'] = 'gbdt'
      lgbm = lgb.LGBMRegressor(boosting_type=best_param['boosting_type'], num_leaves=int(best_param['num_leaves']), 
            max_depth= int(best_param['max_depth']), learning_rate=best_param['learning_rate'], reg_alpha=best_param['reg_alpha'], 
            reg_lambda = best_param['reg_lambda'], colsample_bytree= best_param['colsample_by_tree'], min_child_weight = best_param['min_child_weight'])
      best_param = {'criterion': "squared_error", 'max_depth': 12.0, 'max_features':'auto', 'n_estimators': 113}
      rf = RandomForestRegressor(max_depth=best_param['max_depth'], random_state=random_state, n_estimators = best_param['n_estimators'], max_features = best_param['max_features'])
      model = StackingCVClassifier(classifiers = (xgb, rf), meta_classifier=lgbm)

   if task == 'ABPm':
      best_param = {'alpha': 0.0, 'colsample_bytree': 0.7484520067034888, 'eta': 0.08835020412267972, 'gamma': 2.9087902143782856, 'max_depth': 7.0, 'min_child_weight': 3.0}
      depth = best_param['max_depth']
      xgb = XGBRegressor(random_state = random_state, 
                           max_depth = int(depth), 
                           gamma = best_param['gamma'],
                           eta = best_param['eta'],
                           alpha = (best_param['alpha']),
                           min_child_weight=(best_param['min_child_weight']),
                           colsample_bytree=best_param['colsample_bytree'])

      best_param = {'colsample_by_tree': 0.6079487616414725, 'learning_rate': 0.16337264921768488, 'max_depth': 18.0, 'min_child_weight': 1.0019264444204, 'num_leaves': 32.0, 'reg_alpha': 0.5443469087358566, 'reg_lambda': 0.9756942986738498}
      best_param['boosting_type'] = 'gbdt'
      lgbm = lgb.LGBMRegressor(boosting_type=best_param['boosting_type'], num_leaves=int(best_param['num_leaves']), 
            max_depth= int(best_param['max_depth']), learning_rate=best_param['learning_rate'], reg_alpha=best_param['reg_alpha'], 
            reg_lambda = best_param['reg_lambda'], colsample_bytree= best_param['colsample_by_tree'], min_child_weight = best_param['min_child_weight'])
      best_param = {'criterion': "squared_error", 'max_depth': 15.0, 'max_features':'auto', 'n_estimators': 250}
      rf = RandomForestRegressor(max_depth=best_param['max_depth'], random_state=random_state, n_estimators = best_param['n_estimators'], max_features = best_param['max_features'])
      model = StackingCVClassifier(classifiers = (lgbm, rf), meta_classifier=xgb)
  


model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
pred = model.predict(X_test)

# plt.title('Analysis of the ABPm predictions')
# plt.plot(range(75), pred, c='red')
# plt.plot(range(75), y_test)
# plt.legend(['predicted values (XGBOOST)', 'original values'])
# plt.xlabel('Patient n°')
# plt.ylabel('ABPm (mmHg)')


plt.show()

eval = Evaluation(True, model, 'Tuned ' + model_name, final_data, labels, random_state, SHAP, features, nb_hours)
eval.evaluate_regression()

