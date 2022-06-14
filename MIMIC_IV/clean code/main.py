import numpy as np
import random
import pandas as pd
from hyperopt import fmin, tpe, hp, anneal, Trials
import warnings
from scipy.stats import uniform
import pickle

from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier

from evaluation import Evaluation
import preprocessing as pr
from hyp_tuning import HypTuning

warnings.filterwarnings("ignore")

##Variables 
nb_hours = 24
random_state = 1

##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\Documents\Imperial\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\Documents\Imperial\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\Documents\Imperial\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r'C:\Users\USER\Documents\Imperial\Summer_project\Azure\data\preprocessed_mimic4_med.csv', delimiter=',')
df_demographic = pd.read_csv(r'C:\Users\USER\Documents\Imperial\Summer_project\Azure\data\demographics_mimic4.csv', delimiter=',')

df_hourly = df_hourly.drop(columns = ['icu_intime'])
df_24h = df_24h.drop(columns = ['icu_intime'])
df_48h = df_48h.drop(columns = ['icu_intime'])

#Preprocessing
##truncate to only get 48 hours of stay.
df_hourly = pr.trunc_length(df_hourly, nb_hours)
df_24h = pr.trunc_length(df_24h, nb_hours//24)
df_demographic, df_med, df_hourly, df_24h, df_48h = pr.arrange_ids(df_demographic, df_med, df_hourly, df_24h, df_48h)
np.save('df_demographic', df_demographic, allow_pickle=True)

##label extraction 
labels = df_demographic.pop('los')
labels[labels <= 4] = 0
labels[labels > 4] = 1
labels = labels.values

##pivot the tables 
df_hourly = df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
df_24h = df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
df_48h = df_48h.pivot_table(index = ['stay_id'], columns = 'feature_name', values = 'feature_mean_value')
df_med = df_med.pivot_table(index = ['stay_id'], columns = 'med_name', values = 'amount')

##one-hot encoding for the medication and the sex
df_med = df_med.fillna(value = 0)
df_med[df_med > 0] = 1
df_demographic.gender[df_demographic.gender == 'F'] = 1
df_demographic.gender[df_demographic.gender == 'M'] = 0


##create batches 
df_hourly = df_hourly.reset_index(level=['stay_id'])
df_24h = df_24h.reset_index(level=['stay_id'])
df_48h = df_48h.reset_index(level=['stay_id'])
df_med = df_med.reset_index(level=['stay_id'])

batch_hourly = pr.create_batchs(df_hourly)
batch_24h = pr.create_batchs(df_24h)
batch_48h = pr.create_batchs(df_48h)
batch_med = pr.create_batchs(df_med)
batch_demographic = pr.create_batchs(df_demographic)

##reindex for patients that don't have entries at the begginning of their stays 
for i in range(len(batch_24h)):
    batch_hourly[i] = batch_hourly[i].reindex(range(1, 49), fill_value = None) 
    batch_24h[i] = batch_24h[i].reindex(range(1, 3), fill_value = None) 
    batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
    batch_24h[i] = batch_24h[i].drop(columns = 'stay_id')
    batch_48h[i] = batch_48h[i].drop(columns = 'stay_id')
    batch_med[i] = batch_med[i].drop(columns = 'stay_id')
    batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')


df_hourly = pd.concat(batch_hourly)
df_24h = pd.concat(batch_24h)

##the stay ids column are dropped since we alreasy took care of them being in the same order for all datasets
df_48h = df_48h.drop(columns = 'stay_id')
df_med = df_med.drop(columns = 'stay_id')

##first linear inputation and then replaced by mean when it's not possible 
##pas incroyable de recalculer le mean à chaque itération... é changer 

for i in range(len(batch_hourly)):
   batch_hourly[i] = batch_hourly[i].interpolate(limit = 15)
   batch_24h[i] = batch_24h[i].interpolate(limit = 15)

for i in range(len(batch_hourly)):
   batch_hourly[i] = batch_hourly[i].interpolate(limit = 15)
   batch_24h[i] = batch_24h[i].interpolate(limit = 15)
   batch_48h[i] = batch_48h[i].fillna(df_48h.mean())
   batch_24h[i] = batch_24h[i].fillna(df_24h.mean())
   batch_hourly[i] = batch_hourly[i].fillna(df_hourly.mean())
   batch_demographic[i].bmi = batch_demographic[i].bmi.fillna(0)
   batch_demographic[i].gcs = batch_demographic[i].gcs.fillna(df_demographic.gcs.mean())

# #save the preprocessed arrays
# np.save('batch_demographic', batch_demographic, allow_pickle=True)
# np.save('batch_hourly', batch_hourly, allow_pickle=True)
# np.save('batch_24h', batch_24h, allow_pickle=True)
# np.save('batch_48h', batch_48h, allow_pickle=True)

# np.save('df_hourly', df_hourly, allow_pickle=True)
# np.save('df_24h', df_24h, allow_pickle=True)
# np.save('df_48h', df_48h, allow_pickle=True)

#feature concatenation 
stratify_param = df_demographic.gcs
final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_48h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])
final_data = np.squeeze(final_data).astype('float64')
final_data = normalize(final_data)

#XGBOOST MODEL 
X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)


# # #hyperparameter tuning 
# kfold = KFold(n_splits=4, shuffle=True, random_state=random_state)
# space_hp ={'max_depth': (hp.quniform("max_depth", 3, 18, 1)),
#         'gamma': hp.uniform ('gamma', 0,9),
#         'alpha' : hp.quniform('alpha', 0,180,1),
#         'eta' : hp.uniform('eta', 0,0.4),

#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : (hp.quniform('min_child_weight', 0, 10, 1)),
#         'n_estimators': 180,
#         'seed': 0
#     }

# space_rd = {'max_depth': random.sample(range(3, 18), 2),
#         'gamma': np.random.uniform (1,9,10),
#         'alpha' : random.sample(range(0,180),10),
#         'eta' : np.random.uniform( 0,0.1,10),

#         'colsample_bytree' : np.random.uniform(0.5,1,10),
#         'min_child_weight' : random.sample(range(0, 10),10),
#         'n_estimators': [180],
#         'seed': [0]
#     }

# # xgboost_hyp_tuning = HypTuning(kfold, space_hp, X_train, y_train, X_val, y_val, n_iter = 1000, random_state =random_state)
# # best_param = xgboost_hyp_tuning.hyperopt()

# #final model training and test 
# # best_param = {'alpha': 6.0, 'colsample_bytree': 0.5352461591753516, 'eta': 0.2198100520631916, 'gamma': 0.8693907096181676, 'max_depth': 8.0, 'min_child_weight': 1.0}
# # depth = best_param['max_depth']
# # xgbc = XGBClassifier(random_state = random_state, 
# #                         max_depth = int(depth), 
# #                         gamma = best_param['gamma'],
# #                         eta = best_param['eta'],
# #                         alpha = (best_param['alpha']),

# #                         min_child_weight=(best_param['min_child_weight']),
# #                         colsample_bytree=best_param['colsample_bytree'])

# xgbc = XGBClassifier(random_state = random_state)
# xgbc.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

# #save the model 
# filename = 'LOS_XGBoost_random_sampling.sav'
# pickle.dump(xgbc, open(filename, 'wb'))

# #make the predictions
# y_pred = xgbc.predict(X_test)
# y_pred_proba = xgbc.predict_proba(X_test)
# y_pred_proba = y_pred_proba[:,1]

#model evaluation
model =  pickle.load(open('Master-Project\MIMIC_IV\models\LOS_XGBoost_24_tuned.sav', 'rb'))
eval = Evaluation(model, 'Tuned XGBoost', X_test, y_test)
eval.evaluate()


