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
TBI_split = True 

##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')
features = pd.read_csv(r'MIMIC_IV\resources\features.csv', header = None)



if nb_hours == 24:
   features = features.loc[:415,1] 
else:
   features = features.loc[:,0]

df_hourly = df_hourly.drop(columns = ['icu_intime'])
df_24h = df_24h.drop(columns = ['icu_intime'])
df_48h = df_48h.drop(columns = ['icu_intime'])

if TBI_split:
   df_demographic['severity'] = df_demographic.apply(lambda row: pr.label_severity(row), axis=1)
   mild_idx = df_demographic[df_demographic['severity'] == 'mild']['stay_id']
   severe_idx = df_demographic[df_demographic['severity'] == 'severe']['stay_id']
   df_demographic.pop('severity')

#Preprocessing

##label extraction 
if TBI_split:
   labels_mild = df_demographic[df_demographic.stay_id.isin(mild_idx)].pop('los')
   labels_mild[labels_mild <= 4] = 0
   labels_mild[labels_mild > 4] = 1
   labels_mild = labels_mild.values

   labels_severe = df_demographic[df_demographic.stay_id.isin(severe_idx)].pop('los')
   labels_severe[labels_severe <= 4] = 0
   labels_severe[labels_severe > 4] = 1
   labels_severe = labels_severe.values

   df_demographic.pop('los')

else:
   labels = df_demographic.pop('los')
   labels[labels <= 4] = 0
   labels[labels > 4] = 1
   labels = labels.values

##pivot the tables 
df_hourly = df_hourly.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
df_24h = df_24h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
df_48h = df_48h.pivot_table(index = ['stay_id', 'hour_from_intime'], columns = 'feature_name', values = 'feature_mean_value')
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


if TBI_split:
   batch_hourly_mild = pr.create_batchs(df_hourly[df_hourly.stay_id.isin( mild_idx)])
   batch_24h_mild = pr.create_batchs(df_24h[df_24h.stay_id.isin( mild_idx)])
   batch_48h_mild = pr.create_batchs(df_48h[df_48h.stay_id.isin( mild_idx)])
   batch_med_mild = pr.create_batchs(df_med[df_med.stay_id.isin( mild_idx)])
   batch_demographic_mild = pr.create_batchs(df_demographic[df_demographic.stay_id.isin( mild_idx)])
   batch_hourly_severe = pr.create_batchs(df_hourly[df_hourly.stay_id.isin(severe_idx)])
   batch_24h_severe = pr.create_batchs(df_24h[df_24h.stay_id.isin(severe_idx)])
   batch_48h_severe = pr.create_batchs(df_48h[df_48h.stay_id.isin(severe_idx)])
   batch_med_severe = pr.create_batchs(df_med[df_med.stay_id.isin(severe_idx)])
   batch_demographic_severe = pr.create_batchs(df_demographic[df_demographic.stay_id.isin(severe_idx)])
   
else:
   batch_hourly = pr.create_batchs(df_hourly)
   batch_24h = pr.create_batchs(df_24h)
   batch_48h = pr.create_batchs(df_48h)
   batch_med = pr.create_batchs(df_med)
   batch_demographic = pr.create_batchs(df_demographic)

##reindex for patients that don't have entries at the begginning of their stays + cut to 48h
##aggregation as well
if TBI_split:
   for i in range(len(batch_24h_mild)):
         batch_hourly_mild[i] = batch_hourly_mild[i].reindex(range(1, nb_hours + 1), fill_value = None) 
         batch_24h_mild[i] = batch_24h_mild[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
         batch_48h_mild[i] = batch_48h_mild[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
         
         
         batch_hourly_mild[i] = batch_hourly_mild[i].drop(columns = 'stay_id')
         batch_24h_mild[i] = batch_24h_mild[i].drop(columns = 'stay_id')
         batch_48h_mild[i] = batch_48h_mild[i].drop(columns = 'stay_id')
         batch_med_mild[i] = batch_med_mild[i].drop(columns = 'stay_id')
         batch_demographic_mild[i] = batch_demographic_mild[i].drop(columns = 'stay_id')
         batch_48h_mild[i] = batch_48h_mild[i].agg([np.mean])
         

   for i in range(len(batch_24h_severe)):
      batch_hourly_severe[i] = batch_hourly_severe[i].reindex(range(1, nb_hours + 1), fill_value = None) 
      batch_24h_severe[i] = batch_24h_severe[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
      batch_48h_severe[i] = batch_48h_severe[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
      batch_hourly_severe[i] = batch_hourly_severe[i].drop(columns = 'stay_id')
      batch_24h_severe[i] = batch_24h_severe[i].drop(columns = 'stay_id')
      batch_48h_severe[i] = batch_48h_severe[i].drop(columns = 'stay_id')
      batch_med_severe[i] = batch_med_severe[i].drop(columns = 'stay_id')
      batch_demographic_severe[i] = batch_demographic_severe[i].drop(columns = 'stay_id')
      batch_48h_severe[i] = batch_48h_severe[i].agg([np.mean])
else:
   for i in range(len(batch_24h)):
      batch_hourly[i] = batch_hourly[i].reindex(range(1, nb_hours + 1), fill_value = None) 
      batch_24h[i] = batch_24h[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
      batch_48h[i] = batch_48h[i].reindex(range(1, nb_hours//24 + 1), fill_value = None)
      
      batch_hourly[i] = batch_hourly[i].drop(columns = 'stay_id')
      batch_24h[i] = batch_24h[i].drop(columns = 'stay_id')
      batch_48h[i] = batch_48h[i].drop(columns = 'stay_id')
      batch_med[i] = batch_med[i].drop(columns = 'stay_id')
      batch_demographic[i] = batch_demographic[i].drop(columns = 'stay_id')
      batch_48h[i] = batch_48h[i].agg([np.mean])


if TBI_split:
   df_hourly_mild = pd.concat(batch_hourly_mild)
   df_24h_mild = pd.concat(batch_24h_mild)
   df_48h_mild = pd.concat(batch_48h_mild)
   df_med_mild = pd.concat(batch_med_mild)

   df_hourly_severe = pd.concat(batch_hourly_severe)
   df_24h_severe = pd.concat(batch_24h_severe)
   df_48h_severe = pd.concat(batch_48h_severe)
   df_med_severe = pd.concat(batch_med_severe)

else:
   df_hourly = pd.concat(batch_hourly)
   df_24h = pd.concat(batch_24h)
   df_48h = pd.concat(batch_48h)
   df_med = pd.concat(batch_med)

##the stay ids column are dropped since we alreasy took care of them being in the same order for all datasets
if TBI_split:
   df_demographic_mild = df_demographic[df_demographic.stay_id.isin(mild_idx)].drop(columns = 'stay_id')
   df_demographic_severe = df_demographic[df_demographic.stay_id.isin(severe_idx)].drop(columns = 'stay_id')
else:
   df_demographic = df_demographic.drop(columns = 'stay_id')

##first linear inputation and then replaced by mean when it's not possible 
##pas incroyable de recalculer le mean à chaque itération... é changer 
if TBI_split:
      for i in range(len(batch_hourly_mild)):
         batch_hourly_mild[i] = batch_hourly_mild[i].interpolate(limit = 15)
         batch_24h_mild[i] = batch_24h_mild[i].interpolate(limit = 15)
         batch_48h_mild[i] = batch_48h_mild[i].fillna(df_48h_mild.mean())
         batch_24h_mild[i] = batch_24h_mild[i].fillna(df_24h_mild.mean())
         batch_hourly_mild[i] = batch_hourly_mild[i].fillna(df_hourly_mild.mean())
         batch_demographic_mild[i].bmi = batch_demographic_mild[i].bmi.fillna(0)
         batch_demographic_mild[i].gcs = batch_demographic_mild[i].gcs.fillna(df_demographic_mild.gcs.mean())

      for i in range(len(batch_hourly_severe)):

         batch_hourly_severe[i] = batch_hourly_severe[i].interpolate(limit = 15)
         batch_24h_severe[i] = batch_24h_severe[i].interpolate(limit = 15)
         batch_48h_severe[i] = batch_48h_severe[i].fillna(df_48h_severe.mean())
         batch_24h_severe[i] = batch_24h_severe[i].fillna(df_24h_severe.mean())
         batch_hourly_severe[i] = batch_hourly_severe[i].fillna(df_hourly_severe.mean())
         batch_demographic_severe[i].bmi = batch_demographic_severe[i].bmi.fillna(0)
         batch_demographic_severe[i].gcs = batch_demographic_severe[i].gcs.fillna(df_demographic_severe.gcs.mean())

         
else:
   for i in range(len(batch_hourly)):
      batch_hourly[i] = batch_hourly[i].interpolate(limit = 15)
      batch_24h[i] = batch_24h[i].interpolate(limit = 15)
      batch_48h[i] = batch_48h[i].fillna(df_48h.mean())
      batch_24h[i] = batch_24h[i].fillna(df_24h.mean())
      batch_hourly[i] = batch_hourly[i].fillna(df_hourly.mean())
      batch_demographic[i].bmi = batch_demographic[i].bmi.fillna(0)
      batch_demographic[i].gcs = batch_demographic[i].gcs.fillna(df_demographic.gcs.mean())

#feature concatenation 
stratify_param = df_demographic.gcs
print(batch_demographic_mild[1])
if TBI_split:
      if nb_hours == 24:
         final_data_mild = np.array([[np.concatenate([np.concatenate(batch_demographic_mild[i].values), np.concatenate(batch_hourly_mild[i].values), np.concatenate(batch_24h_mild[i].values), np.concatenate(batch_med_mild[i].values)])] for i in range(len(batch_hourly_mild))])
         final_data_severe = np.array([[np.concatenate([np.concatenate(batch_demographic_severe[i].values), np.concatenate(batch_hourly_severe[i].values), np.concatenate(batch_24h_severe[i].values), np.concatenate(batch_med_severe[i].values)])] for i in range(len(batch_hourly_severe))]) 
      else: 
         final_data_mild = np.array([[np.concatenate([np.concatenate(batch_demographic_mild[i].values), np.concatenate(batch_hourly_mild[i].values), np.concatenate(batch_24h_mild[i].values), np.concatenate(batch_48h_mild[i].values), np.concatenate(batch_med_mild[i].values)])] for i in range(len(batch_hourly_mild))])
         final_data_severe = np.array([[np.concatenate([np.concatenate(batch_demographic_severe[i].values), np.concatenate(batch_hourly_severe[i].values), np.concatenate(batch_24h_severe[i].values), np.concatenate(batch_48h_severe[i].values), np.concatenate(batch_med_severe[i].values)])] for i in range(len(batch_hourly_severe))])

else:
   if nb_hours == 24:
      final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])
   else: 
      final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_48h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])


# df_24h.to_csv('df_24h.csv')
# df_48h.to_csv('df_48h.csv')
# df_hourly.to_csv('df_hourly.csv')
# df_med.to_csv('df_med.csv')
# df_demographic.to_csv('df_demographic.csv')

if TBI_split:
   final_data_mild = np.squeeze(final_data_mild)
   final_data_mild = normalize(final_data_mild)
   final_data_severe = np.squeeze(final_data_severe)
   final_data_severe = normalize(final_data_severe)
else:
   final_data = np.squeeze(final_data)
   final_data = normalize(final_data)

# feature_names = np.concatenate((pr.get_column_name(batch_demographic[1]), pr.get_column_name(batch_hourly[1]), pr.get_column_name(batch_24h[1]), pr.get_column_name(batch_48h[1]), pr.get_column_name(batch_med[1])))
# pd.DataFrame(feature_names).to_csv('features_test_to_delete.csv')

#XGBOOST MODEL 
if TBI_split:
   X_train, X_test, y_train, y_test = train_test_split(final_data_mild, labels_mild, test_size=0.2, shuffle = True, random_state=random_state)
   X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
else:
   X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
   X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

# #hyperparameter tuning 
print(final_data_mild.shape)
space_hp ={'max_depth': (hp.quniform("max_depth", 3, 18, 1)),
        'gamma': hp.uniform ('gamma', 0,9),
        'alpha' : hp.quniform('alpha', 0,180,1),
        'eta' : hp.uniform('eta', 0,0.4),

        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : (hp.quniform('min_child_weight', 0, 10, 1)),
        'n_estimators': 180,
        'seed': 0
    }

print(labels_severe.sum())
# xgboost_hyp_tuning = HypTuning(5, space_hp, X_train, y_train, X_val, y_val, n_iter = 1000, random_state =random_state)
# best_param = xgboost_hyp_tuning.hyperopt()

# final models evaluation using 5-fold cross validation  

if TBI_split:
   best_param_mild = {'alpha': 0.0, 'colsample_bytree': 0.5404929749427543, 'eta': 0.08602706957522405, 'gamma': 1.8786165006154019, 'max_depth': 17.0, 'min_child_weight': 3.0}
   best_param_severe = {'alpha': 13.0, 'colsample_bytree': 0.6268493181974919, 'eta': 0.03821439650403947, 'gamma': 0.4555031107284915, 'max_depth': 15.0, 'min_child_weight': 10.0}
else:
   if nb_hours == 24:
      best_param = {'alpha': 4.0, 'colsample_bytree': 0.6247402571982401, 'eta': 0.36200417198604184, 'gamma': 1.55002501704347, 'max_depth': 4.0, 'min_child_weight': 1.0}
   else:
      best_param = {'alpha': 0.0, 'colsample_bytree': 0.7336138546940976, 'eta': 0.3108407290269413, 'gamma': 2.3874687080350965, 'max_depth': 10.0, 'min_child_weight': 4.0}

best_param = best_param_mild
depth = best_param['max_depth']
model =  XGBClassifier(random_state = random_state, 
                        max_depth = int(depth), 
                        gamma = best_param['gamma'],
                        eta = best_param['eta'],
                        alpha = (best_param['alpha']),

                        min_child_weight=(best_param['min_child_weight']),
                        colsample_bytree=best_param['colsample_bytree'])

if TBI_split:
   eval = Evaluation(model, 'Tuned XGBoost', True, final_data_mild, labels_mild, random_state, True, features, nb_hours, 'mild')
   eval.evaluate()
   # eval = Evaluation(model, 'Tuned XGBoost', True, final_data_severe, labels_severe, random_state, True, features, nb_hours, 'severe')
   # eval.evaluate()

else:
   eval = Evaluation(model, 'Tuned XGBoost', final_data, labels, random_state, True, features, nb_hours)
   eval.evaluate()

# #save the model 
# filename = 'LOS_XGBoost_random_sampling.sav'
# pickle.dump(xgbc, open(filename, 'wb'))





