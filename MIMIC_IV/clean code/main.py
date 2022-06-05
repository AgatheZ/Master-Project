import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import seaborn as sns
import preprocessing as pr
import torch 
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

##Variables 
nb_hours = 48

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
df_hourly = pr.trunc_length(df_hourly, 48)
df_24h = pr.trunc_length(df_24h, 2)
df_demographic, df_med, df_hourly, df_24h, df_48h = pr.arrange_ids(df_demographic, df_med, df_hourly, df_24h, df_48h)

##label extraction 
labels = df_demographic.pop('los')
labels[labels < 4] = 0
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

#feature concatenation 
final_data = np.array([[np.concatenate([np.concatenate(batch_demographic[i].values), np.concatenate(batch_hourly[i].values), np.concatenate(batch_24h[i].values), np.concatenate(batch_48h[i].values), np.concatenate(batch_med[i].values)])] for i in range(len(batch_hourly))])
final_data = np.squeeze(final_data).astype('float64')
final_data = normalize(final_data)

#dataset split 
X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)

# - cross validataion
scores = cross_val_score(xgbc, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

y_pred = xgbc.predict(X_val)
y_pred_proba = xgbc.predict_proba(X_val)
cm = confusion_matrix(y_val,y_pred)
f1 = f1_score(y_val, y_pred)
print(cm, f1)

#AUROC 
# Compute ROC curve and ROC area for each class
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_val))]
lr_probs = y_pred_proba
# calculate scores
ns_auc = roc_auc_score(y_pred_proba, ns_probs)
lr_auc = roc_auc_score(y_pred_proba, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_pred_proba, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_pred_proba, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
