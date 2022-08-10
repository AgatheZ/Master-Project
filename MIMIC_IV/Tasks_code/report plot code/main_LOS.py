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
from xgboost import XGBClassifier
from evaluation import Evaluation
from preprocessing import Preprocessing
from hyp_tuning import HypTuning
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix, mean_absolute_error,  mean_squared_error
import statistics
import numpy as np
import pandas as pd
import shap
import math

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter

warnings.filterwarnings("ignore")

##Variables 
nb_hours = 24
random_state = 1
TBI_split = True
tuning = False
SHAP = False
imputation = 'linear'
model_name = 'XGBoost'
threshold = 4

assert model_name in ['RF', 'XGBoost', 'LightGBM', 'Stacking'], "Please specify a valid model name"
assert imputation in ['No', 'carry_forward', 'linear', 'multivariate'], "Please specify a valid imputation method"

##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour_std.csv', delimiter=',').sort_values(by=['stay_id'])
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',').sort_values(by=['stay_id'])
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',').sort_values(by=['stay_id'])
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',').sort_values(by=['stay_id'])
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',').sort_values(by=['stay_id'])
features = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\Master-Project\MIMIC_IV\resources\features.csv', header = None)
diag = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\mimiciv_diag.csv')



print('Data Loading - done')

if nb_hours == 24:
   features = features.loc[:224,2] 
else:
   features = features.loc[:,0]

fig, ax = plt.subplots()
model_name = 'Tuned XGBoost'
def ROC_plot(rocs, fprs, tprs, col):
    # Compute ROC curve and ROC area for each class
    # generate a no skill prediction (majority class)
        mean_rocs = np.mean(rocs)
        mean_fprs = np.linspace(0, 1, 50)
        mean_tprs = np.mean(tprs, axis = 0)
        mean_tprs[-1] = 1.0

        std_tprs = np.std(tprs, axis = 0)
        std_roc = np.std(rocs)

        # calculate roc curves
        # plot the roc curve for the model
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=.8)
        plt.plot(mean_fprs, mean_tprs, color=col, marker='.',
         label='{mod} \nAveraged AUROC (5-folds) = {auc} ± {std}'.format(mod = model_name , auc = round(mean_rocs,3), std = round(std_roc,3)),
         lw=2, alpha=.8)

        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        plt.fill_between(mean_fprs, tprs_lower, tprs_upper, color=col, alpha=.1,
                 label=r'$\pm$ standard deviation - 5-folds')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
#Preprocessing
pr_24 = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, 48, TBI_split, random_state, imputation, diag)

data_mild, data_severe, labels_mild, labels_severe = pr_24.preprocess_data(threshold)

final_data = data_mild
labels = labels_mild
print(len(labels))
X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)



#hyp tuning
# xgboost_hyp_tuning = HypTuning(False, 5, 'XGBoost', X_train, y_train, X_val, y_val, n_iter = 1000, random_state = random_state)
# best_param = xgboost_hyp_tuning.hyperopt()

best_param_mild = {'alpha': 0.0, 'colsample_bytree': 0.7944489162174009, 'eta': 0.03783520485150966, 'gamma': 0.9245079076024777, 'max_depth': 15.0, 'min_child_weight': 0.0}
best_param_severe = {'alpha': 0.0, 'colsample_bytree': 0.5566668059186075, 'eta': 0.252886125308837, 'gamma': 1.888659757012347, 'max_depth': 12.0, 'min_child_weight': 7.0}
best_param = {'alpha': 12.0, 'colsample_bytree': 0.5265548343516888, 'eta': 0.10955899500196617, 'gamma': 3.7037045423905743, 'max_depth': 15, 'min_child_weight': 8.0}


best_param = best_param
depth = best_param['max_depth']
model = XGBClassifier(
                     random_state = random_state, 
                        max_depth = int(depth), 
                        gamma = best_param['gamma'],
                        eta = best_param['eta'],
                        alpha = (best_param['alpha']),
                        n_estimators=500,
                        min_child_weight=(best_param['min_child_weight']),
                        colsample_bytree=best_param['colsample_bytree'])


accs = []
f1s = []
rocs = []
tprs = []
fprs = []
X = final_data
y = labels
shaps_values = list()
test_idx = list()
mean_fpr = np.linspace(0, 1, 50)
skf = StratifiedKFold(n_splits=2, random_state= random_state, shuffle=True)
for train_index, test_index in skf.split(X, y):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   xgbc = model
   xgbc.fit(X_train, y_train)
   y_pred = xgbc.predict(X_test)
   y_pred_proba = xgbc.predict_proba(X_test)
   y_pred_proba = y_pred_proba[:,1]
   fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
   accuracy = accuracy_score(y_test, y_pred)
   auroc = roc_auc_score(y_test, y_pred_proba)
   f1 = f1_score(y_test, y_pred)
   print('Accuracy:', accuracy)
   print('f1:', f1)

   accs.append(accuracy)
   f1s.append(f1)
   rocs.append(auroc)
   fprs.append(fpr)
   tprs.append(np.interp(mean_fpr, fpr, tpr))
   tprs[-1][0] = 0.0
   ex = shap.Explainer(xgbc)
   shaps_values = ex.shap_values(X_test)
   plt.figure(figsize = (15,15))
   shap.summary_plot(shaps_values, pd.DataFrame(X_test, columns = pr_24.features), show = True)

   # plt.savefig('SHAP_LOS_{}.png'.format(threshold),bbox_inches='tight', dpi=300)



print('Averaged accuracy (5-folds): %.3f ±  %.3f' % (np.mean(accs), statistics.stdev(accs)))
print('Averaged f1-Score (5-folds): %.3f ±  %.3f' % (np.mean(f1s), statistics.stdev(f1s)))
print('Averaged AUROC (5-folds): %.3f ±  %.3f' % (np.mean(rocs), statistics.stdev(rocs)))
# model_name = 'Tuned XGBoost mild/moderate'
# ROC_plot(rocs, fprs, tprs, 'red')


# final_data = data_severe
# labels = labels_severe
# print(len(labels))
# X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
# X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)



# #hyp tuning
# # xgboost_hyp_tuning = HypTuning(False, 5, 'XGBoost', X_train, y_train, X_val, y_val, n_iter = 1000, random_state = random_state)
# # best_param = xgboost_hyp_tuning.hyperopt()

# best_param_mild = {'alpha': 0.0, 'colsample_bytree': 0.7944489162174009, 'eta': 0.03783520485150966, 'gamma': 0.9245079076024777, 'max_depth': 15.0, 'min_child_weight': 0.0}
# best_param_severe = {'alpha': 0.0, 'colsample_bytree': 0.5566668059186075, 'eta': 0.252886125308837, 'gamma': 1.888659757012347, 'max_depth': 12.0, 'min_child_weight': 7.0}
# best_param = {'alpha': 12.0, 'colsample_bytree': 0.5265548343516888, 'eta': 0.10955899500196617, 'gamma': 3.7037045423905743, 'max_depth': 15, 'min_child_weight': 8.0}


# # best_param = best_param_mild 
# depth = best_param['max_depth']
# model = XGBClassifier(
#                      random_state = random_state, 
#                         max_depth = int(depth), 
#                         gamma = best_param['gamma'],
#                         eta = best_param['eta'],
#                         alpha = (best_param['alpha']),
#                         n_estimators=500,
#                         min_child_weight=(best_param['min_child_weight']),
#                         colsample_bytree=best_param['colsample_bytree'])


# best_param = best_param_severe
# depth = best_param['max_depth']
# model = XGBClassifier(
#                      random_state = random_state, 
#                         max_depth = int(depth), 
#                         gamma = best_param['gamma'],
#                         eta = best_param['eta'],
#                         alpha = (best_param['alpha']),
#                         n_estimators=500,
#                         min_child_weight=(best_param['min_child_weight']),
#                         colsample_bytree=best_param['colsample_bytree'])
# accs = []
# f1s = []
# rocs = []
# tprs = []
# fprs = []
# X = final_data
# y = labels
# shaps_values = list()
# test_idx = list()
# mean_fpr = np.linspace(0, 1, 50)
# skf = StratifiedKFold(n_splits=5, random_state= random_state, shuffle=True)
# for train_index, test_index in skf.split(X, y):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    xgbc = model
#    xgbc.fit(X_train, y_train)
#    y_pred = xgbc.predict(X_test)
#    y_pred_proba = xgbc.predict_proba(X_test)
#    y_pred_proba = y_pred_proba[:,1]
#    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#    accuracy = accuracy_score(y_test, y_pred)
#    auroc = roc_auc_score(y_test, y_pred_proba)
#    f1 = f1_score(y_test, y_pred)
#    print('Accuracy:', accuracy)
#    print('f1:', f1)

#    accs.append(accuracy)
#    f1s.append(f1)
#    rocs.append(auroc)
#    fprs.append(fpr)
#    tprs.append(np.interp(mean_fpr, fpr, tpr))
#    tprs[-1][0] = 0.0
#    ex = shap.Explainer(xgbc)
#    shaps_values = ex.shap_values(X_test)
#    plt.figure(figsize = (15,15))
#    shap.summary_plot(shaps_values, pd.DataFrame(X_test, columns = pr_24.features), show = True)

#    plt.savefig('SHAP_LOS_{}.png'.format(threshold),bbox_inches='tight', dpi=300)



# print('Averaged accuracy (5-folds): %.3f ±  %.3f' % (np.mean(accs), statistics.stdev(accs)))
# print('Averaged f1-Score (5-folds): %.3f ±  %.3f' % (np.mean(f1s), statistics.stdev(f1s)))
# print('Averaged AUROC (5-folds): %.3f ±  %.3f' % (np.mean(rocs), statistics.stdev(rocs)))
# model_name = 'Tuned XGBoost severe'
# ROC_plot(rocs, fprs, tprs, 'orange')

# # df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour_std.csv', delimiter=',').sort_values(by=['stay_id'])
# # df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',').sort_values(by=['stay_id'])
# # df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',').sort_values(by=['stay_id'])
# # df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',').sort_values(by=['stay_id'])
# # df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',').sort_values(by=['stay_id'])
# # features = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\Master-Project\MIMIC_IV\resources\features.csv', header = None)
# # diag = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\mimiciv_diag.csv')
# # pr_48 = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, 48, TBI_split, random_state, imputation, diag)


# # final_data, labels = pr_48.preprocess_data(threshold)
# # X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
# # X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

# model = XGBClassifier(random_state=random_state)

# accs = []
# f1s = []
# rocs = []
# tprs = []
# fprs = []
# shaps_values = list()
# test_idx = list()
# mean_fpr = np.linspace(0, 1, 50)
# X = final_data
# y = labels
# skf = KFold(n_splits=5, random_state= random_state, shuffle=True)
# for train_index, test_index in skf.split(X, y):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    xgbc = model
#    xgbc.fit(X_train, y_train)
#    y_pred = xgbc.predict(X_test)
#    y_pred_proba = xgbc.predict_proba(X_test)
#    y_pred_proba = y_pred_proba[:,1]
#    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#    accuracy = accuracy_score(y_test, y_pred)
#    auroc = roc_auc_score(y_test, y_pred_proba)
#    f1 = f1_score(y_test, y_pred)
#    print('Accuracy:', accuracy)
#    print('f1:', f1)

#    accs.append(accuracy)
#    f1s.append(f1)
#    rocs.append(auroc)
#    fprs.append(fpr)
#    tprs.append(np.interp(mean_fpr, fpr, tpr))
#    tprs[-1][0] = 0.0



# print('Averaged accuracy (5-folds): %.3f ±  %.3f' % (np.mean(accs), statistics.stdev(accs)))
# print('Averaged f1-Score (5-folds): %.3f ±  %.3f' % (np.mean(f1s), statistics.stdev(f1s)))
# print('Averaged AUROC (5-folds): %.3f ±  %.3f' % (np.mean(rocs), statistics.stdev(rocs)))
# model_name = 'RF 48h'

# ROC_plot(rocs, fprs, tprs, 'orange')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
# #save the model 
# filename = 'LOS_XGBoost_random_sampling.sav'
# pickle.dump(xgbc, open(filename, 'wb'))





