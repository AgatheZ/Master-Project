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

from xgboost import XGBClassifier 



class Evaluation:
    def __init__(self, reg = True, model = XGBClassifier(), model_name = 'XGBoost', X = None, y = None, random_state = 1, SHAP = False, feature_names = None, nb_hours = 24, severity = '', threshold = 4):
        self.model = model
        self.reg = reg
        self.model_name = model_name
        self.X = X
        self.y = y
        self.random_state = random_state
        self.SHAP = SHAP
        self.feature_names = feature_names
        self.nb_hours = nb_hours
        self.severity = severity
        self.threshold = threshold


    
    def ROC_plot(self, rocs, fprs, tprs):
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
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        plt.plot(mean_fprs, mean_tprs, color='b', marker='.',
         label='{mod} \nAveraged AUROC (5-folds) = {auc} ± {std})'.format(mod = self.model_name , auc = round(mean_rocs,3), std = round(std_roc,3)),
         lw=2, alpha=.8)

        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        plt.fill_between(mean_fprs, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ standard deviation - 5-folds')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.title('Length of stay prediction (> %i days) for %s TBI patients \n ROC curve - %.0fh'%(self.threshold, self.severity, self.nb_hours))
        plt.show()


    def evaluate(self):
        accs = []
        f1s = []
        rocs = []
        tprs = []
        fprs = []
        shaps_values = list()
        test_idx = list()
        mean_fpr = np.linspace(0, 1, 50)
        skf = StratifiedKFold(n_splits=5, random_state= self.random_state, shuffle=True)
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            xgbc = self.model
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
            
            if self.SHAP:
                ex = shap.Explainer(xgbc)
                shaps_values = ex.shap_values(X_test)
                plt.figure(figsize = (15,15))
                shap.summary_plot(shaps_values, pd.DataFrame(X_test, columns = self.feature_names), show = False)

                plt.savefig('SHAP_LOS_{}.png'.format(self.threshold),bbox_inches='tight', dpi=300)
                self.SHAP = False

        print('Averaged accuracy (5-folds): %.3f ±  %.3f' % (np.mean(accs), statistics.stdev(accs)))
        print('Averaged f1-Score (5-folds): %.3f ±  %.3f' % (np.mean(f1s), statistics.stdev(f1s)))
        print('Averaged AUROC (5-folds): %.3f ±  %.3f' % (np.mean(rocs), statistics.stdev(rocs)))


        plt.figure()
        self.ROC_plot(rocs, fprs, tprs)

    def evaluate_regression(self):
        rmse = []
        mae = []
        error = []
        shaps_values = list()
        test_idx = list()
        mean_fpr = np.linspace(0, 1, 50)
        skf = KFold(n_splits=5, random_state= self.random_state, shuffle=True)
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            xgbc = self.model
            xgbc.fit(X_train, y_train)
            y_pred = xgbc.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mean_abs = mean_absolute_error(y_test, y_pred)
            er = y_pred  - y_test
            error.append(er)
            rmse.append(math.sqrt(mse))
            mae.append(mean_abs)
            
            if self.SHAP:
                ex = shap.Explainer(xgbc)
                shaps_values = ex.shap_values((X_test))
                shap.summary_plot(shaps_values, pd.DataFrame(X_test, columns = self.feature_names))
                self.SHAP = False

        print('Averaged RMSE (5-folds): %.3f ±  %.3f' % (np.mean(rmse), statistics.stdev(rmse)))
        print('Averaged MAE (5-folds): %.3f ±  %.3f' % (np.mean(mae), statistics.stdev(mae)))
        print('Averaged error (5-folds): %.3f ±  %.3f' % (np.mean(np.concatenate(error)), statistics.stdev(np.concatenate(error))))

