from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

class Evaluation:
    def __init__(self, model, model_name, X_test, y_test):
        self.model = model
        self.model_name = model_name
        self.X_test = X_test
        self.y_test = y_test
    
    def ROC_plot(self, y_pred_proba):
    # Compute ROC curve and ROC area for each class
    # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(self.y_test))]
        lr_probs = y_pred_proba
        # calculate scores
        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs)

        # # summarize scores
        # print('No Skill: ROC AUC=%.3f' % (ns_auc))
        # print('Logistic: ROC AUC=%.3f' % (lr_auc))
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='{mod} (AUROC = {auc})'.format(mod = self.model_name , auc = round(lr_auc,3)))
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.title('Length of stay prediction (> 4 days) for TBI patients \n ROC curve - 24h')
        plt.show()

    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        y_pred_proba = y_pred_proba[:,1]

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print('Accuracy: %.3f' % (acc))
        print('F1-Score: %.3f' % (f1))
        plot_confusion_matrix(self.model, self.X_test, self.y_test)
        plt.figure()
        self.ROC_plot(y_pred_proba)
