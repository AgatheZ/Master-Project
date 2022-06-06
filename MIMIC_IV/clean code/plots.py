import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def ROC_plot(y_pred_proba, y, model_name):
    # Compute ROC curve and ROC area for each class
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y))]
    lr_probs = y_pred_proba
    # calculate scores
    ns_auc = roc_auc_score(y, ns_probs)
    lr_auc = roc_auc_score(y, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='{mod} (AUROC = {auc})'.format(mod = model_name , auc = round(lr_auc,3)))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.title('Length of stay prediction (> 4 days) for TBI patients (stratified sampling) \n ROC curve')
    plt.show()
