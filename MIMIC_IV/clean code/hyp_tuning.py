from hyperopt import fmin, tpe, hp, anneal, Trials
from xgboost import XGBClassifier

class HypTuning:
    def __init__(self, cv, space, X_train, y_train, X_test, y_test):
        self.cv = cv
        self.space = space
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

# space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
#         'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
#         'n_estimators': 180,
#         'seed': 0
#     }

def objective(self):
    clf=XGBClassifier(
                    n_estimators = self.space['n_estimators'], max_depth = int(self.space['max_depth']), gamma = self.space['gamma'],
                    reg_alpha = int(self.space['reg_alpha']),min_child_weight=int(self.space['min_child_weight']),
                    colsample_bytree=int(self.space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }   