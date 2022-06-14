from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import warnings

class HypTuning:
    def __init__(self, cv, space, X_train, y_train, X_val, y_val, n_iter, random_state):
        self.cv = cv
        self.space = space
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_iter = n_iter
        self.random_state = random_state



    def objective(self, space):
        clf=XGBClassifier(random_state = self.random_state,
                        n_estimators = space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                        alpha = int(space['alpha']), min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=(space['colsample_bytree']), eta = (space['eta']))
        
        evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
        
        clf.fit(self.X_train, self.y_train,
                eval_set=evaluation, eval_metric="auc", verbose=False)
        

        pred = clf.predict(self.X_val)
        pred_proba = clf.predict_proba(self.X_val)[:,1]
        accuracy = accuracy_score(self.y_val, pred>0.5)
        auroc = roc_auc_score(self.y_val, pred_proba )
        print ("accuracy:", accuracy)
        return {'loss': -auroc, 'status': STATUS_OK}

    def optimisation(self):
        
        trials = Trials()
        objective = self.objective
        best_hyperparams = fmin(fn = objective,
                                space = self.space,
                                algo = tpe.suggest,
                                max_evals = self.n_iter,
                                trials = trials)   
        print("The best hyperparameters are : ","\n")
        print(best_hyperparams)

    def hyperopt(self):
        return self.optimisation()
    
    def random_search(self):
        param_grid_rand = self.space
        model = XGBClassifier()
        rs=RandomizedSearchCV(model, param_grid_rand, n_iter = self.n_iter, scoring='roc_auc', 
                n_jobs=-1, cv=self.cv, verbose=True, random_state=self.random_state)

        rs.fit(self.X_train, self.y_train)

        # rs_test_score = roc_auc_score(self.y_val, rs.predict(self.X_val))

        print("Best AUROC {:.3f} params {}".format(-rs.best_score_, rs.best_params_))

        return rs.best_params_