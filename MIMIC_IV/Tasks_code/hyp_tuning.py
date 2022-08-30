from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error,  mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import lightgbm as lgb
import math
import warnings
warnings.filterwarnings("ignore")

class HypTuning:
    '''Class for Hyperparameter tuning using Hyperopt'''
    def __init__(self, reg, cv, model_name, X_train, y_train, X_val, y_val, n_iter, random_state):
        self.cv = cv
        self.reg = reg
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_iter = n_iter
        self.random_state = random_state
        if self.model_name == 'XGBoost':
            self.space = {'max_depth': (hp.quniform("max_depth", 3, 18, 1)),
                            'gamma': hp.uniform ('gamma', 0,9),
                            'alpha' : hp.quniform('alpha', 0,180,1),
                            'eta' : hp.uniform('eta', 0,0.4),

                            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                            'min_child_weight' : (hp.quniform('min_child_weight', 0, 10, 1)),
                            'n_estimators': 180,
                            'seed': 0
                        }
        if self.model_name == 'RF':
            self.space = { 
                'n_estimators': hp.quniform('n_estimators', 100, 300, 1),
                'max_features': hp.choice('max_features', ['auto']),
                'max_depth' : hp.quniform('max_depth', 4, 20, 1),
                'criterion' : hp.choice('criterion', ["squared_error"])
            }

        if self.model_name == 'LightGBM':
            self.space = {
        'max_depth': (hp.quniform("max_depth", 3, 18, 1)),
        'boosting_type': 
                                   'gbdt',
        'num_leaves': hp.quniform('num_leaves', 30, 50, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
        'reg_alpha': hp.uniform('reg_alpha', 0,3),
        'reg_lambda': hp.uniform('reg_lambda',0,3),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'min_child_weight': hp.uniform('min_child_weight', 1, 5), 
    }

    def objective(self, space):
        if self.reg:
            if self.model_name == 'XGBoost':
                clf=XGBRegressor(random_state = self.random_state,
                                n_estimators = space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                                alpha = int(space['alpha']), min_child_weight=int(space['min_child_weight']),
                                colsample_bytree=(space['colsample_bytree']), eta = (space['eta']))
                evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
                clf.fit(self.X_train, self.y_train,
                    eval_set=evaluation, eval_metric="rmse", verbose=False)
            
            if self.model_name == 'RF':
                clf = RandomForestRegressor(random_state = self.random_state, n_estimators = int(space['n_estimators']), 
                max_features = space['max_features'], max_depth = int(space['max_depth']), criterion = space['criterion'])
            
                evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
                clf.fit(self.X_train, self.y_train)
            
            if self.model_name == 'LightGBM':
                clf = lgb.LGBMRegressor(boosting_type=space['boosting_type'], num_leaves=int(space['num_leaves']), 
                max_depth= int(space['max_depth']), learning_rate=space['learning_rate'], reg_alpha=space['reg_alpha'], 
                reg_lambda = space['reg_lambda'], colsample_bytree= space['colsample_bytree'], min_child_weight = space['min_child_weight'])
                evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
                clf.fit(self.X_train, self.y_train,
                    eval_set=evaluation, eval_metric="auc", verbose=False)
        
            pred = clf.predict(self.X_val)

            mae = mean_absolute_error(self.y_val, pred)
            rmse = math.sqrt(mean_squared_error(self.y_val, pred ))
            print ("mae:", mae)
            return {'loss': rmse, 'status': STATUS_OK}
        else:
            if self.model_name == 'XGBoost':
                clf=XGBClassifier(random_state = self.random_state,
                                n_estimators = space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                                alpha = int(space['alpha']), min_child_weight=int(space['min_child_weight']),
                                colsample_bytree=(space['colsample_bytree']), eta = (space['eta']))
                evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
                clf.fit(self.X_train, self.y_train,
                    eval_set=evaluation, eval_metric="auc", verbose=False)
            
            if self.model_name == 'RF':
                clf = RandomForestClassifier(random_state = self.random_state, n_estimators = space['n_estimators'], 
                max_features = space['max_features'], max_depth = int(space['max_depth']), criterion = space['criterion'])
            
                evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
                clf.fit(self.X_train, self.y_train)
            
            if self.model_name == 'LightGBM':
                clf = lgb.LGBMClassifier(boosting_type=space['boosting_type'], num_leaves=int(space['num_leaves']), 
                max_depth= int(space['max_depth']), learning_rate=space['learning_rate'], reg_alpha=space['reg_alpha'], 
                reg_lambda = space['reg_lambda'], colsample_bytree= space['colsample_bytree'], min_child_weight = space['min_child_weight'])
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