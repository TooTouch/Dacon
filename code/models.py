import xgboost as xgb
import numpy as np
from utils import *

class Xgboost_Run:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)

    def run(self, train, val, test, target):
        self.model.fit(train[0], train[1][target],
                          eval_set=[(train[0], train[1][target]), (val[0], val[1][target])],
                          early_stopping_rounds=100,
                          sample_weight=train[1]['AB'],
                          verbose=50)
        prob = self.model.predict(test[0], ntree_limit=self.model.best_iteration)
        e = wrmse(test[1].iloc[:,0], prob, test[1].iloc[:,1])
        e2 = rmse(test[1].iloc[:, 0], prob)
        print('test WRMSE: {0:.6f}'.format(e))
        print('test RMSE: {0:.6f}'.format(e2))

        return prob


