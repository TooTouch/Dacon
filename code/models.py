import xgboost as xgb
import numpy as np
from utils import *

class Xgboost_Run:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)

    def run(self, train, val, test):
        self.model.fit(train[0], train[1],
                          eval_set=[(train[0], train[1]), (val[0], val[1])],
                          early_stopping_rounds=100,
                          verbose=50)
        prob = self.model.predict(test[0], ntree_limit=self.model.best_iteration)
        e = rmse(test[1], prob)
        print('test RMSE: {0:.6f}'.format(e))

        return prob


