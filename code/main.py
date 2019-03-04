import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')

from preprocessing import *
from config import *
from models import *
from utils import *

# data
data_regular = pd.read_csv('../dataset/Regular_Season_Batter.csv', index_col=0)
data_regular = data_regular[data_regular.year >= 2001]

config = Config()
print('features: ',config.features)
print('categorical features: ',config.cat_features)
print('target: ',config.target)

# data split
x_train, y_train, x_val, y_val, x_test, y_test = split_data(data_regular, config.features, config.cat_features, config.target)

# model
xgboost = Xgboost_Run(config.params)
print('## OPS')
prob_ops = xgboost.run([x_train, y_train], [x_val, y_val], [x_test, y_test])


if config.target2:
    # features
    print('target2: ', config.target2)

    # data split
    x_train_obp, y_train_obp, x_val_obp, y_val_obp, x_test_obp, y_test_obp = split_data(data_regular, config.features, config.cat_features, config.target2[0])
    x_train_slg, y_train_slg, x_val_slg, y_val_slg, x_test_slg, y_test_slg = split_data(data_regular, config.features, config.cat_features, config.target2[1])

    # model
    xgboost_obp = Xgboost_Run(config.params)
    xgboost_slg = Xgboost_Run(config.params)
    print('## OBP')
    prob_obp = xgboost_obp.run([x_train_obp, y_train_obp], [x_val_obp, y_val_obp], [x_test_obp, y_test_obp])
    print('## SLG')
    prob_slg = xgboost_slg.run([x_train_slg, y_train_slg], [x_val_slg, y_val_slg], [x_test_slg, y_test_slg])

    prob = prob_obp + prob_slg
    e = wrmse(y_test.iloc[:,0], prob, y_test.iloc[:,1])
    e2 = rmse(y_test.iloc[:, 0], prob)
    print('## OBP + SLG')
    print('test WRMSE: {0:.6f}'.format(e))
    print('test RMSE: {0:.6f}'.format(e2))
