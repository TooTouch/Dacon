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
features_ops = config.features.copy()
features_ops.append(config.target)
print('features_ops: ',features_ops)

# data split
x_train, y_train, x_val, y_val, x_test, y_test = split_data(data_regular, features_ops, config.target)

# model
xgboost = Xgboost_Run(config.params)
print('## OPS')
prob_ops = xgboost.run([x_train, y_train], [x_val, y_val], [x_test, y_test])


if config.target2:
    # features
    features_obp = config.features.copy()
    features_slg = config.features.copy()
    features_obp.append(config.target2[0])
    features_slg.append(config.target2[1])
    print('features_obp: ', features_obp)
    print('features_slg: ', features_slg)

    # data split
    x_train_obp, y_train_obp, x_val_obp, y_val_obp, x_test_obp, y_test_obp = split_data(data_regular, features_obp, config.target2[0])
    x_train_slg, y_train_slg, x_val_slg, y_val_slg, x_test_slg, y_test_slg = split_data(data_regular, features_slg, config.target2[1])

    # model
    xgboost_obp = Xgboost_Run(config.params)
    xgboost_slg = Xgboost_Run(config.params)
    print('## OBP')
    prob_obp = xgboost_obp.run([x_train_obp, y_train_obp], [x_val_obp, y_val_obp], [x_test_obp, y_test_obp])
    print('## SLG')
    prob_slg = xgboost_slg.run([x_train_slg, y_train_slg], [x_val_slg, y_val_slg], [x_test_slg, y_test_slg])

    prob = prob_obp + prob_slg
    e = rmse(y_test, prob)
    print('## OBP + SLG')
    print('test RMSE: {0:.6f}'.format(e))
