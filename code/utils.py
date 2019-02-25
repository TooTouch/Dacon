import numpy as np

def wrmse(y_true, y_pred, AB):
    pass

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))