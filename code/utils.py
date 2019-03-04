import numpy as np

def wrmse(y_true, y_pred, AB):
    return np.sqrt(np.sum((np.power(y_true - y_pred, 2)) * AB)/np.sum(AB))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))