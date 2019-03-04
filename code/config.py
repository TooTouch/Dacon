class Config:
    def __init__(self):
        self.target = 'OPS'
        self.target2 = ['OBP','SLG']
        self.features = ['year','batter_name','height/weight','year_born','starting_salary','SLG','OBP','OPS','AB']
        self.cat_features = ['position','team']
        self.params = {
                    "learning_rate": 0.1,
                    "n_estimators": 10000,
                    "max_depth": 3,
                    "min_child_weight": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "colsample_bylevel": 1.0,
                    "alpha": 0,
                    "lambda": 1,
                    "objective": "gpu:reg:linear",
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "eval_metric":"rmse"
                }
