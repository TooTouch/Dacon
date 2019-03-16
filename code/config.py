class Config:
    def __init__(self):
        self.target = 'OPS'
        self.target2 = ['OBP','SLG']
        self.features = ['year','batter_name','height/weight','year_born','starting_salary','position','career','SLG','OBP','OPS','AB']
        self.cat_features = ['team']
        self.international = ['Selma(고)', '쿠바 Ciego de Avila Maximo Gomez Baez(대)', '캐나다 A.B Lucas Secondary(고)', '필라델피아',
                                '히로시마', '일본 아세아대', '샌프란시스코', '미국 윌리캐넌초', '미국 쿠퍼고', '미국 페퍼다인대',
                                 '미네소타','볼티모어', '미국 Catawba(대)', '미국 Creighton(대)', '미국 Diamond Bar(고)',
                                '미국 Fort Loramie(고)', '미국 Kentucky(대)', '미국 Las Vegas(대)','미국 Smithfield', '미국 Texas at Arlington(대)', '미국 Toledo(대)',
                                '미국 Wabash Valley(대)', '미국 레이노사고', '미국 볼주립대', '미국 위스콘신 라크로스대',
                                '도미니카', '도미니카 Elias Rodriguez(고)', '도미니카 산토도밍고고', '도미니카 알레한드로 바쓰고',
                                '도미니카 엘세이보고', '네덜란드 Voorben Praktyk(고)']
        self.lag_features = ['avg', 'G', 'AB', 'R', 'H', '2B', '3B',
                               'HR', 'TB', 'RBI', 'BB', 'SO', 'GDP', 'SLG', 'OBP','SB', 'CS','HBP','E' ,
                               'OPS']
        self.total_features = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'BB', 'SO','SB', 'CS','HBP','E' ]
        self.params = {
                    "learning_rate": 0.1,
                    "n_estimators": 10000,
                    "max_depth": 4,
                    "min_child_weight": 4,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "colsample_bylevel": 1.0,
                    "alpha": 0,
                    "lambda": 1,
                    "missing": 999,
                    "objective": "gpu:reg:linear",
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "eval_metric":"rmse"
                }
