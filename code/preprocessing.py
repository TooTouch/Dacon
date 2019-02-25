import pandas as pd
import numpy as np

def dummy_feature(df, variables):
    dummies = pd.DataFrame()
    for var in variables:
        dummy = pd.get_dummies(df[var], prefix=var, drop_first=True)
        dummies = pd.concat([dummies, dummy], axis=1)
        df = df.drop(var, axis=1)
    df = pd.concat([df, dummies], axis=1)
    return df


def preprocessing_data(data):
    names = data.batter_name.unique()
    cum_season = pd.Series(np.zeros(data.shape[0]))
    data.loc[:,'cum_season'] = cum_season
    for name in names:
        data.loc[data.batter_name == name, 'cum_season'] = np.arange(data[data.batter_name == name].shape[0])

    data.loc[:,'height'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[0][:3]).astype(float)
    data.loc[:,'weight'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[1][:-2]).astype(float)
    data.loc[:,'age'] = data.year - data.year_born.apply(lambda x: int(x[:4]))
    data.starting_salary = data.starting_salary.apply(lambda x: x[:-2] if str(x) != 'nan' else np.nan).astype(float)

    data = dummy_feature(data, ['team', 'position'])
    data = data.drop(['year_born', 'height/weight', 'batter_name'], axis=1)

    data.starting_salary = data.starting_salary.fillna(np.median(data.starting_salary.dropna()))
    data.height = data.height.fillna(np.median(data.height.dropna()))
    data.weight = data.weight.fillna(np.median(data.weight.dropna()))
    data = data.fillna(0)

    return data


def split_data(data, features, target):
    data = data[features]

    data = preprocessing_data(data)

    x_data = data.drop(target, axis=1)
    y_data = data[target]

    x_train = x_data[(x_data.year != 2018) & (x_data.year != 2017)]
    x_valid = x_data[x_data.year == 2017]
    x_test = x_data[x_data.year == 2018]

    y_train = y_data[(x_data.year != 2018) & (x_data.year != 2017)]
    y_val = y_data[x_data.year == 2017]
    y_test = y_data[x_data.year == 2018]

    x_train = x_train.drop('year', axis=1)
    x_val = x_valid.drop('year', axis=1)
    x_test = x_test.drop('year', axis=1)

    return x_train, y_train, x_val, y_val, x_test, y_test