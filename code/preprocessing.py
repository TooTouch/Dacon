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

def split_data(data, features, cat_features, target):
    data = data[features + cat_features]

    names = data.batter_name.unique()

    data = categorical_variables(data, cat_features)
    data = weight_height_(data)
    data = age_(data)
    data = starting_salary_(data)
    data = cum_season_(data, names)
    data = OPS_nan_(data)

    data = OPS_trend_(data, names)
    data = OBP_trend_(data, names)
    data = SLG_trend_(data, names)
    data = OPS_up_down(data, names)
    data = OBP_up_down(data, names)
    data = SLG_up_down(data, names)

    x_data = data.drop(['batter_name','OBP','SLG','OPS','AB'], axis=1)
    y_data = data[[target, 'AB']]

    print('x_data features: ',x_data.columns)

    x_train = x_data[(x_data.year != 2018) & (x_data.year != 2017)]
    x_valid = x_data[x_data.year == 2017]
    x_test = x_data[x_data.year == 2018]

    y_train = y_data[(x_data.year != 2018) & (x_data.year != 2017)]
    y_val = y_data[x_data.year == 2017]
    y_test = y_data[x_data.year == 2018]

    x_train = x_train.drop('year', axis=1)
    x_val = x_valid.drop('year', axis=1)
    x_test = x_test.drop('year', axis=1)

    y_train.drop('AB',axis=1)
    y_val.drop('AB', axis=1)

    y_train = y_train[target]
    y_val = y_val[target]

    return x_train, y_train, x_val, y_val, x_test, y_test

def drop_variables(data, features):
    data = data.drop(features, axis=1)
    return data

def OPS_nan_(data):
    data = data.fillna(0)
    return data

def weight_height_(data):
    data['height'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[0][:3]).astype(float)
    data['weight'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[1][:-2]).astype(float)
    data.height = data.height.fillna(np.median(data.height.dropna()))
    data.weight = data.weight.fillna(np.median(data.weight.dropna()))
    data = data.drop('height/weight', axis=1)
    return data


def age_(data):
    data['age'] = data.year - data.year_born.apply(lambda x: int(x[:4]))
    data = data.drop('year_born', axis=1)
    return data

def starting_salary_(data):
    data.starting_salary = data.starting_salary.apply(lambda x: x[:-2] if str(x)!='nan' else np.nan).astype(float)
    data.starting_salary = data.starting_salary.fillna(np.median(data.starting_salary.dropna()))
    return data

def categorical_variables(data, features):
    data = dummy_feature(data, features)
    return data

def cum_season_(data, names):
    for name in names:
        data.loc[data.batter_name==name,'cum_season'] = np.arange(data[data.batter_name==name].shape[0])
    return data

def OPS_trend_(data, names):
    for name in names:
        data_batter = data[data.batter_name==name][['year','OPS']]
        OPS_trend = list()
        if data_batter.shape[0] == 1:
            OPS_trend.append(0)
        else:
            for y in range(data_batter.shape[0]):
                data_ops = data_batter.iloc[:y+1,:]
                OPS_weight = 0
                if data_ops.shape[0]!=1:
                    for i in range(data_ops.shape[0]):
                        if i == 0:
                            continue
                        if data_ops.OPS.iloc[i] == 0:
                            continue
                        elif data_ops.OPS.iloc[i-1] == 0:
                            continue
                        else:
                            ops_rate = (data_ops.OPS.iloc[i]/data_ops.OPS.iloc[i-1])
                            if ops_rate < 1:
                                ops_rate = -ops_rate
                            OPS_weight += (data_ops.year.iloc[i]/max(data_ops.year)) * ops_rate
                OPS_trend.append(np.round(OPS_weight,3))
        data.loc[data.batter_name==name,'OPS_trend'] = OPS_trend
    return data

def OBP_trend_(data, names):
    for name in names:
        data_batter = data[data.batter_name==name][['year','OBP']]
        OBP_trend = list()
        if data_batter.shape[0] == 1:
            OBP_trend.append(0)
        else:
            for y in range(data_batter.shape[0]):
                data_OBP = data_batter.iloc[:y+1,:]
                OBP_weight = 0
                if data_OBP.shape[0]!=1:
                    for i in range(data_OBP.shape[0]):
                        if i == 0:
                            continue
                        if data_OBP.OBP.iloc[i] == 0:
                            continue
                        elif data_OBP.OBP.iloc[i-1] == 0:
                            continue
                        else:
                            OBP_rate = (data_OBP.OBP.iloc[i]/data_OBP.OBP.iloc[i-1])
                            if OBP_rate < 1:
                                OBP_rate = -OBP_rate
                            OBP_weight += (data_OBP.year.iloc[i]/max(data_OBP.year)) * OBP_rate
                OBP_trend.append(np.round(OBP_weight,3))
        data.loc[data.batter_name==name,'OBP_trend'] = OBP_trend
    return data

def SLG_trend_(data, names):
    for name in names:
        data_batter = data[data.batter_name==name][['year','SLG']]
        SLG_trend = list()
        if data_batter.shape[0] == 1:
            SLG_trend.append(0)
        else:
            for y in range(data_batter.shape[0]):
                data_SLG = data_batter.iloc[:y+1,:]
                SLG_weight = 0
                if data_SLG.shape[0]!=1:
                    for i in range(data_SLG.shape[0]):
                        if i == 0:
                            continue
                        if data_SLG.SLG.iloc[i] == 0:
                            continue
                        elif data_SLG.SLG.iloc[i-1] == 0:
                            continue
                        else:
                            SLG_rate = (data_SLG.SLG.iloc[i]/data_SLG.SLG.iloc[i-1])
                            if SLG_rate < 1:
                                SLG_rate = -SLG_rate
                            SLG_weight += (data_SLG.year.iloc[i]/max(data_SLG.year)) * SLG_rate
                SLG_trend.append(np.round(SLG_weight,3))
        data.loc[data.batter_name==name,'SLG_trend'] = SLG_trend
    return data

def OPS_up_down(data, names):
    for name in names:
        data_name = data[data.batter_name == name]
        for i in range(data_name.shape[0]):
            OPS_up, OPS_down = 0, 0
            year = data_name.year.iloc[i]
            data_ = data_name.iloc[:i + 1, :]
            if data_.shape[0] == 1:
                data.loc[(data.batter_name == name) & (data.year == year), 'OPS_up'] = OPS_up
                data.loc[(data.batter_name == name) & (data.year == year), 'OPS_down'] = OPS_down
                continue
            for j in range(data_.shape[0]):
                pre_OPS = [ops for ops in data_.iloc[:j, :]['OPS'] if ops != 0]
                if j == 0:
                    continue
                elif data_.iloc[j, :]['OPS'] == 0:
                    continue
                elif len(pre_OPS) == 0:
                    continue
                up_down = data_.iloc[j, :]['OPS'] / pre_OPS[-1]
                if up_down > 1:
                    OPS_up += 1
                elif up_down < 1:
                    OPS_down += 1

            data.loc[(data.batter_name == name) & (data.year == year), 'OPS_up'] = OPS_up
            data.loc[(data.batter_name == name) & (data.year == year), 'OPS_down'] = OPS_down

    return data


def OBP_up_down(data, names):
    for name in names:
        data_name = data[data.batter_name == name]
        for i in range(data_name.shape[0]):
            OBP_up, OBP_down = 0, 0
            year = data_name.year.iloc[i]
            data_ = data_name.iloc[:i + 1, :]
            if data_.shape[0] == 1:
                data.loc[(data.batter_name == name) & (data.year == year), 'OBP_up'] = OBP_up
                data.loc[(data.batter_name == name) & (data.year == year), 'OBP_down'] = OBP_down
                continue
            for j in range(data_.shape[0]):
                pre_OBP = [OBP for OBP in data_.iloc[:j, :]['OBP'] if OBP != 0]
                if j == 0:
                    continue
                elif data_.iloc[j, :]['OBP'] == 0:
                    continue
                elif len(pre_OBP) == 0:
                    continue
                up_down = data_.iloc[j, :]['OBP'] / pre_OBP[-1]
                if up_down > 1:
                    OBP_up += 1
                elif up_down < 1:
                    OBP_down += 1

            data.loc[(data.batter_name == name) & (data.year == year), 'OBP_up'] = OBP_up
            data.loc[(data.batter_name == name) & (data.year == year), 'OBP_down'] = OBP_down

    return data


def SLG_up_down(data, names):
    for name in names:
        data_name = data[data.batter_name == name]
        for i in range(data_name.shape[0]):
            SLG_up, SLG_down = 0, 0
            year = data_name.year.iloc[i]
            data_ = data_name.iloc[:i + 1, :]
            if data_.shape[0] == 1:
                data.loc[(data.batter_name == name) & (data.year == year), 'SLG_up'] = SLG_up
                data.loc[(data.batter_name == name) & (data.year == year), 'SLG_down'] = SLG_down
                continue
            for j in range(data_.shape[0]):
                pre_SLG = [SLG for SLG in data_.iloc[:j, :]['SLG'] if SLG != 0]
                if j == 0:
                    continue
                elif data_.iloc[j, :]['SLG'] == 0:
                    continue
                elif len(pre_SLG) == 0:
                    continue
                up_down = data_.iloc[j, :]['SLG'] / pre_SLG[-1]
                if up_down > 1:
                    SLG_up += 1
                elif up_down < 1:
                    SLG_down += 1

            data.loc[(data.batter_name == name) & (data.year == year), 'SLG_up'] = SLG_up
            data.loc[(data.batter_name == name) & (data.year == year), 'SLG_down'] = SLG_down

    return data