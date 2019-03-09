import pandas as pd
import numpy as np
import os

from tqdm import tqdm

def dummy_feature(df, variables):
    dummies = pd.DataFrame()
    for var in variables:
        dummy = pd.get_dummies(df[var], prefix=var, drop_first=True)
        dummies = pd.concat([dummies, dummy], axis=1)
    df = pd.concat([df, dummies], axis=1)
    return df

def split_data(data, config, target):
    feature = list(set(config.features + config.cat_features + config.lag_features))
    data = data[feature]

    names = data.batter_name.unique()


    for f in tqdm(range(len(config.lag_features))):
        data.loc[:,'lag_1_' + config.lag_features[f]] = data.apply(lambda x: lag_n(data, x['batter_name'], x['year'], config.lag_features[f]), axis=1)
        # data.loc[:,'lag_2_' + config.lag_features[f]] = data.apply(lambda x: lag_n(data, x['batter_name'], x['year'], config.lag_features[f], lag_num=2), axis=1)
    print('## lag_n')

    for f in tqdm(range(len(config.total_features))):
        data.loc[:,'total_' + config.total_features[f]] = data.apply(lambda x: get_total(data, x['batter_name'], x['year'], config.total_features[f]), axis=1)
    print('## total')

    data = get_luck(data)
    config.lag_features.append(['1B', '1b_luck', '2b_luck', '3b_luck'])
    print('## luck')

    # data = split_position(data)
    # print('## split_position')
    data = weight_height_(data)
    print('## height/weight')
    data = age_(data)
    print('## age')
    data = starting_salary_(data)
    print('## starting_salary')
    data = cum_season_(data, names)
    print('## cum_season')

    data = OPS_trend_(data, names)
    print('## OPS trend')
    data = OBP_trend_(data, names)
    print('## OBP trend')
    data = SLG_trend_(data, names)
    print('## SLG trend')
    data = OPS_up_down(data, names)
    print('## OPS_up_down')
    data = OBP_up_down(data, names)
    print('## OBP_up_down')
    data = SLG_up_down(data, names)
    print('## SLG_up_down')
    # data = pre_data_trend(data, names, ['avg','AB','OBP','SLG','OPS'])
    # print('## pre_data_trend')
    # data = pre_data(data, names, ['avg','AB','OBP','SLG','OPS'])
    # print('## pre_data_trend')
    # data = era_by_team(data)
    # print('## era_by_team')
    data = grad_status_(data)
    print('## grad_status')
    data = career_count(data)
    print('## career_count')
    data = from_inter_(data, config.international)
    print('## from inter')

    data = categorical_variables(data, ['position'])
    # print('## dummy variables')
    if target == 'OPS':
        data.OPS = data.OPS.fillna(0)
    else:
        data[target] = data[target].fillna(0)
    data = data.fillna(999)

    drop_features = list(set(['batter_name','position','height/weight','year_born','team','career'] + config.lag_features))
    x_data = data.drop(drop_features, axis=1)
    print(x_data.info())
    y_data = data[[target, 'AB', 'G']]

    print('x_data features: ',x_data.columns)

    x_train = x_data[x_data.year <= 2016]
    x_valid = x_data[x_data.year == 2017]
    x_test = x_data[x_data.year == 2018]

    y_train = y_data[x_data.year <= 2016]
    y_val = y_data[x_data.year == 2017]
    y_test = y_data[x_data.year == 2018]

    x_train = x_train.drop('year', axis=1)
    x_val = x_valid.drop('year', axis=1)
    x_test = x_test.drop('year', axis=1)


    return x_train, y_train, x_val, y_val, x_test, y_test

def weight_height_(data):
    data.loc[:,'height'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[0][:3]).astype(float)
    data.loc[:,'weight'] = data['height/weight'].apply(lambda x: np.nan if str(x) == 'nan' else x.split('/')[1][:-2]).astype(float)
    data.height = data.height.fillna(np.median(data.height.dropna()))
    data.weight = data.weight.fillna(np.median(data.weight.dropna()))
    return data


def age_(data):
    data.loc[:, 'age'] = data.year - data.year_born.apply(lambda x: int(x[:4]))
    return data

def starting_salary_(data):
    data.loc[:,'starting_salary'] = data.starting_salary.apply(lambda x: x[:-2] if str(x)!='nan' else np.nan).astype(float)
    data.loc[:,'starting_salary'] = data.starting_salary.fillna(np.median(data.starting_salary.dropna()))
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
        year_diff = data_batter.year.iloc[-1] - data_batter.year.iloc[0]
        OPS_trend = list()
        # 전전 시즌과 전시즌 데이터가 있어야 이전 추세를 구할 수 있기때문에 시즌참여횟수가 최소 3회는 되어야함
        if data_batter.shape[0] < 3:
            OPS_trend = 0
        else:
            for y in range(data_batter.shape[0]):
                data_ops = data_batter.iloc[:y+1,:]
                OPS_weight = 0
                # 위와 마찬가지 다만 시즌참여 횟수가 3회이상인 경우 누점참여수가 2회까지는 모두 0
                if data_ops.shape[0]!=3:
                    for i in range(2, data_ops.shape[0]):
                        # 전시즌과 전전시즌의 OPS가 0이면 pass
                        if (data_ops.OPS.iloc[i-1] == 0) | (data_ops.OPS.iloc[i-2] == 0):
                            continue
                        else:
                            ops_rate = (data_ops.OPS.iloc[i-1]/data_ops.OPS.iloc[i-2])
                            if ops_rate < 1:
                                ops_rate = -ops_rate
                            OPS_weight += ((data_ops.year.iloc[i]-data_ops.year.iloc[i-1])/year_diff) * ops_rate
                OPS_trend.append(np.round(OPS_weight,3))
        data.loc[data.batter_name==name,'OPS_trend'] = OPS_trend
    return data

def OBP_trend_(data, names):
    for name in names:
        data_batter = data[data.batter_name==name][['year','OBP']]
        year_diff = data_batter.year.iloc[-1] - data_batter.year.iloc[0]
        OBP_trend = list()
        # 전전 시즌과 전시즌 데이터가 있어야 이전 추세를 구할 수 있기때문에 시즌참여횟수가 최소 3회는 되어야함
        if data_batter.shape[0] < 3:
            OBP_trend = 0
        else:
            for y in range(data_batter.shape[0]):
                data_OBP = data_batter.iloc[:y+1,:]
                OBP_weight = 0
                # 위와 마찬가지 다만 시즌참여 횟수가 3회이상인 경우 누점참여수가 2회까지는 모두 0
                if data_OBP.shape[0]!=3:
                    for i in range(2, data_OBP.shape[0]):
                        # 전시즌과 전전시즌의 OBP가 0이면 pass
                        if (data_OBP.OBP.iloc[i-1] == 0) | (data_OBP.OBP.iloc[i-2] == 0):
                            continue
                        else:
                            OBP_rate = (data_OBP.OBP.iloc[i-1]/data_OBP.OBP.iloc[i-2])
                            if OBP_rate < 1:
                                OBP_rate = -OBP_rate
                            OBP_weight += ((data_OBP.year.iloc[i]-data_OBP.year.iloc[i-1])/year_diff) * OBP_rate
                OBP_trend.append(np.round(OBP_weight,3))
        data.loc[data.batter_name==name,'OBP_trend'] = OBP_trend
    return data

def SLG_trend_(data, names):
    for name in names:
        data_batter = data[data.batter_name==name][['year','SLG']]
        year_diff = data_batter.year.iloc[-1] - data_batter.year.iloc[0]
        SLG_trend = list()
        # 전전 시즌과 전시즌 데이터가 있어야 이전 추세를 구할 수 있기때문에 시즌참여횟수가 최소 3회는 되어야함
        if data_batter.shape[0] < 3:
            SLG_trend = 0
        else:
            for y in range(data_batter.shape[0]):
                data_SLG = data_batter.iloc[:y+1,:]
                SLG_weight = 0
                # 위와 마찬가지 다만 시즌참여 횟수가 3회이상인 경우 누점참여수가 2회까지는 모두 0
                if data_SLG.shape[0]!=3:
                    for i in range(2, data_SLG.shape[0]):
                        # 전시즌과 전전시즌의 SLG가 0이면 pass
                        if (data_SLG.SLG.iloc[i-1] == 0) | (data_SLG.SLG.iloc[i-2] == 0):
                            continue
                        else:
                            SLG_rate = (data_SLG.SLG.iloc[i-1]/data_SLG.SLG.iloc[i-2])
                            if SLG_rate < 1:
                                SLG_rate = -SLG_rate
                            SLG_weight += ((data_SLG.year.iloc[i]-data_SLG.year.iloc[i-1])/year_diff) * SLG_rate
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
            # 시즌 누점 참여횟수가 3회가 안되면 역대시즌의 상승 수 하락 수를 구할 수 없음
            if data_.shape[0] < 3:
                data.loc[(data.batter_name == name) & (data.year == year), 'OPS_up'] = OPS_up
                data.loc[(data.batter_name == name) & (data.year == year), 'OPS_down'] = OPS_down
                continue
            for j in range(data_.shape[0]):
                if j < 3:
                    continue
                pre_OPS = [ops for ops in data_.iloc[:j - 1, :]['OPS'] if ops != 0]
                if (len(pre_OPS) == 0) | (data_.iloc[j - 1, :]['OPS'] == 0):
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
            # 시즌 누점 참여횟수가 3회가 안되면 역대시즌의 상승 수 하락 수를 구할 수 없음
            if data_.shape[0] < 3:
                data.loc[(data.batter_name == name) & (data.year == year), 'OBP_up'] = OBP_up
                data.loc[(data.batter_name == name) & (data.year == year), 'OBP_down'] = OBP_down
                continue
            for j in range(data_.shape[0]):
                if j < 3:
                    continue
                pre_OBP = [OBP for OBP in data_.iloc[:j - 1, :]['OBP'] if OBP != 0]
                if (len(pre_OBP) == 0) | (data_.iloc[j - 1, :]['OBP'] == 0):
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
            # 시즌 누점 참여횟수가 3회가 안되면 역대시즌의 상승 수 하락 수를 구할 수 없음
            if data_.shape[0] < 3:
                data.loc[(data.batter_name == name) & (data.year == year), 'SLG_up'] = SLG_up
                data.loc[(data.batter_name == name) & (data.year == year), 'SLG_down'] = SLG_down
                continue
            for j in range(data_.shape[0]):
                if j < 3:
                    continue
                pre_SLG = [SLG for SLG in data_.iloc[:j - 1, :]['SLG'] if SLG != 0]
                if (len(pre_SLG) == 0) | (data_.iloc[j - 1, :]['SLG'] == 0):
                    continue
                up_down = data_.iloc[j, :]['SLG'] / pre_SLG[-1]
                if up_down > 1:
                    SLG_up += 1
                elif up_down < 1:
                    SLG_down += 1

            data.loc[(data.batter_name == name) & (data.year == year), 'SLG_up'] = SLG_up
            data.loc[(data.batter_name == name) & (data.year == year), 'SLG_down'] = SLG_down

    return data


def era_by_team(data):
    data_dir = os.path.abspath(os.path.join(os.getcwd(),'../dataset'))
    era = pd.read_excel('{}/era_by_team_and_season.xlsx'.format(data_dir))
    era.columns = ['year', 'team', 'team_era', 'rank', 'team_n', 'rating']
    era.rating = era.rating.apply(lambda x: float(x.split('/')[0]) / float(x.split('/')[1]))
    era.year = era.year + 1  # 이전 시즌 데이터를 합쳐야해서 + 1
    data = pd.merge(data, era[['team', 'year', 'team_era','rating']], on=['team', 'year'], how='left')

    return data


def career_count(data):
    data.loc[:, 'career_count'] = data.career.apply(lambda x: len(x.split('-')))

    return data



def grad_status_(data):
    def university(careers):
        for career in careers:
            if (career != '현대') and ('대' in career[-1]):
                return career
    def grad_univ(careers, univ):
        y = 0
        for career in careers:
            if career in univ:
                y += 1
        if y == 0:
            return 'high'
        else:
            return 'univ'

    univ = data.career.apply(lambda x: university(x.split('-'))).unique()
    data.loc[:,'grad_status'] = data.career.apply(lambda x: grad_univ(x.split('-'), univ))
    return data


def grad_status_(data):
    def university(careers):
        for career in careers:
            if (career != '현대') and ('대' in career[-1]):
                return career

    def grad_univ(careers, univ):
        y = 0
        for career in careers:
            if career in univ:
                y += 1
        if y == 0:
            return 0 # 고졸
        else:
            return 1 # 대졸

    univ = data.career.apply(lambda x: university(x.split('-'))).unique()
    data.loc[:,'grad_status'] = data.career.apply(lambda x: grad_univ(x.split('-'), univ))
    return data


def split_position(data):
    position1 = data.position.apply(lambda x: x[:x.index('(')] if str(x)!='nan' else x)
    position2 = data.position.apply(lambda x: x[x.index(')')-2:x.index(')')] if str(x)!='nan' else x)
    data.loc[:,'position1'] = position1
    data.loc[:,'position2'] = position2

    return data


def from_inter_(data, international):
    def career_inter(careers, international):
        inter_y = 0
        for career in careers:
            if career in international:
                inter_y += 1
        if inter_y == 0:
            return 0  # 국내파
        else:
            return 1  # 국제파

    data.loc[:, 'from_inter'] = data.career.apply(lambda x: career_inter(x.split('-'), international))

    return data


def pre_data(data, names, features):
    pre_data = pd.DataFrame()
    for name in names:
        data_name = data[data.batter_name==name][['batter_name','year'] + features].copy()
        data_name.iloc[1:,2:] = data_name.iloc[:-1,2:]
        data_name.iloc[0,2:] = 0
        new_colnames = list()
        for feature in features:
            new_colnames.append('pre_{}'.format(feature))
        data_name.columns = ['batter_name','year'] + new_colnames
        pre_data = pd.concat([pre_data,data_name], axis=0)
    data = pd.merge(data, pre_data, on=['batter_name','year'], how='left')
    return data

def pre_data_trend(data, names, features):
    for feature in features:
        for name in names:
            data_batter = data[data.batter_name==name][['year',feature]]
            year_diff = data_batter.year.iloc[-1] - data_batter.year.iloc[0]
            pre_trend = list()
            # 전전 시즌과 전시즌 데이터가 있어야 이전 추세를 구할 수 있기때문에 시즌참여횟수가 최소 3회는 되어야함
            if data_batter.shape[0] < 3:
                pre_trend = 0
            else:
                for y in range(data_batter.shape[0]):
                    data_pre = data_batter.iloc[:y+1,:]
                    pre_weight = 0
                    # 위와 마찬가지 다만 시즌참여 횟수가 3회이상인 경우 누점참여수가 2회까지는 모두 0
                    if data_pre.shape[0]!=3:
                        for i in range(2, data_pre.shape[0]):
                            # 전시즌과 전전시즌의 pre가 0이면 pass
                            if (data_pre[feature].iloc[i-1] == 0) | (data_pre[feature].iloc[i-2] == 0):
                                continue
                            else:
                                pre_rate = (data_pre[feature].iloc[i-1]/data_pre[feature].iloc[i-2])
                                if pre_rate < 1:
                                    pre_rate = -pre_rate
                                pre_weight += ((data_pre.year.iloc[i]-data_pre.year.iloc[i-1])/year_diff) * pre_rate
                    pre_trend.append(np.round(pre_weight,3))
            data.loc[data.batter_name==name,'pre_{}_trend'.format(feature)] = pre_trend
    return data

def lag_n(data,name,year,var_name,lag_num=1):
    if len(data.loc[(data['batter_name']==name)&(data['year']==year-lag_num),var_name])==0:
        return np.nan
    else:
        return data.loc[(data['batter_name']==name)&(data['year']==year-lag_num),var_name].iloc[0]

def get_total(data, name, year, feature):
    if (len(data.loc[(data['batter_name']==name)&(data['year']<year),feature])!=0):
        return data.loc[(data['batter_name']==name)&(data['year']<year),feature].sum()
    else:
        return np.nan


def get_luck(data):
    data.loc[:,'1B'] = data['H'] - data['2B'] - data['3B'] - data['HR']
    data.loc[:,'1b_luck'] = data['1B'] / (data['AB'] - data['HR'] - data['SO'])
    data.loc[:,'2b_luck'] = data['2B'] / (data['AB'] - data['HR'] - data['SO'])
    data.loc[:,'3b_luck'] = data['3B'] / (data['AB'] - data['HR'] - data['SO'])
    return data