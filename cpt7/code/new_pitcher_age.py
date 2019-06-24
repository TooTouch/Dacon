import pandas as pd
import numpy as np

from selenium import webdriver

import os
import time

data_dir = os.path.abspath(os.path.join(os.getcwd(), '../dataset'))
pitcher_19 = pd.read_csv('{}/fangraphs_foreigners_2019.csv'.format(data_dir))

pitcher_name_19 = pitcher_19.pitcher_name.unique()

# batter height / weight dataframe
pitcher_df = pd.DataFrame()


'''
webdriver diretory
- D:/Projects/crawling/webdriver

PhantomJS diretory
- D:/Projects/crawling/phantomjs-2.1.1-windows
'''

driver = webdriver.Chrome('D:/Project/Crawling/chromedriver_win32/chromedriver.exe')
driver.get('http://www.statiz.co.kr/main.php')

for name in pitcher_name_19:
    driver.implicitly_wait(1)
    driver.find_element_by_xpath('//*[@id="search_text"]').send_keys(name)
    driver.find_element_by_xpath('//*[@id="search-btn"]').click()
    time.sleep(0.5)
    if name == '켈리':
        driver.find_element_by_xpath('/html/body/div/div[1]/div/section[2]/div/div[2]/div/div/div/div[2]/table/tbody/tr[3]/td[2]/a').click()
    time.sleep(0.5)
    driver.find_element_by_xpath('/html/body/div/div[1]/div/section[2]/div/div[1]/div/div[3]/div/div[2]/table/tbody/tr/td/a[2]').click()
    time.sleep(0.5)

    features = {1:'연도', 3:'나이'}
    table = driver.find_element_by_xpath('/ html / body / div / div[1] / div / section[2] / div / div[2] / div / div[3] / div / div / table / tbody')
    row_count = len(table.find_elements_by_tag_name('tr')) - 5
    print(row_count)

    value = np.zeros((row_count, len(list(features.keys()))))
    df = pd.DataFrame(value, columns=list(features.values()))
    for row in range(row_count):
        for i in features.keys():
            df.loc[row,features[i]] = driver.find_element_by_xpath('/ html / body / div / div[1] / div / section[2] / div / div[2] / div / div[3] / div / div / table / tbody / tr[{}] / td[{}]'.format(row+3, i)).text
    df.loc[:, '이름'] = name
    print(df)

    pitcher_df = pd.concat([pitcher_df, df], axis=0)

pitcher_df.to_csv('../dataset/new_pitcher_19.csv',index=False, encoding='cp949')

