import pandas as pd
import numpy as np

from selenium import webdriver

import os
import time

data_dir = os.path.abspath(os.path.join(os.getcwd(), '../dataset'))
regular = pd.read_csv('{}/Regular_Season_Batter.csv'.format(data_dir))

regular_name = regular[regular['height/weight'].isnull()].batter_name.unique()

# batter height / weight dataframe
batter_df = pd.DataFrame({'batter_name':regular_name})


'''
webdriver diretory
- D:/Projects/crawling/webdriver

PhantomJS diretory
- D:/Projects/crawling/phantomjs-2.1.1-windows
'''

driver = webdriver.Chrome('D:/Project/Crawling/chromedriver_win32/chromedriver.exe')

for name in batter_df.batter_name:
    driver.get('https://www.koreabaseball.com/Player/Search.aspx')
    driver.implicitly_wait(3)
    driver.find_element_by_name('ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$txtSearchPlayerName').send_keys(name)
    driver.find_element_by_xpath('//*[@class="compare"]/input').click()
    time.sleep(0.5)
    check = driver.find_element_by_xpath('//*[@id="cphContents_cphContents_cphContents_udpRecord"]/div[2]/p/strong/span')
    if int(check.text) > 0:
        height_weight = driver.find_element_by_xpath('//*[@id="cphContents_cphContents_cphContents_udpRecord"]/div[2]/table/tbody/tr/td[6]')
        h_w_lst = height_weight.text.split(',')
        batter_df.loc[batter_df.batter_name == name,'height'] = h_w_lst[0].strip()[:-2]
        batter_df.loc[batter_df.batter_name == name,'weight'] = h_w_lst[1].strip()[:-2]
        print('{} 건 / name: {} / height: {}/weight: {}'.format(check.text, name, h_w_lst[0], h_w_lst[1]))

batter_df.to_csv('{}/batter_height_weight.csv'.format(data_dir), index=False)