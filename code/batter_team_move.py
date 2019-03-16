'''
//*[@id="tblTradeList"]/tbody

'''

import pandas as pd
import numpy as np

from selenium import webdriver
from selenium.webdriver.support.ui import Select

import os
import time

'''
webdriver diretory
- D:/Projects/crawling/webdriver

PhantomJS diretory
- D:/Projects/crawling/phantomjs-2.1.1-windows
'''

data_dir = os.path.abspath(os.path.join(os.getcwd(), '../dataset'))

date = list()
reason = list()
team = list()
batter = list()
etc = list()

driver = webdriver.Chrome('D:/Project/Crawling/chromedriver_win32/chromedriver.exe')


driver.get('https://www.koreabaseball.com/Player/Trade.aspx')
driver.implicitly_wait(3)
select = Select(driver.find_element_by_id('selYear'))
time.sleep(0.5)

for index in range(len(select.options)):
    select.select_by_index(index)
    driver.find_element_by_xpath('//*[@id="btnSearch"]').click()
    time.sleep(0.5)
    print('='*100)
    print('Year: ',select.options[index].text)
    num_lst = list(driver.find_element_by_xpath('//*[@id="contents"]/div[2]/div[3]/span').text)

    page = 1
    left_page = True
    while left_page:
        print('page: ',page)
        if page > 1:
            num_lst = np.array(num_lst)[1::2]
        if len(num_lst) > 9:
            num_lst = np.arange(1,11)
        else:
            left_page = False

        print('num_lst: ',num_lst)
        for num in num_lst:
            print('num: ', num)
            driver.find_element_by_xpath('// *[ @ id = "contents"] / div[2] / div[3] / span / a[{}]'.format(num)).click()
            time.sleep(0.5)
            t = driver.find_element_by_xpath('//*[@id="tblTradeList"]/tbody')
            print('table: ',len(t.text.split('\n')))

            ts = t.text.split('\n')
            for line in ts:
                lst = line.split()
                if len(lst) < 4:
                    continue
                date.append(lst[0])
                reason.append(lst[1])
                team.append(lst[2])
                batter.append(lst[3])
                etc.append(lst[4] if len(lst) == 5 else np.nan)

            if num == 10 and len(t.text.split('\n')) < 20:
                left_page = False
        driver.find_element_by_xpath('//*[@id="contents"]/div[2]/div[3]/a[3]').click()
        time.sleep(0.5)
        num_lst = list(driver.find_element_by_xpath('//*[@id="contents"]/div[2]/div[3]/span').text)
        page += 1


pd.DataFrame({'DATE':date, 'REASON':reason, 'TEAM':team, 'BATTER':batter, 'ETC':etc}).to_csv('{}/batter_team_move.csv'.format(data_dir), index=False)




