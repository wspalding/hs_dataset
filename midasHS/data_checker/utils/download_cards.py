import mechanicalsoup
import pandas as pd
# import urllib
import requests
import shutil

from data_checker.utils.misc import *

import os
import re
from os import path, stat

path = 'data_checker/static/'
file = 'data_checker/utils/' + 'data_tracker.csv'
# does filename need the name of the card????
dir_name_format = 'image_datasets/{id}_{type}_{datasets}/'
file_name_format = '{id}_{style}.{filetype}'

csv_columns = {'name':[], 'card_id':[], 'img_url':[], 'golden_img_url':[], 'card_type':[], 'normal_status':[], 'golden_status':[]}

'''
statuses:
0. found
1. downloaded normal raw
2. downloaded golden raw
3. labeled
4. cropped normal
5. cropped golden
6. completed
7. skip
'''

if os.path.exists(file):
    data = pd.read_csv(file)
else:
    data = pd.DataFrame(columns=csv_columns)

def collect_new_cards_from_hearthpwn():
    browser = mechanicalsoup.StatefulBrowser()
    start_url = 'https://www.hearthpwn.com/cards'

    if os.path.exists(file):
        data = pd.read_csv(file)
    else:
        data = pd.DataFrame(columns=csv_columns)

    print(data.tail())

    browser.open(start_url)
    page = browser.get_current_page()

    next = page.find('a', rel='next')
    i = 0
    while(next):
        print('page: ', i)
        i += 1
        page = browser.get_current_page()
        table = page.find('table', id='cards').find_all('tr')
        # print(table)
        for row in table:
            # print(row)
            h3 = row.find('h3')
            img = row.find('img')
            # print(name)
            if not h3:
                continue
            # print(name)
            '''
            stuff to check no missing or duplicate ids
            if len(card_id) <= 0:
                print('found example with no id', h3)
                continue
            if card_id[0] in found:
                print('found duplicate', card_id)
            found.add(card_id[0])
            '''
            card_id = int(re.findall(r'\d+', h3.find('a')['href'])[0])

            if data.isin({'card_id': [card_id]})['card_id'].any():
                continue

            name = h3.string
            img_url = img['data-imageurl']
            gold_img_url = img['data-animationurl']
            card_type = row.find('ul').find('li').find('a').string

            if img_url and card_type != 'NONE':
                in_normal_dataset = 'found'
            else:
                in_normal_dataset = 'skip'

            if gold_img_url and card_type != 'NONE':
                in_golden_dataset = 'found'
            else:
                in_golden_dataset = 'skip'

            # print(card_type)
            row = {
                'name':name,
                'card_id':card_id,
                'img_url':img_url,
                'golden_img_url':gold_img_url,
                'card_type':card_type,
                'normal_status':in_normal_dataset,
                'golden_status':in_golden_dataset,
            }
            data = append_row_to_df(row, data)

        next = page.find('a', rel='next')
        if next:
            browser.follow_link(next)
    # print(data.head())
    data.to_csv(file, index=False)




def download_new_normal_cards_from_hearthpwn():
    # file = 'data_checker/utils/card_links.csv'
    # data = pd.read_csv(file)
    total = len(data.index)
    for index, row in data.iterrows():
        print_status_bar(index, total, row['name'])

        if row['normal_status'] != 'found':
            continue

        temp_file_dir = path + 'temp_dir/'
        if not os.path.exists(temp_file_dir):
            os.makedirs(temp_file_dir)

        file_dir = path + dir_name_format.format(id=row['card_id'],
                                            type=row['card_type'],
                                            datasets='')

        file_name = file_name_format.format(id=row['card_id'],
                                                style='normal',
                                                filetype='png')
        made_change = False
        normal_url = row['img_url']
        # print(file_dir+filename, normal_url)
        status = download_from_url_to_file(normal_url, temp_file_dir + file_name)
        if status != 200:
            data.loc[index, 'normal_status'] = 'Error {}'.format(status)
            # made_change = True
        else:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            os.rename(temp_file_dir + file_name, file_dir + file_name)
            data.loc[index, 'normal_status'] = 'downloaded'
        shutil.rmtree(temp_file_dir)

        data.to_csv(file, index=False)


def download_new_golden_cards_from_hearthpwn():
    # file = 'data_checker/utils/card_links.csv'
    # data = pd.read_csv(file)
    total = len(data.index)
    for index, row in data.iterrows():
        print_status_bar(index, total, row['name'])

        if row['golden_status'] != 'found':
            continue

        temp_file_dir = path + 'temp_dir/'
        if not os.path.exists(temp_file_dir):
            os.makedirs(temp_file_dir)

        file_dir = path + dir_name_format.format(id=row['card_id'],
                                            type=row['card_type'],
                                            datasets='')

        file_name = file_name_format.format(id=row['card_id'],
                                                style='golden',
                                                filetype='webm')
        made_change = False
        golden_url = row['golden_img_url']

        status = download_from_url_to_file(golden_url, temp_file_dir + file_name)
        if status != 200:
            print(status, row['card_id'])
        shutil.rmtree(temp_file_dir)

        if made_change:
            data.to_csv(file, index=False)
