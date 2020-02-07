# import mechanicalsoup
import pandas as pd
# import urllib
import requests
import shutil

from os import path, stat



file = 'card_links.csv'
data = pd.read_csv(file)
# print(data.head())

total = len(data.index)

for index, row in data.iterrows():
    # print(row)
    percent_complete = index/total * 100
    bar_len = int(percent_complete/5)
    loading_bar = '█'*bar_len + '-'*(20-bar_len)
    print('\r {}/{} {} {:.4f}% complete, current: {} '.format(index, total, loading_bar, percent_complete, row['name']), end='\r')
    # print(index,":",row['name'], row['img_url'], row['golden_img_url'])

    name = "unknown_{}".format(index)
    if not pd.isna(row['name']):
        name = row['name']

    file_name_golden = 'golden_cards/{}_{}_golden.gif'.format(index, name)
    file_name = 'normal_cards/{}_{}_normal.png'.format(index, name)

    made_change = False

    if not row['have_normal']:
        if not pd.isna(row['img_url']):
            # print(row['img_url'])
            r = requests.get(row['img_url'], stream=True)
            if r.status_code == 200:
                with open(file_name, 'wb+') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
                    data.at[index, 'have_normal'] = True
                    made_change = True

        # else:
        #     print('already have', row['img_url'])

    if not row['have_golden']:
        if not pd.isna(row['golden_img_url']):
            # print(row['golden_img_url'])
            g = requests.get(row['golden_img_url'], stream=True)
            if g.status_code == 200:
                with open(file_name_golden, 'wb+') as f:
                    g.raw.decode_content = True
                    shutil.copyfileobj(g.raw, f)
                    data.at[index, 'have_golden'] = True
                    made_change = True
        # else:
        #     print('already have', row['golden_img_url'])

    if made_change:
        data.to_csv(file, index=False)
    # print('done')



# for index, row in data.iterrows():
#     print(row['name'], row['img_url'], row['golden_img_url'])
