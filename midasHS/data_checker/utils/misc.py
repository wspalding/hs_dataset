import pandas as pd
import re
import os
import requests
import shutil


path = 'data_checker/static/'
file = 'data_checker/utils/' + 'data_tracker.csv'
# does filename need the name of the card????
dir_name_format = 'image_datasets/{id}_{type}_{datasets}/'

def get_name(index, row):
    name = "unknown_{}".format(index)
    if not pd.isna(row['name']):
        name = row['name']

    bad_chars = '*.\"\\\/[]:;|,!?'
    for c in bad_chars:
        name = name.replace(c,'')
    return name

def get_file_names(index, name, path=''):
    file_name = path + 'normal_cards/{}_{}_normal.png'.format(index, name)
    file_name_golden = path + 'golden_cards/{}_{}_golden.gif'.format(index, name)
    return file_name, file_name_golden

def get_card_id_from_url(url):
    id = re.findall(r'\d+', url)[-1]
    return id


def append_row_to_df(row, df):
    df = df.append(row, ignore_index=True)
    return df


def print_status_bar(curr, total, curr_name):
    percent_complete = curr/total * 100
    bar_len = int(percent_complete/5)
    loading_bar = '█'*bar_len + '-'*(20-bar_len)
    print('\r {}/{} {} {:.4f}% complete, current: {} '.format(curr, total, loading_bar, percent_complete, curr_name), end='\r')


def download_from_url_to_file(url, file):
    g = requests.get(url, stream=True)
    if g.status_code == 200:
        with open(file, 'wb+') as f:
            g.raw.decode_content = True
            shutil.copyfileobj(g.raw, f)
    return g.status_code

def add_dir_to_normal_dataset(id, type, df):
    in_normal_dataset = df.loc[df['card_id'] == int(id), 'normal_status'].values[0]
    if in_normal_dataset == 'cropped':
        in_normal_dataset = 'n'
    else:
        in_normal_dataset = ''
    in_golden_dataset = df.loc[df['card_id'] == int(id), 'golden_status'].values[0]
    if in_golden_dataset == 'cropped':
        in_golden_dataset = 'g'
    else:
        in_golden_dataset = ''
    df.loc[df['card_id'] == int(id), 'normal_status'] = 'cropped'
    old = path + dir_name_format.format(id=id,
                                        type=type,
                                        datasets=in_normal_dataset + in_golden_dataset)
    new = path + dir_name_format.format(id=id,
                                        type=type,
                                        datasets='n' + in_golden_dataset)
    os.rename(old, new)
