# import mechanicalsoup
import pandas as pd
import numpy as np
# import cv2
from PIL import Image, GifImagePlugin
import matplotlib.pyplot as plt
# import urllib
import requests
import shutil
import sys

from os import path, stat
import os

def show_image(f):
    im = Image.open(f, 'r')
    plt.imshow(im)
    plt.show()
    return im

def close_image(im):
    im.close()


file = 'card_links.csv'
data = pd.read_csv(file)
# print(data.head())

# data['skip'] = ''
bad_data = [0,3149,3,4,5,11,18,19,20,21,22,24,27,30,31,45,46,50,53,54,55,56,58,60,61,67,70,72,73,74,75,76,78,79,110,112,113,114,118,121,122,123,126]
bad_data = []

for index, row in data.iterrows():
    # print(row)
    if not pd.isna(row['skip']):
        continue
    if not row['have_golden']:
        continue

    # if row['is_cropped']: 7
    #     continue
    #
    # percent_complete = index/total * 100
    # bar_len = int(percent_complete/5)
    # loading_bar = '█'*bar_len + '-'*(20-bar_len)
    # print('\r {}/{} {} {:.4f}% complete, current: {} '.format(index, total, loading_bar, percent_complete, row['name']), end='\r')
    # print(index,":",row['name'], row['img_url'], row['golden_img_url'])

    name = "unknown_{}".format(index)
    if not pd.isna(row['name']):
        name = row['name']

    bad_chars = '*.\"\\\/[]:;|,!?'
    for c in bad_chars:
        name = name.replace(c,'')

    file_name_golden = 'golden_cards/{}_{}_golden.gif'.format(index, name)
    file_name = 'normal_cards/{}_{}_normal.png'.format(index, name)

    # display golden card and take input on if it is bad
    print(name, index)
    # im = show_image(file_name_golden)
    skip = input()
    if skip == "n":
        print('adding to bad data')
        bad_data.append(index)
        data.at[index,'skip'] = True
    else:
        data.at[index,'skip'] = False
    # close_image(im)

    # if index > 10:
    #     break


# print(data.head())
#
# for i in bad_data:
#     data.at[i,'skip'] = True




data.to_csv(file, index=False)
